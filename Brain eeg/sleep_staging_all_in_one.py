#!/usr/bin/env python3
"""2026 睡眠分期挑战赛：单文件离线 CPU 推理程序。

功能：EDF 读取、330 维手工特征、4 通道时频图、LightGBM + stacker、
特征 BiGRU、时频 BiGRU、概率融合、记录内 Viterbi、严格 JSON 校验。

依赖：numpy、scipy、lightgbm、torch。模型权重不在源码中，默认自动寻找
脚本旁的 model/ 或 solution/model/，也可用 --model-dir 显式指定。

示例：
  python sleep_staging_all_in_one.py --input /path/to/edf --output /path/to/json
  python sleep_staging_all_in_one.py --input /path/to/edf --output /path/to/json --resume

标签映射：W/N1/N2/N3/REM = 0/1/2/3/4，每 30 秒输出一条。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
from torch import nn

CHANNELS = {"EEG", "EOG-L", "EOG-R", "M1", "ECG"}
BANDS = [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 25), (25, 35)]


# ----------------------------- 通用文件与 EDF -----------------------------
def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text("utf-8-sig"), object_pairs_hook=pairs, parse_constant=reject)


class Edf:
    """严格读取本比赛五通道、250 Hz、完整 30 秒片段的 EDF。"""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header = stream.read(256)
            self.n = int(header[252:256])
            raw = stream.read(self.n * 256)
        self.header_bytes = int(header[184:192])
        self.records = int(header[236:244])
        self.seconds = float(header[244:252])
        self.start = datetime.strptime(
            (header[168:176] + b" " + header[176:184]).decode(), "%d.%m.%y %H.%M.%S"
        )
        self.fields, offset = [], 0
        for width in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]:
            self.fields.append([
                raw[offset + i * width:offset + (i + 1) * width].decode("latin1").strip()
                for i in range(self.n)
            ])
            offset += width * self.n
        self.names = self.fields[0]
        self.samples = list(map(int, self.fields[8]))
        self.offsets = np.cumsum([0] + self.samples)
        duration = self.records * self.seconds
        if (
            self.n != 5 or set(self.names) != CHANNELS or self.seconds <= 0 or self.records <= 0
            or duration % 30 or any(s / self.seconds != 250 for s in self.samples)
            or self.header_bytes != 256 * (self.n + 1)
            or self.path.stat().st_size != self.header_bytes + self.records * sum(self.samples) * 2
        ):
            raise ValueError(f"{self.path.name}: expected five-channel 250 Hz EDF with complete 30-second epochs")
        if any(unit not in ("uV", "µV", "μV") for unit in self.fields[2]):
            raise ValueError(f"{self.path.name}: signal physical units must be microvolts (uV)")
        self.epochs = int(duration / 30)

    def digital(self):
        return np.memmap(
            self.path, dtype="<i2", mode="r", offset=self.header_bytes,
            shape=(self.records, sum(self.samples))
        )

    def signal(self, name):
        i = self.names.index(name)
        values = self.digital()[:, self.offsets[i]:self.offsets[i + 1]].astype(np.float32).reshape(-1)
        p0, p1 = float(self.fields[3][i]), float(self.fields[4][i])
        d0, d1 = float(self.fields[5][i]), float(self.fields[6][i])
        if not all(np.isfinite([p0, p1, d0, d1])) or d1 <= d0 or p1 <= p0:
            raise ValueError(f"{name}: invalid EDF physical calibration")
        return (values - d0) * ((p1 - p0) / (d1 - d0)) + p0


# ----------------------------- 手工特征与时频 -----------------------------
def features_for_signal(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    x = x.reshape(n // 3000, 3000)
    std = x.std(1)
    dx = np.diff(x, axis=1)
    d2 = np.diff(dx, axis=1)

    def log(v):
        return np.log10(np.maximum(v, 1e-8))

    f, p = signal.welch(x, fs=100, nperseg=400, noverlap=200, axis=1)
    use = (f >= 0.5) & (f <= 35)
    p, f = p[:, use], f[use]
    total = np.sum(p, axis=1) * 0.25 + 1e-8
    powers = np.stack([p[:, (f >= a) & (f < b)].sum(1) * 0.25 for a, b in BANDS], axis=1)
    rel = powers / total[:, None]
    prob = p / (p.sum(1, keepdims=True) + 1e-8)
    cum = prob.cumsum(1)
    q = np.percentile(x, [5, 25, 50, 75, 95], axis=1)
    var = std ** 2 + 1e-8
    dv = dx.var(1) + 1e-8
    out = [
        log(std + 1e-8), log(np.ptp(x, axis=1) + 1e-8), log(q[4] - q[0] + 1e-8),
        log(q[3] - q[1] + 1e-8), np.nan_to_num(stats.skew(x, axis=1)),
        np.nan_to_num(stats.kurtosis(x, axis=1)), np.sqrt(dv / var),
        np.sqrt(d2.var(1) / dv) / (np.sqrt(dv / var) + 1e-8),
        np.mean((x[:, 1:] > 0) != (x[:, :-1] > 0), axis=1), np.mean(abs(dx) < 0.01, axis=1),
        np.mean(abs(x) > 150, axis=1), np.mean(abs(x) > 500, axis=1), log(total),
        -(prob * np.log(prob + 1e-12)).sum(1) / np.log(len(f)),
        f[np.argmax(cum >= 0.5, axis=1)], f[np.argmax(cum >= 0.95, axis=1)],
        (prob * f).sum(1), log(np.mean(abs(dx), axis=1) + 1e-8),
    ]
    out.extend(log(powers[:, j] + 1e-8) for j in range(8))
    out.extend(rel[:, j] for j in range(8))
    blocks = x.reshape(-1, 6, 500)
    bst = blocks.std(2)
    out += [
        log(bst.min(1) + 1e-8), log(bst.max(1) + 1e-8), bst.std(1) / (bst.mean(1) + 1e-8),
        log((powers[:, :3].sum(1) + 1e-8) / (powers[:, 3:6].sum(1) + 1e-8)),
        log((powers[:, 3] + 1e-8) / (powers[:, 4] + 1e-8)),
        log((powers[:, 5] + 1e-8) / (powers[:, :3].sum(1) + 1e-8)),
    ]
    return np.nan_to_num(np.stack(out, axis=1), nan=0, posinf=20, neginf=-20).astype(np.float32)


def resampled_signals(edf: Edf):
    return {
        name: signal.resample_poly(edf.signal(name), 2, 5).astype(np.float32)
        for name in ["EEG", "EOG-L", "EOG-R", "M1", "ECG"]
    }


def extract_features(path):
    edf = Edf(path)
    s = resampled_signals(edf)
    derived = {
        "EEG_M1": s["EEG"] - s["M1"], "EOG_L_M1": s["EOG-L"] - s["M1"],
        "EOG_R_M1": s["EOG-R"] - s["M1"], "EOG_diff": s["EOG-L"] - s["EOG-R"],
        "raw_EEG": s["EEG"], "raw_M1": s["M1"],
        "eye_common": (s["EOG-L"] + s["EOG-R"]) / 2, "ECG": s["ECG"],
    }
    features = [features_for_signal(x) for x in derived.values()]
    n = edf.epochs
    for a, b in [("EEG", "M1"), ("EOG-L", "EOG-R"), ("EEG", "EOG-L"),
                 ("EEG", "EOG-R"), ("EOG-L", "M1"), ("EOG-R", "M1")]:
        x = s[a].reshape(n, 3000)
        y = s[b].reshape(n, 3000)
        x = x - x.mean(1, keepdims=True)
        y = y - y.mean(1, keepdims=True)
        corr = np.mean(x * y, axis=1) / (x.std(1) * y.std(1) + 1e-8)
        features.append(corr[:, None])
    t = np.arange(n) * 30
    clock = edf.start.hour * 3600 + edf.start.minute * 60 + edf.start.second + t
    features.append(np.stack([t / 3600, (n * 30 - t) / 3600,
                              np.sin(clock * 2 * np.pi / 86400),
                              np.cos(clock * 2 * np.pi / 86400)], axis=1))
    out = np.concatenate(features, axis=1).astype(np.float32)
    if out.shape != (n, 330) or not np.isfinite(out).all():
        raise ValueError("Expected 330 finite auxiliary features per epoch")
    return out


def extract_spectra(path):
    edf = Edf(path)
    s = resampled_signals(edf)
    out = []
    for x in [s["EEG"] - s["M1"], s["EOG-L"] - s["M1"],
              s["EOG-R"] - s["M1"], s["EOG-L"] - s["EOG-R"]]:
        f, _, p = signal.spectrogram(
            x.reshape(-1, 3000), fs=100, window="hann", nperseg=200, noverlap=100,
            nfft=200, detrend="constant", scaling="density", mode="psd", axis=-1
        )
        out.append(np.log10(np.maximum(p[:, (f >= 0.5) & (f <= 32), :], 1e-8)))
    spec = np.stack(out, axis=1).astype(np.float16)
    assert spec.shape[1:] == (4, 64, 29) and np.isfinite(spec).all()
    return spec


# ----------------------------- 记录内变换与解码 -----------------------------
def robust_normalize(x):
    median = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25, 75], axis=0)
    scale = q75 - q25
    std = np.std(x, axis=0)
    scale = np.where(scale > 1e-5, scale, np.where(std > 1e-5, std, 1.0))
    return np.clip((x - median) / scale, -8, 8).astype(np.float32)


def shifted_mean(x, radius):
    if radius == 0:
        return x
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return uniform_filter1d(padded, size=2 * radius + 1, axis=0, mode="nearest")[radius:-radius].astype(np.float32)


def make_features(x, groups, mode):
    if mode not in {"raw", "norm", "context"}:
        raise ValueError(mode)
    if mode == "raw":
        return np.ascontiguousarray(x, dtype=np.float32)
    blocks, start = [], 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) == 0 or not np.array_equal(indices, np.arange(start, start + len(indices))):
            raise ValueError("Record groups must be contiguous")
        current = x[indices]
        z = robust_normalize(current)
        parts = [current, z]
        if mode == "context":
            parts += [shifted_mean(z, 1), shifted_mean(z, 3)]
        blocks.append(np.concatenate(parts, axis=1))
        start += len(indices)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)


def viterbi_record(probabilities, transition, strength):
    n = len(probabilities)
    if strength == 0 or n < 2:
        return probabilities.argmax(axis=1)
    emission = np.log(np.clip(probabilities, 1e-7, 1)).astype(np.float64)
    link = strength * np.log(np.clip(transition, 1e-7, 1))
    score = emission[0].copy()
    back = np.zeros((n, 5), dtype=np.int8)
    for i in range(1, n):
        candidates = score[:, None] + link
        back[i] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emission[i]
    result = np.zeros(n, dtype=np.int8)
    result[-1] = score.argmax()
    for i in range(n - 2, -1, -1):
        result[i] = back[i + 1, result[i + 1]]
    return result


# ----------------------------- OOF 时序 stacker -----------------------------
def shift(x, offset):
    indices = np.clip(np.arange(len(x)) + offset, 0, len(x) - 1)
    return x[indices]


def stack_features(probabilities, groups):
    output = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        ps = [p[idx] for p in probabilities]
        blend = 0.4 * ps[0] + 0.6 * ps[2]
        parts = [np.log(np.clip(blend, 1e-6, 1)), *ps]
        for offset in [-12, -6, -3, -1, 1, 3, 6, 12]:
            parts.append(shift(blend, offset))
        for radius in [1, 3, 6]:
            parts.append(shifted_mean(blend, radius))
        parts += [blend - shifted_mean(blend, 3), blend - shifted_mean(blend, 6)]
        output.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.concatenate(output)


# ----------------------------- 特征神经网络 -----------------------------
class NeuralSleepSequence(nn.Module):
    def __init__(self, inputs=660):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(inputs, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15))
        self.conv = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.sequence_norm = nn.LayerNorm(128)
        self.gru = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(192, 5))

    def forward(self, x):
        z = self.input(x)
        z = z + 0.5 * self.conv(z.transpose(1, 2)).transpose(1, 2)
        z = self.sequence_norm(z)
        return self.head(self.gru(z)[0])


def windows_for_record(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.inference_mode()
def predict_neural(model, x, groups, indices, device="cpu", window=32, batch_size=32):
    model.eval()
    selected = np.zeros(len(groups), dtype=bool)
    selected[indices] = True
    output = np.zeros((len(indices), 5), dtype=np.float32)
    position_map = np.full(len(groups), -1, dtype=np.int64)
    position_map[indices] = np.arange(len(indices))
    for group in np.unique(groups[indices]):
        idx = np.flatnonzero((groups == group) & selected)
        starts = windows_for_record(len(idx), window, window // 2)
        total = np.zeros((len(idx), 5), dtype=np.float64)
        counts = np.zeros((len(idx), 1), dtype=np.float64)
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            sequences = [idx[start:min(start + window, len(idx))] for start in batch_starts]
            tensor = torch.from_numpy(np.stack([x[sequence] for sequence in sequences])).to(device)
            probabilities = model(tensor).softmax(dim=-1).cpu().numpy()
            for sequence, p in zip(sequences, probabilities):
                local = position_map[sequence] - position_map[idx[0]]
                total[local] += p[:len(sequence)]
                counts[local] += 1
        output[position_map[idx]] = total / counts
    return output


# ----------------------------- 时频神经网络 -----------------------------
class SpectrogramSleepSequence(nn.Module):
    def __init__(self, aux_inputs=660):
        super().__init__()
        layers, channels = [], 4
        for output in [12, 24, 32]:
            layers += [nn.Conv2d(channels, output, 3, padding=1), nn.BatchNorm2d(output),
                       nn.GELU(), nn.MaxPool2d(2)]
            channels = output
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 1)), nn.Flatten(), nn.Linear(128, 64), nn.GELU())
        self.aux = nn.Sequential(nn.Linear(aux_inputs, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.15))
        self.fusion = nn.Sequential(nn.Linear(160, 160), nn.LayerNorm(160), nn.GELU(), nn.Dropout(0.2))
        self.gru = nn.GRU(160, 96, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(192, 5))

    def forward(self, spectra, aux):
        batch, time = spectra.shape[:2]
        s = self.cnn(spectra.flatten(0, 1))
        a = self.aux(aux.flatten(0, 1))
        z = self.fusion(torch.cat([s, a], dim=1)).reshape(batch, time, 160)
        return self.head(self.gru(z)[0])


@torch.inference_mode()
def predict_spectrogram(model, spectra, aux, spec_mean, spec_std, groups, indices, window=32, batch_size=8):
    model.eval()
    selected = np.zeros(len(groups), dtype=bool)
    selected[indices] = True
    output = np.zeros((len(indices), 5), dtype=np.float32)
    position_map = np.full(len(groups), -1, dtype=np.int64)
    position_map[indices] = np.arange(len(indices))
    for group in np.unique(groups[indices]):
        idx = np.flatnonzero((groups == group) & selected)
        starts = windows_for_record(len(idx), window, window // 2)
        total = np.zeros((len(idx), 5), dtype=np.float64)
        counts = np.zeros((len(idx), 1), dtype=np.float64)
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            sequences = [idx[start:min(start + window, len(idx))] for start in batch_starts]
            raw = np.stack([np.asarray(spectra[sequence], dtype=np.float32) for sequence in sequences])
            s = torch.from_numpy(np.clip((raw - spec_mean) / spec_std, -8, 8))
            a = torch.from_numpy(np.stack([aux[sequence] for sequence in sequences]))
            probabilities = model(s, a).softmax(dim=-1).cpu().numpy()
            for sequence, p in zip(sequences, probabilities):
                local = position_map[sequence] - position_map[idx[0]]
                total[local] += p[:len(sequence)]
                counts[local] += 1
        output[position_map[idx]] = total / counts
    return output


# ----------------------------- 正式融合模型 -----------------------------
def auxiliary_features(x, mean, std):
    raw = np.clip((x - mean) / std, -8, 8)
    return np.concatenate([raw, robust_normalize(x)], axis=1).astype(np.float32)


def checked_path(directory, name, expected_hash):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha(path) != expected_hash:
        raise ValueError(f"Model file missing, invalid, or hash mismatch: {name}")
    return path


class CompetitionEnsemble:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = load_json(self.directory / "config.json")
        self.boosters = {}
        for mode in ["raw", "norm", "context"]:
            item = self.config["base_models"][mode]
            self.boosters[mode] = lgb.Booster(model_file=str(checked_path(self.directory, item["file"], item["sha256"])))
        item = self.config["stacker"]
        self.stacker = lgb.Booster(model_file=str(checked_path(self.directory, item["file"], item["sha256"])))
        self.neural = []
        for item in self.config["neural_models"]:
            checkpoint = torch.load(checked_path(self.directory, item["file"], item["sha256"]), map_location="cpu", weights_only=True)
            model = NeuralSleepSequence()
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.neural.append((model, checkpoint))
        self.spectrogram = []
        for item in self.config["spectrogram_models"]:
            checkpoint = torch.load(checked_path(self.directory, item["file"], item["sha256"]), map_location="cpu", weights_only=True)
            model = SpectrogramSleepSequence()
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.spectrogram.append((model, checkpoint))
        self.transition = np.asarray(self.config["transition_matrix"], dtype=np.float64)
        self.calibration = np.asarray(self.config["calibration"], dtype=np.float64)
        if self.transition.shape != (5, 5) or self.calibration.shape != (5,):
            raise ValueError("Invalid transition/calibration configuration")

    def predict(self, edf_path):
        x = extract_features(edf_path)
        spectra = extract_spectra(edf_path)
        n = len(x)
        groups = np.zeros(n, dtype=np.int16)
        indices = np.arange(n)
        base_predictions = []
        for mode in ["raw", "norm", "context"]:
            transformed = make_features(x, groups, mode)
            base_predictions.append(self.boosters[mode].predict(transformed, num_threads=8))
        base = 0.4 * base_predictions[0] + 0.6 * base_predictions[2]
        stacked = self.stacker.predict(stack_features(base_predictions, groups), num_threads=8)
        tree = 0.5 * base + 0.5 * stacked
        neural_probability = np.zeros((n, 5), dtype=np.float64)
        for model, checkpoint in self.neural:
            aux = auxiliary_features(x, checkpoint["mean"].numpy(), checkpoint["std"].numpy())
            neural_probability += predict_neural(model, aux, groups, indices, window=32, batch_size=32) / len(self.neural)
        spectrogram_probability = np.zeros((n, 5), dtype=np.float64)
        for model, checkpoint in self.spectrogram:
            aux = auxiliary_features(x, checkpoint["aux_mean"].numpy(), checkpoint["aux_std"].numpy())
            spectrogram_probability += predict_spectrogram(
                model, spectra, aux, checkpoint["spec_mean"].numpy(), checkpoint["spec_std"].numpy(),
                groups, indices, window=32, batch_size=8
            ) / len(self.spectrogram)
        weights = self.config["final_weights"]
        probability = (
            weights["tree_stacker"] * tree + weights["feature_neural"] * neural_probability
            + weights["spectrogram_neural"] * spectrogram_probability
        )
        probability *= self.calibration[None, :]
        probability /= probability.sum(axis=1, keepdims=True)
        if probability.shape != (n, 5) or not np.isfinite(probability).all():
            raise ValueError("Invalid ensemble probabilities")
        return viterbi_record(probability, self.transition, float(self.config["transition_strength"]))


def validate_output(edf_path, output_path):
    """逐条校验输出结构、时间、条数与标签范围。"""
    edf = Edf(edf_path)
    value = load_json(output_path)
    if set(value) != {"Sleep"} or not isinstance(value["Sleep"], list) or len(value["Sleep"]) != edf.epochs:
        raise ValueError(f"{edf_path.stem}: invalid Sleep array")
    for i, item in enumerate(value["Sleep"]):
        expected_time = (edf.start + timedelta(seconds=30 * i)).strftime("%Y-%m-%d %H:%M:%S")
        if not isinstance(item, dict) or set(item) != {"time", "period"} or item["time"] != expected_time:
            raise ValueError(f"{edf_path.stem}: invalid item {i}")
        if type(item["period"]) is not int or not 0 <= item["period"] <= 4:
            raise ValueError(f"{edf_path.stem}: invalid period at {i}")
    return True


def default_model_dir():
    root = Path(__file__).resolve().parent
    candidates = [root / "model", root / "solution" / "model", Path.cwd() / "model", Path.cwd() / "solution" / "model"]
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/input"), help="直接存放 EDF 的目录")
    parser.add_argument("--output", type=Path, default=Path("/output"), help="JSON 输出目录")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir(), help="模型权重目录")
    parser.add_argument("--resume", action="store_true", help="跳过条数正确且通过校验的已有输出")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    files = sorted(path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() == ".edf")
    if not files or len({path.stem for path in files}) != len(files):
        raise ValueError("Input must contain EDF files with distinct stems")
    ensemble = CompetitionEnsemble(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    for path in files:
        started = time.perf_counter()
        edf = Edf(path)
        destination = args.output / f"{path.stem}.json"
        if args.resume and destination.exists():
            try:
                validate_output(path, destination)
                print(f"{path.name}: already complete and valid, skipped", flush=True)
                continue
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        labels = ensemble.predict(path)
        if len(labels) != edf.epochs or np.any(labels < 0) or np.any(labels > 4):
            raise ValueError(f"{path.name}: invalid output labels")
        save_json(destination, {"Sleep": [
            {"time": (edf.start + timedelta(seconds=30 * i)).strftime("%Y-%m-%d %H:%M:%S"), "period": int(label)}
            for i, label in enumerate(labels)
        ]})
        validate_output(path, destination)
        counts = np.bincount(labels, minlength=5).tolist()
        print(f"{path.name}: {len(labels)} epochs, classes={counts}, elapsed={time.perf_counter() - started:.2f}s", flush=True)
    print(f"Completed {len(files)} EDF files in {time.perf_counter() - total_started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
