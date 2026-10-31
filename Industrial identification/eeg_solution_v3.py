#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2026 脑机接口大赛：静息态 / 认知负荷态单通道 EEG 解码。

核心设计（第三版冻结配置）：
1. 所有特征只读取当前及过去共 1,500 点原始 EEG，感受野严格为 6 秒；
2. 106 维多分辨率时频特征 + AR(60) + 6 维 DFA + 33 维 5 秒特征，共 205 维；
3. 只利用当前记录过去无标签特征的 EWMA 个体基线，消除被试间差异；
4. 40% 原互补集成 + 60% 增强平衡 Ridge 概率融合，再做单向 EWMA；
5. 完全重复记录只保留一份，交叉验证按记录分组，杜绝窗口泄漏；
6. 支持任意大小的按时间顺序分块输入，不使用当前时刻之后的数据。

命令示例：
    python eeg_solution.py train --data-dir "完整试验数据_最终版" --output-dir artifacts --nested
    python eeg_solution.py predict --model artifacts/eeg_decoder.joblib --bdf "001.bdf" --output predictions.csv
    python eeg_solution.py stream-demo --model artifacts/eeg_decoder.joblib --bdf "001.bdf" --limit 5

流式 API 输入单位：微伏（µV）；采样率必须为 250 Hz。
score 越大越支持 task（认知负荷态，标签 1）；越小越支持 relax（静息态，标签 0）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- 固定算法配置 -----------------------------
SFREQ = 250
FEATURE_WINDOW_SECONDS = 4
FIR_TAPS = 501                       # 4 秒特征的 501 点单向 FIR
FIR5_TAPS = 251                      # 5 秒特征的 251 点单向 FIR
STRIDE_SECONDS = 2
ADAPT_ALPHA = 0.02                   # 基线 EWMA：约 100 秒量级
SMOOTH_ALPHA = 0.03                  # 输出 EWMA：约 67 秒量级
RIDGE_ALPHA = 10.0
AUGMENTED_RIDGE_ALPHA = 1.0
TREE_WEIGHT = 0.2                    # 原集成内部：20% Extra Trees + 80% Ridge
AUGMENTED_WEIGHT = 0.6               # 最终：60% 增强 Ridge + 40% 原集成
AR_ORDER = 60
N_OUTER_FOLDS = 6
RANDOM_STATE = 42

FEATURE_SAMPLES = FEATURE_WINDOW_SECONDS * SFREQ
FEATURE5_SAMPLES = 5 * SFREQ
FIR_HISTORY_SAMPLES = FIR_TAPS - 1
STRIDE_SAMPLES = STRIDE_SECONDS * SFREQ
RECEPTIVE_SAMPLES = 6 * SFREQ         # 所有分支共用严格 6 秒原始信号上限
N_FULL_FEATURES = 33
N_DYNAMIC_FEATURES = 40
N_LAST2_FEATURES = 33
N_CURRENT_FEATURES = N_FULL_FEATURES + N_DYNAMIC_FEATURES + N_LAST2_FEATURES
N_DFA_FEATURES = 6
N_FEATURES = N_CURRENT_FEATURES + AR_ORDER + N_DFA_FEATURES + N_FULL_FEATURES
# Extra Trees 仍只使用原 106 维中的 full4s + last2s，共 66 维。
TREE_COLUMNS = np.r_[np.arange(33), np.arange(73, 106)]

CONFIG = {
    "sfreq": SFREQ,
    "feature_window_seconds": FEATURE_WINDOW_SECONDS,
    "fir_taps": FIR_TAPS,
    "fir5_taps": FIR5_TAPS,
    "stride_seconds": STRIDE_SECONDS,
    "max_receptive_field_seconds": RECEPTIVE_SAMPLES / SFREQ,
    "adapt_alpha": ADAPT_ALPHA,
    "smooth_alpha": SMOOTH_ALPHA,
    "ridge_alpha": RIDGE_ALPHA,
    "augmented_ridge_alpha": AUGMENTED_RIDGE_ALPHA,
    "tree_weight": TREE_WEIGHT,
    "augmented_weight": AUGMENTED_WEIGHT,
    "base_ensemble_weight": 1-AUGMENTED_WEIGHT,
    "ar_order": AR_ORDER,
    "n_current_features": N_CURRENT_FEATURES,
    "n_features": N_FEATURES,
    "input_unit": "uV",
    "positive_class": "task",
    "score_direction": "larger_score_means_task",
}

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


# ----------------------------- 特征提取 -----------------------------
def extract_window_features(x: np.ndarray, sfreq: int = SFREQ) -> tuple[np.ndarray, list[str]]:
    """从一个已到达的短时窗提取 33 维特征；不访问窗口外的未来数据。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("本方案针对单通道 EEG，输入必须是一维数组。")

    # 去趋势只使用当前窗口，抑制基线漂移；不涉及任何未来窗口。
    x = signal.detrend(x, type="linear")
    scale = np.std(x) + 1e-12
    z = x / scale

    # 周期图/Welch 频谱完全在当前窗口内计算，频带积分自然限制到 1–45 Hz。
    f, p = signal.welch(
        x, fs=sfreq, window="hann", nperseg=len(x), noverlap=0, detrend=False
    )
    total_mask = (f >= 1) & (f <= 45)
    total = np.trapezoid(p[total_mask], f[total_mask]) + 1e-20

    feats: list[float] = []
    names: list[str] = []
    powers: dict[str, float] = {}

    # 每个频带同时保留绝对对数能量与相对对数能量。
    for name, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f < hi)
        power = np.trapezoid(p[mask], f[mask]) + 1e-20
        powers[name] = power
        feats.extend([np.log(power), np.log(power / total)])
        names.extend([f"log_{name}", f"rel_{name}"])

    # 常见负荷相关频谱比值；对数使分布更接近线性模型假设。
    for name, a, b in [
        ("theta_alpha", "theta", "alpha"),
        ("beta_alpha", "beta", "alpha"),
        ("beta_theta", "beta", "theta"),
        ("delta_alpha", "delta", "alpha"),
    ]:
        feats.append(np.log(powers[a] / powers[b]))
        names.append(name)
    feats.append(np.log((powers["theta"] + powers["beta"]) / powers["alpha"]))
    names.append("theta_beta_alpha")

    # 频谱质心和归一化频谱熵描述能量分布形态。
    pn = p[total_mask]
    pn = pn / (pn.sum() + 1e-20)
    fn = f[total_mask]
    feats.extend([
        np.log(scale),
        np.sum(fn * pn),
        -np.sum(pn * np.log(pn + 1e-20)) / np.log(len(pn)),
        np.mean(np.abs(z)),
        np.sqrt(np.mean(z ** 2)),
        np.mean(np.abs(np.diff(z))),
        np.sum(np.diff(np.signbit(z)) != 0) / len(z),
        stats.skew(z),
        stats.kurtosis(z),
        np.percentile(z, 95) - np.percentile(z, 5),
    ])
    names.extend([
        "log_std", "spectral_centroid", "spectral_entropy", "mean_abs", "rms",
        "line_length", "zero_cross", "skew", "kurtosis", "robust_range",
    ])

    # Hjorth 活动性相关统计、移动性和复杂性。
    dx = np.diff(z)
    ddx = np.diff(dx)
    var0, var1, var2 = np.var(z), np.var(dx), np.var(ddx)
    mobility = np.sqrt(var1 / (var0 + 1e-20))
    complexity = np.sqrt(var2 / (var1 + 1e-20)) / (mobility + 1e-20)
    feats.extend([mobility, complexity])
    names.extend(["hjorth_mobility", "hjorth_complexity"])

    # 多尺度自相关补充短时节律信息。
    for lag in [1, 5, 10, 25, 50, 125]:
        if np.std(z[:-lag]) > 0 and np.std(z[lag:]) > 0:
            corr = np.corrcoef(z[:-lag], z[lag:])[0, 1]
        else:
            corr = 0.0
        feats.append(corr)
        names.append(f"autocorr_{lag}")

    result = np.nan_to_num(np.asarray(feats), nan=0.0, posinf=1e6, neginf=-1e6)
    return result, names


def extract_dynamic_features(segment: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """四个 1 秒子窗的频带动态：标准差、斜率、末减初和极差，共 40 维。"""
    pieces = signal.detrend(segment.reshape(4, SFREQ), axis=-1, type="linear")
    f, p = signal.welch(pieces, fs=SFREQ, window="hann", nperseg=SFREQ,
                        noverlap=0, detrend=False, axis=-1)
    total_mask = (f >= 1) & (f <= 45)
    total = np.trapezoid(p[:, total_mask], f[total_mask], axis=1) + 1e-20
    log_bands, rel_bands = [], []
    for lo, hi in BANDS.values():
        band_mask = (f >= lo) & (f < hi)
        power = np.trapezoid(p[:, band_mask], f[band_mask], axis=1) + 1e-20
        log_bands.append(np.log(power))
        rel_bands.append(np.log(power / total))

    values, names = [], []
    t = np.arange(4, dtype=float)
    t = t-t.mean()
    for kind, array in [("log", np.asarray(log_bands)), ("rel", np.asarray(rel_bands))]:
        statistics = {
            "std": np.std(array, axis=1),
            "slope": (array @ t)/np.sum(t*t),
            "last_first": array[:, -1]-array[:, 0],
            "range": np.ptp(array, axis=1),
        }
        for statistic, bands_values in statistics.items():
            values.extend(bands_values)
            names.extend([f"dynamic_{kind}_{statistic}_{band}" for band in BANDS])
    values = np.nan_to_num(np.asarray(values), nan=0, posinf=1e6, neginf=-1e6)
    return values, names


def extract_combined_features(segment: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """组合完整 4 秒、1 秒子窗动态和最近 2 秒三种时间分辨率。"""
    full, full_names = extract_window_features(segment)
    dynamic, dynamic_names = extract_dynamic_features(segment)
    last2, last2_names = extract_window_features(segment[-2*SFREQ:])
    features = np.concatenate([full, dynamic, last2])
    names = full_names + dynamic_names + [f"last2_{name}" for name in last2_names]
    return features, names


def extract_ar_dfa_features(raw6: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """从当前 6 秒原始窗提取 AR(60) 与 DFA 六维特征。

    降采样只是隔点抽取，不引入任何额外历史；标准化也只使用当前窗。
    """
    x = np.asarray(raw6, dtype=np.float64)
    if x.shape != (RECEPTIVE_SAMPLES,):
        raise ValueError(f"AR/DFA 分支需要 {RECEPTIVE_SAMPLES} 点，即完整 6 秒窗。")
    x = x-x.mean()
    x = x/(x.std()+1e-8)

    # Yule-Walker AR(60)：62.5 Hz 序列、FFT 自相关和带微小岭正则的求解。
    xa = x[::4]
    spectrum = np.fft.rfft(xa, n=1024)
    ac = np.fft.irfft(spectrum*np.conj(spectrum), n=1024)[:AR_ORDER+1]/len(xa)
    indices = np.arange(AR_ORDER)
    matrix = ac[np.abs(indices[:, None]-indices[None, :])]
    ar = np.linalg.solve(matrix+1e-5*np.eye(AR_ORDER), ac[1:AR_ORDER+1])
    ar = np.nan_to_num(ar, nan=0.0, posinf=1e6, neginf=-1e6)

    # DFA：41.7 Hz 序列，五个尺度的去趋势波动及其 log-log 斜率。
    xd = x[::6]
    profile = np.cumsum(xd-xd.mean())
    scales = [5, 10, 20, 40, 60]
    fluctuations = []
    for scale in scales:
        count = len(xd)//scale
        pieces = profile[:count*scale].reshape(count, scale)
        t = np.arange(scale, dtype=float)
        centered = t-t.mean()
        slopes = (pieces@centered)/(centered@centered)
        intercept = pieces.mean(axis=1)-slopes*t.mean()
        residual = pieces-slopes[:, None]*t-intercept[:, None]
        fluctuations.append(np.sqrt(np.mean(residual**2))+1e-12)
    log_fluctuations = np.log(fluctuations)
    dfa_alpha = np.polyfit(np.log(scales), log_fluctuations, 1)[0]
    dfa = np.r_[log_fluctuations, dfa_alpha]
    names = [f"ar_{i+1}" for i in range(AR_ORDER)]
    names += [f"dfa_log_rms_{s}" for s in scales] + ["dfa_alpha"]
    return np.r_[ar, dfa], names


def extract_third_features(segment4: np.ndarray, raw6: np.ndarray,
                           segment5: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """组合 106 维原特征、66 维 AR/DFA 和 33 维 5 秒特征。"""
    current, current_names = extract_combined_features(segment4)
    ar_dfa, ar_dfa_names = extract_ar_dfa_features(raw6)
    feature5, names5 = extract_window_features(segment5)
    features = np.concatenate([current, ar_dfa, feature5])
    names = current_names + ar_dfa_names + [f"window5_{name}" for name in names5]
    return features, names


def fit_ensemble(X: np.ndarray, y: np.ndarray) -> tuple[object, object]:
    """训练原 106 维互补集成；预处理只在当前训练折拟合。"""
    tree = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=30, max_features=.8,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )
    ridge = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=RIDGE_ALPHA, class_weight="balanced"),
    )
    tree.fit(X[:, TREE_COLUMNS], y)
    ridge.fit(X, y)
    return tree, ridge


def fit_augmented(X: np.ndarray, y: np.ndarray) -> object:
    """训练 205 维增强 Ridge，标准化不接触验证记录。"""
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=AUGMENTED_RIDGE_ALPHA, class_weight="balanced"),
    )
    model.fit(X, y)
    return model


def ensemble_decision(tree: object, ridge: object, augmented: object,
                      X_current: np.ndarray, X_augmented: np.ndarray) -> np.ndarray:
    """40% 原集成 + 60% 增强 Ridge，在概率空间融合后转 logit。"""
    tree_probability = tree.predict_proba(X_current[:, TREE_COLUMNS])[:, 1]
    ridge_probability = expit(ridge.decision_function(X_current))
    base_probability = TREE_WEIGHT*tree_probability + (1-TREE_WEIGHT)*ridge_probability
    augmented_probability = expit(augmented.decision_function(X_augmented))
    probability = ((1-AUGMENTED_WEIGHT)*base_probability
                   + AUGMENTED_WEIGHT*augmented_probability)
    return logit(np.clip(probability, 1e-6, 1-1e-6))


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """只用训练/OOF 预测搜索 bACC 最大阈值；并列时取中间候选。"""
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 193)))
    values = np.array([balanced_accuracy_score(y, score >= t) for t in candidates])
    best = np.flatnonzero(values >= values.max() - 1e-12)
    return float(candidates[best[len(best) // 2]])


def causal_smooth(score: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """每条记录独立做 adjust=False 的单向 EWMA，与流式递推完全一致。"""
    return pd.Series(score).groupby(subjects).transform(
        lambda s: s.ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()
    ).to_numpy()


# ----------------------------- 数据读取与训练集构建 -----------------------------
def discover_records(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """兼容数据根目录外多嵌套一层的情况，返回编号、BDF、事件 CSV。"""
    records = []
    for events_path in sorted(data_dir.rglob("*_events.csv")):
        sid = events_path.name.replace("_events.csv", "")
        bdf_path = events_path.with_name(f"{sid}.bdf")
        if bdf_path.exists():
            records.append((sid, bdf_path, events_path))
    if not records:
        raise FileNotFoundError(f"在 {data_dir} 下未找到成对的 BDF 和 *_events.csv。")
    return records


def read_single_channel_bdf(path: Path) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """MNE 默认返回伏特；统一乘 1e6 转为微伏，避免数值尺度过小。"""
    raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    if len(raw.ch_names) != 1:
        raise ValueError(f"{path} 有 {len(raw.ch_names)} 个通道；本模型要求单通道 EEG。")
    if not np.isclose(raw.info["sfreq"], SFREQ):
        raise ValueError(f"{path} 采样率为 {raw.info['sfreq']}，模型要求 {SFREQ} Hz。")
    x = raw.get_data()[0].astype(np.float64) * 1e6
    return raw, x


def bad_annotation_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """将 BAD_ACQ_SKIP 标注转换为采样点布尔掩码。"""
    bad = np.zeros(raw.n_times, dtype=bool)
    for ann in raw.annotations:
        if ann["description"] == "BAD_ACQ_SKIP":
            begin = max(0, int(np.floor(ann["onset"] * SFREQ)))
            end = min(raw.n_times, int(np.ceil((ann["onset"] + ann["duration"]) * SFREQ)))
            bad[begin:end] = True
    return bad


def read_trials(raw: mne.io.BaseRaw, events_path: Path) -> list[tuple[int, int, int]]:
    """统一 UTC 时区并配对 trial_start/end，返回起点、终点、标签。"""
    events = pd.read_csv(events_path)
    events["utc"] = pd.to_datetime(events["local_time"], utc=True)
    meas_date = pd.Timestamp(raw.info["meas_date"])
    if meas_date.tzinfo is None:
        meas_date = meas_date.tz_localize("UTC")

    trials = []
    current = None
    for event in events.itertuples():
        if event.event == "trial_start":
            current = event
        elif event.event == "trial_end" and current is not None:
            begin = int(round((current.utc - meas_date).total_seconds() * SFREQ))
            end = int(round((event.utc - meas_date).total_seconds() * SFREQ))
            label = 1 if current.phase == "task" else 0
            trials.append((begin, end, label))
            current = None
    return trials


def fir_coefficients() -> np.ndarray:
    """4 秒分支的线性相位单向 FIR。"""
    return signal.firwin(FIR_TAPS, [0.5, 45], pass_zero=False, fs=SFREQ)


def fir5_coefficients() -> np.ndarray:
    """5 秒分支的 251 点 FIR，1 秒历史 + 5 秒特征仍恰为 6 秒。"""
    return signal.firwin(FIR5_TAPS, [0.5, 45], pass_zero=False, fs=SFREQ)


def build_training_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """构建 205 维在线基线残差；完全重复信号在入口处直接去重。"""
    all_x, all_y, all_subjects, rows = [], [], [], []
    seen_hashes: dict[str, str] = {}
    feature_names: list[str] = []
    fir, fir5 = fir_coefficients(), fir5_coefficients()

    for sid, bdf_path, events_path in discover_records(data_dir):
        started = time.time()
        raw, raw_x = read_single_channel_bdf(bdf_path)
        digest = hashlib.sha256(raw_x.tobytes()).hexdigest()
        if digest in seen_hashes:
            print(f"{sid}: 与 {seen_hashes[digest]} 完全重复，跳过。", flush=True)
            continue
        seen_hashes[digest] = sid

        # 离线训练与部署均使用单向 lfilter，不调用双向滤波。
        filtered = signal.lfilter(fir, [1.0], raw_x)
        filtered5 = signal.lfilter(fir5, [1.0], raw_x)
        bad = bad_annotation_mask(raw)
        baseline = None

        for trial_begin, trial_end, label in read_trials(raw, events_path):
            for end in range(trial_begin + FEATURE_SAMPLES, trial_end + 1, STRIDE_SAMPLES):
                start = end - FEATURE_SAMPLES
                receptive_start = end - RECEPTIVE_SAMPLES
                if start < 0 or receptive_start < 0 or end > len(filtered):
                    continue
                # 三个分支共享的完整 6 秒原始感受野内不得有缺失标记。
                if bad[receptive_start:end].any():
                    continue
                segment = filtered[start:end]
                raw6 = raw_x[receptive_start:end]
                segment5 = filtered5[end-FEATURE5_SAMPLES:end]
                if np.std(segment) < 1e-8:
                    continue

                feature, feature_names = extract_third_features(segment, raw6, segment5)
                # 原始特征统一量化为 float32，再以 float64 更新基线。
                feature = feature.astype(np.float32)
                residual = np.zeros_like(feature) if baseline is None else feature - baseline
                all_x.append(residual)
                all_y.append(label)
                all_subjects.append(sid)
                rows.append({
                    "subject": sid, "start_sample": start, "end_sample": end,
                    "label": label, "phase": "task" if label == 1 else "relax",
                    "sha256": digest,
                })

                # 当前特征只能在当前预测之后并入基线，严格对应 shift(1)。
                if baseline is None:
                    baseline = feature.astype(np.float64).copy()
                else:
                    baseline = (1 - ADAPT_ALPHA) * baseline + ADAPT_ALPHA * feature

        print(f"{sid}: 累计 {len(all_y)} 个有效窗，耗时 {time.time() - started:.1f}s", flush=True)

    X = np.asarray(all_x, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int8)
    subjects = np.asarray(all_subjects, dtype=str)
    metadata = pd.DataFrame(rows)
    return X, y, subjects, metadata, feature_names


# ----------------------------- 交叉验证与模型训练 -----------------------------
def nested_validation(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[dict, np.ndarray]:
    """冻结配置后用新分组种子做 6×4 嵌套验证，并返回外层 OOF 分数。"""
    outer_splits = list(GroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=2026).split(X, y, subjects))
    outer_raw = np.full(len(y), np.nan)
    prediction = np.zeros(len(y), dtype=bool)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(outer_splits):
        inner_raw = np.full(len(train_idx), np.nan)
        inner_splitter = GroupKFold(n_splits=4, shuffle=True, random_state=3026+fold)
        for inner_train, inner_valid in inner_splitter.split(X[train_idx], y[train_idx], subjects[train_idx]):
            fit_idx = train_idx[inner_train]
            predict_idx = train_idx[inner_valid]
            tree, ridge = fit_ensemble(X[fit_idx, :N_CURRENT_FEATURES], y[fit_idx])
            augmented = fit_augmented(X[fit_idx], y[fit_idx])
            inner_raw[inner_valid] = ensemble_decision(
                tree, ridge, augmented, X[predict_idx, :N_CURRENT_FEATURES], X[predict_idx]
            )

        inner_smoothed = causal_smooth(inner_raw, subjects[train_idx])
        threshold = best_threshold(y[train_idx], inner_smoothed)
        tree, ridge = fit_ensemble(X[train_idx, :N_CURRENT_FEATURES], y[train_idx])
        augmented = fit_augmented(X[train_idx], y[train_idx])
        outer_raw[valid_idx] = ensemble_decision(
            tree, ridge, augmented, X[valid_idx, :N_CURRENT_FEATURES], X[valid_idx]
        )
        fold_rows.append({"fold": fold, "threshold": threshold})

    outer_smoothed = causal_smooth(outer_raw, subjects)
    for row, (_, valid_idx) in zip(fold_rows, outer_splits):
        prediction[valid_idx] = outer_smoothed[valid_idx] >= row["threshold"]
        row.update({
            "n_windows": int(len(valid_idx)),
            "n_records": int(len(np.unique(subjects[valid_idx]))),
            "bACC": balanced_accuracy_score(y[valid_idx], prediction[valid_idx]),
            "AUC": roc_auc_score(y[valid_idx], outer_smoothed[valid_idx]),
        })
        row["mean"] = (row["bACC"] + row["AUC"]) / 2

    pooled = {
        "bACC": balanced_accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, outer_smoothed),
    }
    pooled["mean"] = (pooled["bACC"] + pooled["AUC"]) / 2
    return {"config": CONFIG, "pooled_nested": pooled, "folds": fold_rows}, outer_smoothed


def train(data_dir: Path, output_dir: Path, run_nested: bool = False) -> None:
    """训练、OOF 选阈值、保存最终模型及可复现的预测明细。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "training_features.npz"
    metadata_path = output_dir / "training_metadata.csv"
    # 有效缓存可让耗时特征提取与嵌套验证分步执行；旧版 33 维缓存不会误用。
    if cache_path.exists() and metadata_path.exists():
        cached = np.load(cache_path)
        if cached["X"].shape[1] == N_FEATURES:
            X, y = cached["X"], cached["y"]
            subjects = cached["subjects"].astype(str)
            feature_names = list(cached["feature_names"])
            metadata = pd.read_csv(metadata_path)
            metadata["subject"] = metadata["subject"].astype(str).str.zfill(3)
            print("已加载缓存的训练特征。", flush=True)
        else:
            X, y, subjects, metadata, feature_names = build_training_features(data_dir)
    else:
        X, y, subjects, metadata, feature_names = build_training_features(data_dir)
    np.savez_compressed(cache_path, X=X, y=y, subjects=subjects, feature_names=np.asarray(feature_names))
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"训练集：{X.shape[0]} 窗，{X.shape[1]} 维，{len(np.unique(subjects))} 条独立记录。", flush=True)

    splitter = GroupKFold(n_splits=N_OUTER_FOLDS)
    oof_raw = np.full(len(y), np.nan)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, subjects)):
        tree, ridge = fit_ensemble(X[train_idx, :N_CURRENT_FEATURES], y[train_idx])
        augmented = fit_augmented(X[train_idx], y[train_idx])
        oof_raw[valid_idx] = ensemble_decision(
            tree, ridge, augmented, X[valid_idx, :N_CURRENT_FEATURES], X[valid_idx]
        )
        print(f"OOF 第 {fold + 1}/{N_OUTER_FOLDS} 折完成。", flush=True)
    oof_smoothed = causal_smooth(oof_raw, subjects)

    # 两套独立记录分组的 OOF 共同选择部署阈值，降低单一划分的偶然性。
    nested, nested_smoothed = None, None
    if run_nested:
        nested, nested_smoothed = nested_validation(X, y, subjects)
        threshold = best_threshold(np.concatenate([y, y]), np.concatenate([oof_smoothed, nested_smoothed]))
        nested["deployment_threshold_repeated_oof"] = threshold
    else:
        threshold = best_threshold(y, oof_smoothed)

    oof_metrics = {
        "bACC": balanced_accuracy_score(y, oof_smoothed >= threshold),
        "AUC": roc_auc_score(y, oof_smoothed),
        "threshold": threshold,
        "note": ("阈值为两套分组 OOF 的共同结果；泛化估计以 nested_metrics.json 为准。"
                 if nested is not None else "阈值在单套 OOF 上选择；加入 --nested 可使用双分组阈值并获得严格复核。"),
    }
    oof_metrics["mean"] = (oof_metrics["bACC"] + oof_metrics["AUC"]) / 2

    # 阈值确定后用全部独立记录重训最终部署集成。
    final_tree, final_ridge = fit_ensemble(X[:, :N_CURRENT_FEATURES], y)
    final_augmented = fit_augmented(X, y)
    artifact = {
        "format_version": 3,
        "tree": final_tree,
        "ridge": final_ridge,
        "augmented": final_augmented,
        "tree_columns": TREE_COLUMNS,
        "threshold": threshold,
        "config": CONFIG,
        "feature_names": feature_names,
        "training_records": sorted(np.unique(subjects).tolist()),
        "training_windows": int(len(y)),
    }
    model_path = output_dir / "eeg_decoder.joblib"
    joblib.dump(artifact, model_path, compress=3)

    np.savez_compressed(
        output_dir / "training_features.npz",
        X=X, y=y, subjects=subjects, feature_names=np.asarray(feature_names),
    )
    prediction_frame = metadata.copy()
    prediction_frame["score"] = oof_smoothed
    prediction_frame["prediction"] = (oof_smoothed >= threshold).astype(int)
    prediction_frame.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if nested is not None:
        (output_dir / "nested_metrics.json").write_text(
            json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(nested["folds"]).to_csv(
            output_dir / "nested_fold_metrics.csv", index=False, encoding="utf-8-sig"
        )
        np.savez_compressed(output_dir / "nested_oof_scores.npz", score=nested_smoothed)
        print("嵌套验证：", nested["pooled_nested"], flush=True)

    print("OOF 验证：", oof_metrics, flush=True)
    print(f"模型已保存：{model_path}", flush=True)


# ----------------------------- 严格因果流式推理 -----------------------------
class CausalEEGDecoder:
    """
    可按任意 chunk 大小顺序输入单通道 EEG（µV）的解码器。

    每条新记录必须新建实例或调用 reset()。
    update() 每累计 2 秒产生一个结果；无效窗 valid=False 且不更新任何基线。
    """

    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        if artifact.get("format_version") != 3:
            raise ValueError("模型文件格式版本不兼容，请使用 v3 模型。")
        self.tree = artifact["tree"]
        self.ridge = artifact["ridge"]
        self.augmented = artifact["augmented"]
        self.threshold = float(artifact["threshold"])
        self.config = artifact["config"]
        self.fir = fir_coefficients()
        self.fir5 = fir5_coefficients()
        self.reset()

    def reset(self) -> None:
        """清空上一条记录的全部状态，防止跨记录泄漏。"""
        self.filter_zi = np.zeros(FIR_TAPS - 1, dtype=np.float64)
        self.filter5_zi = np.zeros(FIR5_TAPS - 1, dtype=np.float64)
        self.raw_buffer = np.empty(0, dtype=np.float64)
        self.filtered_buffer = np.empty(0, dtype=np.float64)
        self.filtered5_buffer = np.empty(0, dtype=np.float64)
        self.bad_buffer = np.empty(0, dtype=bool)
        self.sample_count = 0
        self.next_decision_sample = RECEPTIVE_SAMPLES
        self.baseline = None
        self.smoothed_score = None

    def update(self, chunk: Iterable[float], bad_mask: Iterable[bool] | None = None) -> list[dict]:
        """输入按时间到达的样本，返回本次新产生的 0 个或多个判断结果。"""
        x = np.asarray(list(chunk) if not isinstance(chunk, np.ndarray) else chunk, dtype=np.float64).reshape(-1)
        if len(x) == 0:
            return []
        if not np.isfinite(x).all():
            raise ValueError("输入包含 NaN/Inf；请先将这些样本通过 bad_mask 标记并替换为有限值。")
        if bad_mask is None:
            bad = np.zeros(len(x), dtype=bool)
        else:
            bad = np.asarray(list(bad_mask) if not isinstance(bad_mask, np.ndarray) else bad_mask, dtype=bool).reshape(-1)
            if len(bad) != len(x):
                raise ValueError("bad_mask 与 chunk 长度必须一致。")

        # 两个 FIR 都只保存有限卷积状态；AR/DFA 另保留最多 6 秒原始信号。
        filtered, self.filter_zi = signal.lfilter(self.fir, [1.0], x, zi=self.filter_zi)
        filtered5, self.filter5_zi = signal.lfilter(self.fir5, [1.0], x, zi=self.filter5_zi)
        self.raw_buffer = np.concatenate([self.raw_buffer, x])
        self.filtered_buffer = np.concatenate([self.filtered_buffer, filtered])
        self.filtered5_buffer = np.concatenate([self.filtered5_buffer, filtered5])
        self.bad_buffer = np.concatenate([self.bad_buffer, bad])
        self.sample_count += len(x)
        results = []

        while self.sample_count >= self.next_decision_sample:
            end = self.next_decision_sample
            # 所有分支使用同一个绝对判断时刻，保证任意分块结果一致。
            future_in_buffer = self.sample_count - end
            raw_end = len(self.raw_buffer) - future_in_buffer
            raw6 = self.raw_buffer[raw_end-RECEPTIVE_SAMPLES:raw_end]
            segment_end = len(self.filtered_buffer) - future_in_buffer
            segment = self.filtered_buffer[segment_end-FEATURE_SAMPLES:segment_end]
            segment5_end = len(self.filtered5_buffer) - future_in_buffer
            segment5 = self.filtered5_buffer[segment5_end-FEATURE5_SAMPLES:segment5_end]
            bad_end = len(self.bad_buffer) - future_in_buffer
            receptive_bad = self.bad_buffer[bad_end-RECEPTIVE_SAMPLES:bad_end]
            result = {"end_sample": int(end), "end_time_seconds": end/SFREQ, "valid": True}

            if (len(raw6) != RECEPTIVE_SAMPLES or len(segment) != FEATURE_SAMPLES
                    or len(segment5) != FEATURE5_SAMPLES or receptive_bad.any()):
                # 无效窗不输出伪造分数，也不更新个体基线或平滑状态。
                result.update({"valid": False, "label": None, "state": None,
                               "score": None, "task_probability": None})
            else:
                feature, _ = extract_third_features(segment, raw6, segment5)
                feature = feature.astype(np.float32)
                residual = np.zeros_like(feature) if self.baseline is None else feature-self.baseline
                raw_score = float(ensemble_decision(
                    self.tree, self.ridge, self.augmented,
                    residual[None, :N_CURRENT_FEATURES], residual.reshape(1, -1)
                )[0])
                if self.smoothed_score is None:
                    self.smoothed_score = raw_score
                else:
                    self.smoothed_score = ((1-SMOOTH_ALPHA)*self.smoothed_score
                                           + SMOOTH_ALPHA*raw_score)
                score = float(self.smoothed_score)
                label = int(score >= self.threshold)
                result.update({
                    "label": label, "state": "task" if label == 1 else "relax", "score": score,
                    # 仅是单调 logistic 映射，AUC 请直接使用 score。
                    "task_probability": float(expit(score-self.threshold)),
                })

                # 先预测、后更新，当前窗绝不进入自己的基线。
                if self.baseline is None:
                    self.baseline = feature.astype(np.float64).copy()
                else:
                    self.baseline = (1-ADAPT_ALPHA)*self.baseline + ADAPT_ALPHA*feature

            results.append(result)
            self.next_decision_sample += STRIDE_SAMPLES

        # 各缓冲区只保留下一次判断所需的有限历史。
        self.raw_buffer = self.raw_buffer[-RECEPTIVE_SAMPLES:]
        self.filtered_buffer = self.filtered_buffer[-FEATURE_SAMPLES:]
        self.filtered5_buffer = self.filtered5_buffer[-FEATURE5_SAMPLES:]
        self.bad_buffer = self.bad_buffer[-RECEPTIVE_SAMPLES:]
        return results


def predict_bdf(model_path: Path, bdf_path: Path, output_path: Path, chunk_seconds: float = 1.0) -> pd.DataFrame:
    """以固定小块读取整条 BDF，演示真实的按时间顺序分块推理。"""
    raw, x = read_single_channel_bdf(bdf_path)
    bad = bad_annotation_mask(raw)
    decoder = CausalEEGDecoder(model_path)
    chunk_samples = max(1, int(round(chunk_seconds * SFREQ)))
    results = []
    for start in range(0, len(x), chunk_samples):
        end = min(len(x), start + chunk_samples)
        results.extend(decoder.update(x[start:end], bad[start:end]))
    frame = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"共输出 {len(frame)} 窗，其中有效窗 {int(frame.valid.sum())} 个。")
    print(f"预测已保存：{output_path}")
    return frame


def stream_demo(model_path: Path, bdf_path: Path, chunk_samples: int, limit: int) -> None:
    """最小流式示例：逐块输入并逐次打印 JSON，不依赖事件标签。"""
    raw, x = read_single_channel_bdf(bdf_path)
    bad = bad_annotation_mask(raw)
    decoder = CausalEEGDecoder(model_path)
    printed = 0
    for start in range(0, len(x), chunk_samples):
        end = min(len(x), start + chunk_samples)
        for result in decoder.update(x[start:end], bad[start:end]):
            print(json.dumps(result, ensure_ascii=False))
            printed += 1
            if printed >= limit:
                return


# ----------------------------- 命令行入口 -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单通道 EEG 静息/认知负荷因果解码")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并保存模型")
    train_parser.add_argument("--data-dir", type=Path, required=True, help="包含编号子目录的数据根目录")
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--nested", action="store_true", help="额外执行嵌套交叉验证")

    for command, help_text in [("predict", "分块预测完整 BDF"), ("stream-demo", "打印最小流式输出")]:
        p = subparsers.add_parser(command, help=help_text)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--bdf", type=Path, required=True)
        if command == "predict":
            p.add_argument("--output", type=Path, default=Path("predictions.csv"))
            p.add_argument("--chunk-seconds", type=float, default=1.0)
        else:
            p.add_argument("--chunk-samples", type=int, default=137, help="故意使用非整秒块也可运行")
            p.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args.data_dir, args.output_dir, args.nested)
    elif args.command == "predict":
        predict_bdf(args.model, args.bdf, args.output, args.chunk_seconds)
    elif args.command == "stream-demo":
        stream_demo(args.model, args.bdf, args.chunk_samples, args.limit)


if __name__ == "__main__":
    main()
