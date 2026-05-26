"""
H5-based ECG datasets for ECGFounder.

Expected H5 file layout (same as cohort_h5_conversion_reference.py):
    attrs:
        sampling_rate  (int)   — original sampling rate of the recording
    /signals/
        I              (N,) or (N,1)  float32
        II
        III
        aVR
        aVL
        aVF
        V1 … V6
    /annotations/              (optional)
        <label_name>   (1,)   float32

labels_df conventions (same as dataset.py):
    column 1   — H5 file name (relative to h5_dir, without extension or with .h5)
    last col   — label for classification tasks
    second-to-last col — label for regression tasks

Inference-only (no labels):
    Use ECG_H5_InferenceDataset — accepts a list of H5 file paths directly.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from util import filter_bandpass


TARGET_LENGTH = 5000          # 500 Hz × 10 s
TARGET_FS     = 500

STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _resolve_lead_name(sig_group, name: str) -> str | None:
    """Return the actual key for `name` in sig_group, or None if not found.

    Falls back to the first key that starts with ``name`` (case-insensitive)
    when an exact match is absent.
    """
    if name in sig_group:
        return name
    prefix = name.lower()
    for key in sig_group.keys():
        if key.lower().startswith(prefix):
            return key
    return None


def _has_leads(h5_path: str, lead_names: list[str]) -> bool:
    """Return True if all lead_names can be resolved in the file's /signals/ group."""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'signals' not in f:
                return False
            sig_group = f['signals']
            return all(_resolve_lead_name(sig_group, n) is not None for n in lead_names)
    except Exception:
        return False


def _load_leads_from_h5(h5_path: str, lead_names: list[str]) -> tuple[np.ndarray, int]:
    """Return (data, fs) where data has shape (n_leads, n_samples)."""
    with h5py.File(h5_path, "r") as f:
        fs = int(f.attrs["sampling_rate"])
        sig_group = f["signals"]
        arrays = []
        for name in lead_names:
            key = _resolve_lead_name(sig_group, name)
            if key is None:
                raise KeyError(f"No signal matching '{name}' (or '{name}*') in {list(sig_group.keys())}")
            arr = sig_group[key][:].reshape(-1)
            arrays.append(arr)
    data = np.stack(arrays, axis=0).astype(np.float64)   # (n_leads, N)
    return data, fs


def _resample(ts: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample multichannel signal (n_leads, N) from fs_in to fs_out."""
    if fs_in == fs_out:
        return ts
    if fs_in == 2 * fs_out:
        return ts[:, ::2]
    n_in  = ts.shape[1]
    n_out = int(round(n_in * fs_out / fs_in))
    t_in  = np.linspace(0, n_in / fs_in, num=n_in,  endpoint=True)
    t_out = np.linspace(0, n_in / fs_in, num=n_out, endpoint=True)
    out   = np.zeros((ts.shape[0], n_out), dtype=ts.dtype)
    for i in range(ts.shape[0]):
        f = interp1d(t_in, ts[i], kind='linear', bounds_error=False,
                     fill_value=(ts[i, 0], ts[i, -1]))
        out[i] = f(t_out)
    return out


def _pad_or_crop(ts: np.ndarray, length: int) -> np.ndarray:
    """Ensure the time dimension equals `length`."""
    n = ts.shape[1]
    if n >= length:
        return ts[:, :length]
    pad = np.zeros((ts.shape[0], length - n), dtype=ts.dtype)
    return np.concatenate([ts, pad], axis=1)


def _z_score(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + 1e-8)


class ECG_H5_InferenceDataset(Dataset):
    """
    Inference-only dataset — no labels required.

    Accepts either:
      - a list of full H5 file paths, or
      - h5_dir + a glob pattern (default '*.h5') to auto-discover files.

    Sliding-window mode (window_sec is not None):
      __getitem__ returns (signal, file_path, window_start_sec)
    Full-file mode (window_sec is None):
      __getitem__ returns (signal, file_path)
    """

    def __init__(self, h5_files: list[str] | None = None,
                 h5_dir: str | None = None,
                 pattern: str = '*.h5',
                 lead_names: list[str] | None = None,
                 window_sec: int | None = None,
                 stride_sec: int | None = None):
        if h5_files is not None:
            self.file_paths = list(h5_files)
        elif h5_dir is not None:
            from pathlib import Path
            self.file_paths = sorted(str(p) for p in Path(h5_dir).glob(pattern))
        else:
            raise ValueError("Provide either h5_files or h5_dir.")

        self.lead_names = lead_names if lead_names is not None else STANDARD_LEADS
        self.window_sec = window_sec
        self.stride_sec = stride_sec if stride_sec is not None else window_sec

        # filter out files that don't contain the required leads
        before = len(self.file_paths)
        self.file_paths = [p for p in self.file_paths if _has_leads(p, self.lead_names)]
        skipped = before - len(self.file_paths)
        if skipped:
            print(f"[ECG_H5_InferenceDataset] Skipped {skipped} file(s) with no matching ECG channel.")

        # sliding-window mode: pre-compute (path, start_sample, orig_fs) for every window
        if window_sec is not None:
            self._segments: list[tuple[str, int, int]] = []
            for path in self.file_paths:
                with h5py.File(path, 'r') as f:
                    fs = int(f.attrs["sampling_rate"])
                    sig_group = f["signals"]
                    key = _resolve_lead_name(sig_group, self.lead_names[0])
                    n_samples = sig_group[key].shape[0]
                win = int(window_sec * fs)
                stride = int(self.stride_sec * fs)
                start = 0
                while start + win <= n_samples:
                    self._segments.append((path, start, fs))
                    start += stride
        else:
            self._segments = None

    def __len__(self):
        return len(self._segments) if self._segments is not None else len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self._segments is not None:
            path, start, fs = self._segments[idx]
            win = int(self.window_sec * fs)
            with h5py.File(path, 'r') as f:
                sig_group = f["signals"]
                arrays = []
                for name in self.lead_names:
                    key = _resolve_lead_name(sig_group, name)
                    if key is None:
                        raise KeyError(f"No signal matching '{name}' in {list(sig_group.keys())}")
                    arrays.append(sig_group[key][start:start + win].reshape(-1))
            data = np.stack(arrays, axis=0).astype(np.float64)
            data = np.nan_to_num(data, nan=0.0)
            data = filter_bandpass(data, fs)
            data = _resample(data, fs, TARGET_FS)
            data = _pad_or_crop(data, TARGET_LENGTH)
            data = _z_score(data)
            return torch.FloatTensor(data), path, float(start) / fs

        path = self.file_paths[idx]
        data, fs = _load_leads_from_h5(path, self.lead_names)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, fs)
        data = _resample(data, fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        return torch.FloatTensor(data), path


class ECG_H5_12lead_cls_Dataset(Dataset):
    """12-lead ECG → classification label, reading from grouped H5 files."""

    def __init__(self, h5_dir: str, labels_df, lead_names: list[str] | None = None):
        """
        Args:
            h5_dir    : directory that contains .h5 files.
            labels_df : DataFrame; column[1] = filename, last column = label array.
            lead_names: lead names stored under /signals/ in each H5 file.
                        Defaults to the standard 12-lead order.
        """
        self.h5_dir    = h5_dir.rstrip('/') + '/'
        self.labels_df = labels_df
        self.lead_names = lead_names if lead_names is not None else STANDARD_LEADS

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = str(self.labels_df.iloc[idx, 1])
        if not file_name.endswith('.h5'):
            file_name += '.h5'
        labels = self.labels_df.iloc[idx, -1].astype(np.float32)

        data, fs = _load_leads_from_h5(self.h5_dir + file_name, self.lead_names)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, fs)
        data = _resample(data, fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        signal = torch.FloatTensor(data)

        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return signal, labels


class ECG_H5_12lead_reg_Dataset(Dataset):
    """12-lead ECG → regression label, reading from grouped H5 files."""

    def __init__(self, h5_dir: str, labels_df, lead_names: list[str] | None = None):
        self.h5_dir     = h5_dir.rstrip('/') + '/'
        self.labels_df  = labels_df
        self.lead_names = lead_names if lead_names is not None else STANDARD_LEADS

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = str(self.labels_df.iloc[idx, 1])
        if not file_name.endswith('.h5'):
            file_name += '.h5'
        label = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([label], dtype=torch.float32)

        data, fs = _load_leads_from_h5(self.h5_dir + file_name, self.lead_names)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, fs)
        data = _resample(data, fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        signal = torch.FloatTensor(data)
        return signal, labels


class ECG_H5_1lead_cls_Dataset(Dataset):
    """Single-lead (Lead I) ECG → classification label, reading from grouped H5 files."""

    def __init__(self, h5_dir: str, labels_df, lead_name: str = 'I'):
        self.h5_dir    = h5_dir.rstrip('/') + '/'
        self.labels_df = labels_df
        self.lead_names = [lead_name]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = str(self.labels_df.iloc[idx, 1])
        if not file_name.endswith('.h5'):
            file_name += '.h5'
        labels = self.labels_df.iloc[idx, -1].astype(np.float32)

        data, fs = _load_leads_from_h5(self.h5_dir + file_name, self.lead_names)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, fs)
        data = _resample(data, fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        signal = torch.FloatTensor(data)

        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return signal, labels


class ECG_H5_1lead_reg_Dataset(Dataset):
    """Single-lead (Lead I) ECG → regression label, reading from grouped H5 files."""

    def __init__(self, h5_dir: str, labels_df, lead_name: str = 'I'):
        self.h5_dir     = h5_dir.rstrip('/') + '/'
        self.labels_df  = labels_df
        self.lead_names = [lead_name]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = str(self.labels_df.iloc[idx, 1])
        if not file_name.endswith('.h5'):
            file_name += '.h5'
        label = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([label], dtype=torch.float32)

        data, fs = _load_leads_from_h5(self.h5_dir + file_name, self.lead_names)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, fs)
        data = _resample(data, fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        signal = torch.FloatTensor(data)
        return signal, labels


class ECG_XyMatrix_InferenceDataset(Dataset):
    """
    Legacy Xy matrix H5 files (shape: n_channels × n_samples).
    Extracts one channel by index, segments into fixed-length windows,
    and applies the same preprocessing pipeline.

    __getitem__ returns (signal, file_path, window_start_sec)
      signal: FloatTensor of shape (1, TARGET_LENGTH)
    """

    def __init__(self,
                 h5_files: list[str] | None = None,
                 h5_dir: str | None = None,
                 pattern: str = '*.h5',
                 channel_idx: int = 13,
                 fs: int = 200,
                 window_sec: int = 10,
                 stride_sec: int = 10):
        if h5_files is not None:
            self.file_paths = list(h5_files)
        elif h5_dir is not None:
            from pathlib import Path
            self.file_paths = sorted(str(p) for p in Path(h5_dir).glob(pattern))
        else:
            raise ValueError("Provide either h5_files or h5_dir.")

        self.channel_idx = channel_idx
        self.fs = fs
        self.window_samples = window_sec * fs
        self.stride_samples = stride_sec * fs

        # Pre-compute (file_path, start_sample) for every window
        self.segments: list[tuple[str, int]] = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                n_samples = f['Xy'].shape[1]
            start = 0
            while start + self.window_samples <= n_samples:
                self.segments.append((path, start))
                start += self.stride_samples

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path, start = self.segments[idx]
        end = start + self.window_samples

        with h5py.File(path, 'r') as f:
            raw = f['Xy'][self.channel_idx, start:end].astype(np.float64)

        data = raw[np.newaxis, :]          # (1, window_samples)
        data = np.nan_to_num(data, nan=0.0)
        data = filter_bandpass(data, self.fs)
        data = _resample(data, self.fs, TARGET_FS)
        data = _pad_or_crop(data, TARGET_LENGTH)
        data = _z_score(data)
        signal = torch.FloatTensor(data)

        start_sec = float(start) / self.fs
        return signal, path, start_sec
