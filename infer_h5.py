"""
Run ECGFounder inference on a directory of H5 files (no labels needed).

Supports two H5 formats:
  1. Grouped /signals/ format (new):
       python infer_h5.py --h5_dir /path/to/h5/ --leads I II ...
  2. Legacy Xy matrix format (old CAISR/Morgoth2 PSG files):
       python infer_h5.py --h5_dir /path/to/h5/ --xy_channel_idx 13 --xy_fs 200

In legacy Xy mode the recording is segmented into windows (default 10 s / no overlap)
and each window produces one row in the output CSV.

Output CSV columns:
  file               — path to the H5 file
  window_start_sec   — (Xy mode only) window start time in seconds
  window_end_sec     — (Xy mode only) window end time in seconds
  <task_0> … <task_N> — sigmoid probability for each class
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from net1d import Net1D
from dataset_h5 import ECG_H5_InferenceDataset, ECG_XyMatrix_InferenceDataset


def build_model(in_channels: int, n_classes: int, device: torch.device) -> torch.nn.Module:
    model = Net1D(
        in_channels=in_channels,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes,
    )
    return model.to(device)


def load_tasks(tasks_path: str) -> list[str]:
    with open(tasks_path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir",          required=True,  help="Directory containing H5 files")
    parser.add_argument("--ckpt",            default="./checkpoint/1_lead_ECGFounder.pth")
    parser.add_argument("--tasks",           default="./tasks.txt")
    parser.add_argument("--out",             default="./results",
                        help="Output directory; each H5 file gets its own <stem>.csv")
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--pattern",         default="*.h5",
                        help="Glob pattern to find H5 files inside h5_dir")
    parser.add_argument("--rewrite",         action="store_true",
                        help="Reprocess all files. Default: skip files whose output CSV already exists.")

    # ── Legacy Xy matrix mode ──────────────────────────────────────────────
    parser.add_argument("--xy_channel_idx",  type=int, default=None,
                        help="Channel index in legacy Xy matrix to use as ECG "
                             "(e.g. 13). When set, Xy matrix mode is activated.")
    parser.add_argument("--xy_fs",           type=int, default=200,
                        help="Sampling rate of the legacy Xy recording in Hz (default 200)")
    parser.add_argument("--window_sec",      type=int, default=10,
                        help="Segmentation window length in seconds (default 10)")
    parser.add_argument("--stride_sec",      type=int, default=10,
                        help="Stride between consecutive windows in seconds (default 10)")

    # ── /signals/ grouped format mode ─────────────────────────────────────
    parser.add_argument("--leads",           nargs="+", default=None,
                        help="Channel name(s) under /signals/ to load. "
                             "Single-channel example: --leads ECG")
    parser.add_argument("--tile_to_12",      action="store_true",
                        help="Repeat a single-channel signal 12 times to fit "
                             "the 12-lead pretrained model.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tasks = load_tasks(args.tasks)
    n_classes = len(tasks)
    print(f"Tasks: {n_classes} classes")

    # ── Discover files, optionally skip already-done ones ─────────────────
    all_files = sorted(str(p) for p in Path(args.h5_dir).glob(args.pattern))
    if not args.rewrite:
        pending = [p for p in all_files
                   if not (out_dir / (Path(p).stem + ".csv")).exists()]
        skipped = len(all_files) - len(pending)
        if skipped:
            print(f"Skipping {skipped} file(s) with existing results "
                  f"(use --rewrite to reprocess).")
        all_files = pending
    if not all_files:
        print("Nothing to process.")
        return

    # ── Select dataset and model input channels ────────────────────────────
    xy_mode = args.xy_channel_idx is not None

    if xy_mode:
        dataset = ECG_XyMatrix_InferenceDataset(
            h5_files=all_files,
            channel_idx=args.xy_channel_idx,
            fs=args.xy_fs,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
        n_leads = 1
        print(f"Mode: legacy Xy matrix  |  channel_idx={args.xy_channel_idx}  "
              f"fs={args.xy_fs} Hz  window={args.window_sec}s  stride={args.stride_sec}s")
        print(f"Total windows: {len(dataset)}")
    else:
        dataset = ECG_H5_InferenceDataset(
            h5_files=all_files,
            lead_names=args.leads,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
        n_leads = 1 if (args.leads and len(args.leads) == 1 and not args.tile_to_12) else 12
        print(f"Mode: /signals/ grouped  |  leads={args.leads}  tile_to_12={args.tile_to_12}  "
              f"window={args.window_sec}s  stride={args.stride_sec}s")
        print(f"Total windows: {len(dataset)}")

    model = build_model(in_channels=n_leads, n_classes=n_classes, device=device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}  (in_channels={n_leads})")

    window_mode = xy_mode or args.window_sec is not None
    window_len  = args.window_sec

    total_rows = 0
    # ── Per-file inference with file-level progress bar ───────────────────────
    for h5_path in tqdm(all_files, desc="Files"):
        stem    = Path(h5_path).stem
        out_csv = out_dir / f"{stem}.csv"

        if xy_mode:
            file_dataset = ECG_XyMatrix_InferenceDataset(
                h5_files=[h5_path],
                channel_idx=args.xy_channel_idx,
                fs=args.xy_fs,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
            )
        else:
            file_dataset = ECG_H5_InferenceDataset(
                h5_files=[h5_path],
                lead_names=args.leads,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
            )

        loader = DataLoader(file_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=min(os.cpu_count(), 4))

        probs_list, starts_list, ends_list = [], [], []

        with torch.no_grad():
            for batch in loader:
                if window_mode:
                    signals, _, starts = batch
                    starts_list.extend(starts.numpy().tolist())
                    ends_list.extend((starts + window_len).numpy().tolist())
                else:
                    signals, _ = batch

                if args.tile_to_12 and signals.shape[1] == 1:
                    signals = signals.repeat(1, 12, 1)

                logits = model(signals.to(device))
                probs_list.append(F.sigmoid(logits).cpu().numpy())

        if not probs_list:
            continue

        df = pd.DataFrame(np.concatenate(probs_list, axis=0), columns=tasks)
        if window_mode:
            df.insert(0, "window_start_sec", starts_list)
            df.insert(1, "window_end_sec",   ends_list)

        df.to_csv(out_csv, index=False, float_format="%.5f")
        total_rows += len(df)

    print(f"Done. {total_rows} total predictions across {len(all_files)} file(s) → {out_dir}/")


if __name__ == "__main__":
    main()
