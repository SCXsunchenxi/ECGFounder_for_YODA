"""
Readable reference code for cohort-specific raw EDF/XML/TXT -> prepared H5 conversion.

Old files using the code pieces:

- /home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py
- /home/wolfgang/repos/sleep_cognition/nsrr_functions.py
- /home/wolfgang/repos/sleep_cognition/mros_sof_annotations_parser.py
- /home/wolfgang/repos/sleep_cognition/step2_prepared_data_shhs.py
- /home/wolfgang/repos/sleep_cognition/step2_prepared_data_koges.py
- /home/wolfgang/repos/CAISR_internal/prepared_data/prepare_data.py
- /home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_shhs.ipynb
- /home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_mesa.ipynb
- /home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_mros.ipynb
- /home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_penn.ipynb
- /home/wolfgang/repos/caisr_dev/code_from_thijs/Penn_human_IRA.py

Notes
-----
- This is a readable reference, not a byte-for-byte reproduction of every historical script.
- It uses the newer grouped prepared-H5 layout:
    attrs: sampling_rate, unit_voltage
    /signals/<channel>
    /annotations/<label>
- Historical CAISR notebooks often wrote a single legacy "Xy" matrix instead.
- SOF does not appear to have a dedicated standalone raw->H5 converter in the searched files.
  The SOF function below is therefore an inference from the explicitly shared
  MrOS/SOF native XML parser plus the very similar MrOS channel normalization.
- Amplitude scaling differs between historical projects. By default this reference keeps
  MNE's EDF amplitudes in volts. Pass `to_uv=True` if you want the older "sleep_cognition"
  style for biopotential channels.

Stage coding used in the found sources
--------------------------------------
- 1: N3 / Stage 3-4
- 2: N2
- 3: N1
- 4: REM
- 5: Wake
- 6: Movement (occasionally present in NSRR)
- 9: Unscored

Respiratory coding used in the found sources
--------------------------------------------
- 0: no event / artifact / ignored class
- 1: obstructive apnea
- 2: central apnea
- 3: mixed apnea
- 4: hypopnea
- 5: RERA / respiratory-effort-related arousal

Limb coding used in the found sources
-------------------------------------
- 0: no event
- 1: isolated limb movement
- 2: periodic limb movement
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


FS_TARGET = 200


@dataclass
class PreparedStudy:
    signals: pd.DataFrame
    annotations: pd.DataFrame
    sampling_rate: int
    unit_voltage: str
    notes: str = ""


def save_prepared_h5(path: str | Path, study: PreparedStudy) -> None:
    """Save grouped prepared H5 in the newer /signals + /annotations format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["sampling_rate"] = int(study.sampling_rate)
        f.attrs["unit_voltage"] = study.unit_voltage
        if study.notes:
            f.attrs["notes"] = study.notes

        signals_group = f.create_group("signals")
        for name in study.signals.columns:
            values = study.signals[name].to_numpy(dtype=np.float32).reshape(-1, 1)
            signals_group.create_dataset(name, data=values, dtype="float32", compression="gzip")

        if not study.annotations.empty:
            ann_group = f.create_group("annotations")
            for name in study.annotations.columns:
                values = study.annotations[name].to_numpy(dtype=np.float32).reshape(-1, 1)
                ann_group.create_dataset(name, data=values, dtype="float32", compression="gzip")


def load_prepared_h5(
    path: str | Path,
    signals_to_load: list[str] | None = None,
    annotations_to_load: list[str] | None = None,
) -> PreparedStudy:
    """Load grouped prepared H5, adapted from sleep_cognition/load_data.py."""
    path = Path(path)
    signals = pd.DataFrame()
    annotations = pd.DataFrame()

    with h5py.File(path, "r") as f:
        signals_group = f["signals"]
        signal_names = list(signals_group.keys()) if signals_to_load is None else signals_to_load
        for name in signal_names:
            signals[name] = signals_group[name][:].reshape(-1)

        if "annotations" in f:
            ann_group = f["annotations"]
            ann_names = list(ann_group.keys()) if annotations_to_load is None else annotations_to_load
            for name in ann_names:
                annotations[name] = ann_group[name][:].reshape(-1)

        notes = f.attrs.get("notes", "")
        if isinstance(notes, bytes):
            notes = notes.decode()

        return PreparedStudy(
            signals=signals.astype(float),
            annotations=annotations.astype(float),
            sampling_rate=int(f.attrs["sampling_rate"]),
            unit_voltage=str(f.attrs["unit_voltage"]),
            notes=str(notes),
        )


def _read_edf_dataframe(path_edf: str | Path) -> tuple[pd.DataFrame, float]:
    raw = mne.io.read_raw_edf(str(path_edf), stim_channel=None, preload=False, verbose=False)
    fs = float(raw.info["sfreq"])
    channels = [c.lower() for c in raw.info["ch_names"]]
    values = raw.get_data(picks=raw.info["ch_names"]).T
    return pd.DataFrame(values, columns=channels), fs


def _rename_if_present(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    mapping_present = {old: new for old, new in mapping.items() if old in df.columns}
    if mapping_present:
        df = df.rename(columns=mapping_present)
    return df


def _ensure_column(df: pd.DataFrame, name: str, fill_value: float = 0.0) -> pd.DataFrame:
    if name not in df.columns:
        df[name] = fill_value
    return df


def _first_matching(columns: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        for col in columns:
            if pattern in col:
                return col
    return None


def _resample_signals_and_annotations(
    signals: pd.DataFrame,
    annotations: pd.DataFrame,
    fs_in: float,
    fs_out: int = FS_TARGET,
    nearest_signal_columns: tuple[str, ...] = ("spo2",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(fs_in) == int(fs_out):
        return signals.reset_index(drop=True), annotations.reset_index(drop=True)

    resampled_signals = pd.DataFrame(
        resample_poly(signals.to_numpy(), fs_out, int(fs_in), axis=0),
        columns=signals.columns,
    )

    idx_signal = np.linspace(0, len(signals) - 1, len(resampled_signals)).astype(int)
    for col in nearest_signal_columns:
        if col in signals.columns:
            resampled_signals[col] = signals[col].iloc[idx_signal].to_numpy()

    idx_ann = np.linspace(0, len(annotations) - 1, len(resampled_signals)).astype(int)
    resampled_annotations = annotations.iloc[idx_ann].reset_index(drop=True)

    return resampled_signals.reset_index(drop=True), resampled_annotations


def _trim_to_multiple_of_seconds(
    signals: pd.DataFrame,
    annotations: pd.DataFrame,
    fs: int,
    seconds: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    remainder = len(signals) % (seconds * fs)
    if remainder == 0:
        return signals, annotations
    if annotations.empty:
        return signals.iloc[:-remainder].reset_index(drop=True), annotations
    return (
        signals.iloc[:-remainder].reset_index(drop=True),
        annotations.iloc[:-remainder].reset_index(drop=True),
    )


def _scale_biopotentials_to_uv(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col] * 1e6
    return df


NSRR_STAGE_MAP = {
    "rem sleep|5": 4,
    "stage 1 sleep|1": 3,
    "stage 2 sleep|2": 2,
    "stage 3 sleep|3": 1,
    "stage 4 sleep|4": 1,
    "wake|0": 5,
    "wake": 5,
    "movement|6": 6,
    "unscored|9": 9,
    "unscored": 9,
}

NSRR_RESP_MAP = {
    "obstructive apnea|obstructive apnea": 1,
    "obstructive apnea": 1,
    "central apnea|central apnea": 2,
    "central apnea": 2,
    "mixed apnea|mixed apnea": 3,
    "mixed apnea": 3,
    "mixed apnea|mixed apnea ": 3,
    "hypopnea|hypopnea": 4,
    "hypopnea": 4,
    "unsure|unsure": 4,
    "unsure": 4,
    "respiratory effort related arousal|rera": 5,
    "rera": 5,
    "respiratory artifact|respiratory artifact": 0,
    "periodic breathing|periodic breathing": 0,
}

NSRR_SHHS_RESP_MAP = dict(NSRR_RESP_MAP)
NSRR_SHHS_RESP_MAP["unsure|unsure"] = 0
NSRR_SHHS_RESP_MAP["unsure"] = 0

NSRR_LIMB_MAP = {
    "periodic leg movement - left|plm (left)": 2,
    "plm (left)": 2,
    "periodic leg movement - right|plm (right)": 2,
    "plm (right)": 2,
    "limb movement (left)": 1,
    "limb movement - left|limb movement (left)": 1,
    "limb movement (right)": 1,
    "limb movement - right|limb movement (right)": 1,
}

MROS_SOF_STAGE_MAP = {
    0: 5,
    1: 3,
    2: 2,
    3: 1,
    4: 1,
    5: 4,
    6: 9,
    9: 9,
}


def load_nsrr_annotations(
    path_xml: str | Path,
    fs: float,
    signal_len: int,
    cohort: str,
) -> pd.DataFrame:
    """Distilled from sleep_cognition/nsrr_functions.py."""
    path_xml = Path(path_xml)
    cohort = cohort.lower()
    assert cohort in {"shhs", "mesa", "mros"}

    root = ET.parse(path_xml).getroot()
    events = root.find("ScoredEvents").findall("ScoredEvent")

    stage = np.zeros(signal_len, dtype=int)
    resp = np.zeros(signal_len, dtype=int)
    arousal = np.zeros(signal_len, dtype=int)
    limb = np.zeros(signal_len, dtype=int)
    spo2_desat = np.zeros(signal_len, dtype=int)
    spo2_artifact = np.zeros(signal_len, dtype=int)

    resp_map = NSRR_SHHS_RESP_MAP if cohort == "shhs" else NSRR_RESP_MAP

    for event in events:
        event_type = event.findtext("EventType")
        concept = (event.findtext("EventConcept") or "").strip()
        concept_key = concept.lower()
        start = int(float(event.findtext("Start", default="0")) * fs)
        duration = int(float(event.findtext("Duration", default="0")) * fs)
        stop = min(signal_len, start + duration)

        if event_type == "Stages|Stages":
            if concept_key in NSRR_STAGE_MAP:
                stage[start:stop] = NSRR_STAGE_MAP[concept_key]

        elif event_type == "Respiratory|Respiratory":
            if concept_key == "spo2 artifact|spo2 artifact":
                spo2_artifact[start:stop] = 1
            elif concept_key == "spo2 desaturation|spo2 desaturation":
                baseline = int(float(event.findtext("SpO2Baseline", default="0")))
                nadir = int(float(event.findtext("SpO2Nadir", default="0")))
                spo2_desat[start:stop] = baseline - nadir
            else:
                resp[start:stop] = resp_map.get(concept_key, 0)

        elif event_type == "Arousals|Arousals":
            arousal[start:stop] = 1
            lowered = concept_key.lower()
            if "limb" in lowered:
                limb[start:stop] = 1
            if "periodic" in lowered:
                limb[start:stop] = 2
            if "respiratory" in lowered or lowered == "rera":
                resp[start:stop] = 5

        elif event_type == "Limb Movement|Limb Movement":
            limb[start:stop] = NSRR_LIMB_MAP.get(concept_key, 0)

    return pd.DataFrame(
        {
            "stage": stage,
            "resp": resp,
            "arousal": arousal,
            "limb": limb,
            "spo2_desat": spo2_desat,
            "spo2_artifact": spo2_artifact,
        }
    )


def load_mros_sof_native_xml(path_xml: str | Path, fs: float, signal_len: int) -> pd.DataFrame:
    """Distilled from sleep_cognition/mros_sof_annotations_parser.py."""
    root = ET.parse(path_xml).getroot()
    events = root.find("ScoredEvents").findall("ScoredEvent")

    stage = np.zeros(signal_len, dtype=int)
    resp = np.zeros(signal_len, dtype=int)
    arousal = np.zeros(signal_len, dtype=int)
    limb = np.zeros(signal_len, dtype=int)
    spo2_desat = np.zeros(signal_len, dtype=int)
    spo2_artifact = np.zeros(signal_len, dtype=int)

    for event in events:
        name = (event.findtext("Name") or "").strip().lower()
        start = int(float(event.findtext("Start", default="0")) * fs)
        duration = int(float(event.findtext("Duration", default="0")) * fs)
        stop = min(signal_len, start + duration)

        if name in NSRR_RESP_MAP:
            resp[start:stop] = NSRR_RESP_MAP[name]
        elif name == "spo2 artifact":
            spo2_artifact[start:stop] = 1
        elif name == "spo2 desaturation":
            spo2_desat[start:stop] = int(float(event.findtext("Desaturation", default="0")))
        elif name in NSRR_LIMB_MAP:
            limb[start:stop] = NSRR_LIMB_MAP[name]
        elif "arousal" in name or name == "rera":
            arousal[start:stop] = 1
            if "limb" in name:
                limb[start:stop] = 1
            if "periodic" in name:
                limb[start:stop] = 2
            if "respiratory" in name or name == "rera":
                resp[start:stop] = 5

    scored_stages = root.find("SleepStages").findall("SleepStage")
    stage_values = [MROS_SOF_STAGE_MAP[int(node.text)] for node in scored_stages]
    stage_values = np.repeat(np.asarray(stage_values, dtype=int), int(30 * fs))
    stage[: len(stage_values)] = stage_values[:signal_len]

    return pd.DataFrame(
        {
            "stage": stage,
            "resp": resp,
            "arousal": arousal,
            "limb": limb,
            "spo2_desat": spo2_desat,
            "spo2_artifact": spo2_artifact,
        }
    )


def convert_shhs(
    path_edf: str | Path,
    path_xml: str | Path,
    *,
    to_uv: bool = False,
    trim_to_two_seconds: bool = True,
) -> PreparedStudy:
    """Readable SHHS conversion distilled from sleep_cognition + CAISR sources."""
    signals, fs = _read_edf_dataframe(path_edf)
    annotations = load_nsrr_annotations(path_xml, fs, len(signals), cohort="shhs")

    airflow_candidates = [c for c in signals.columns if "air" in c]
    if not airflow_candidates:
        airflow_candidates = [c for c in signals.columns if "epms" in c]
    if not airflow_candidates:
        airflow_candidates = [c for c in signals.columns if "aux" in c]
    if len(airflow_candidates) == 1:
        airflow = airflow_candidates[0]
    elif len(airflow_candidates) == 2 and set(airflow_candidates) == {"airflow-0", "airflow-1"}:
        airflow = "airflow-0"
    else:
        airflow = _first_matching(airflow_candidates, ["new"]) or airflow_candidates[0]

    unused_airflow = [c for c in airflow_candidates if c != airflow]
    if unused_airflow:
        signals = signals.drop(columns=unused_airflow)

    signals = _rename_if_present(
        signals,
        {
            airflow: "resp_airflow",
            "sao2": "spo2",
            "eeg": "c4-m1",
            "eog(l)": "e1-m2",
            "eog(r)": "e2-m1",
            "thor res": "chest",
            "abdo res": "abd",
            "emg": "chin",
        },
    )

    eeg2 = _first_matching(list(signals.columns), ["eeg(sec)", "eeg2", "eeg 2", "sec"])
    if eeg2 is None:
        eeg2 = next(c for c in signals.columns if c.startswith("eeg") and c != "c4-m1")
    signals = _rename_if_present(signals, {eeg2: "c3-m2"})

    signals = signals[
        ["c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin", "abd", "chest", "resp_airflow", "spo2", "ecg"]
    ].copy()

    if to_uv:
        signals = _scale_biopotentials_to_uv(
            signals,
            ["c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin", "ecg"],
        )

    annotations = annotations[["stage", "arousal", "resp"]].copy()
    signals, annotations = _resample_signals_and_annotations(signals, annotations, fs, FS_TARGET)
    if trim_to_two_seconds:
        signals, annotations = _trim_to_multiple_of_seconds(signals, annotations, FS_TARGET, seconds=2)

    return PreparedStudy(
        signals=signals,
        annotations=annotations,
        sampling_rate=FS_TARGET,
        unit_voltage="uV" if to_uv else "V",
        notes="SHHS distilled from sleep_cognition/preprocessing_and_spectrograms.py and dataset_shhs.ipynb",
    )


def convert_mesa(
    path_edf: str | Path,
    path_xml: str | Path,
    *,
    to_uv: bool = False,
) -> PreparedStudy:
    signals, fs = _read_edf_dataframe(path_edf)
    annotations = load_nsrr_annotations(path_xml, fs, len(signals), cohort="mesa")

    for col in [
        "hr",
        "dhr",
        "oxstatus",
        "pleth",
        "eeg1_off",
        "eeg2_off",
        "eeg3_off",
        "emg_off",
        "eog-l_off",
        "eog-r_off",
        "ekg_off",
        "pos",
        "position",
        "snore",
        "pres",
    ]:
        if col in signals.columns:
            signals = signals.drop(columns=[col])

    signals = _rename_if_present(
        signals,
        {
            "eeg1": "fz-cz",
            "eeg2": "cz-oz",
            "eeg3": "c4-m1",
            "eog-l": "e1-m2",
            "eog-r": "e2-m1",
            "thor": "chest",
            "abdo": "abd",
            "emg": "chin",
            "ekg": "ecg",
            "flow": "resp_airflow",
            "therm": "airflow",
        },
    )

    signals = _ensure_column(signals, "airflow", 0.0)
    signals = signals[
        ["cz-oz", "c4-m1", "e1-m2", "e2-m1", "chin", "abd", "chest", "resp_airflow", "spo2", "ecg", "leg"]
    ].copy()

    if to_uv:
        signals = _scale_biopotentials_to_uv(
            signals,
            ["cz-oz", "c4-m1", "e1-m2", "e2-m1", "chin", "ecg", "leg"],
        )

    annotations = annotations[["stage", "arousal", "resp", "limb"]].copy()
    signals, annotations = _resample_signals_and_annotations(signals, annotations, fs, FS_TARGET)

    return PreparedStudy(
        signals=signals,
        annotations=annotations,
        sampling_rate=FS_TARGET,
        unit_voltage="uV" if to_uv else "V",
        notes="MESA distilled from sleep_cognition/preprocessing_and_spectrograms.py and dataset_mesa.ipynb",
    )


def _convert_mros_like_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_if_present(
        df,
        {
            "ecg1": "ecg l",
            "ecg2": "ecg r",
            "ecgl": "ecg l",
            "ecgr": "ecg r",
            "ecgr-ecgl": "ecg l-ecg r",
            "leg/l": "leg l",
            "leg/r": "leg r",
            "legl": "leg l",
            "legr": "leg r",
            "leg l-0": "leg l",
            "leg r-0": "leg r",
            "e1": "loc",
            "e2": "roc",
            "roc-0": "roc",
            "loc-0": "loc",
            "c4-m1": "c4-a1",
            "c3-m2": "c3-a2",
            "lchin": "l chin",
            "rchin": "r chin",
            "lchin-rchin": "l chin-r chin",
            "thoracic": "chest",
            "abdominal": "abd",
            "cannula flow": "ptaf",
            "nasal pressure": "ptaf",
            "sao2": "spo2",
        },
    )

    df = _ensure_column(df, "ptaf", 0.0)
    if "a1" not in df.columns and "m1" in df.columns:
        df = _rename_if_present(df, {"m1": "a1"})
    if "a2" not in df.columns and "m2" in df.columns:
        df = _rename_if_present(df, {"m2": "a2"})

    if "c4-a1" not in df.columns and {"c4", "a1"}.issubset(df.columns):
        df["c4-a1"] = df["c4"] - df["a1"]
    if "c3-a2" not in df.columns and {"c3", "a2"}.issubset(df.columns):
        df["c3-a2"] = df["c3"] - df["a2"]

    if "l chin" not in df.columns and "emg/l" in df.columns:
        df = _rename_if_present(df, {"emg/l": "l chin"})
    if "r chin" not in df.columns and "emg/r" in df.columns:
        df = _rename_if_present(df, {"emg/r": "r chin"})
    if "l chin-r chin" not in df.columns and {"l chin", "r chin"}.issubset(df.columns):
        df["l chin-r chin"] = df["l chin"] - df["r chin"]

    if "ecg l-ecg r" not in df.columns and {"ecg l", "ecg r"}.issubset(df.columns):
        df["ecg l-ecg r"] = df["ecg l"] - df["ecg r"]

    if {"loc", "a2"}.issubset(df.columns):
        df["e1-m2"] = df["loc"] - df["a2"]
    elif "loc" in df.columns:
        df["e1-m2"] = df["loc"]
    if {"roc", "a1"}.issubset(df.columns):
        df["e2-m1"] = df["roc"] - df["a1"]
    elif "roc" in df.columns:
        df["e2-m1"] = df["roc"]

    df = _rename_if_present(
        df,
        {
            "c4-a1": "c4-m1",
            "c3-a2": "c3-m2",
            "l chin-r chin": "chin",
            "ecg l-ecg r": "ecg",
            "leg l": "lat",
            "leg r": "rat",
        },
    )

    if "ptaf" in df.columns:
        df["resp_airflow"] = df["ptaf"]
    elif "airflow" in df.columns:
        df["resp_airflow"] = df["airflow"]
    else:
        df["resp_airflow"] = 0.0

    for col in ["hr", "dhr", "stat", "sum", "position", "c4", "c3", "a1", "a2", "ecg l", "ecg r", "loc", "roc", "l chin", "r chin"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df[
        ["c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin", "abd", "chest", "resp_airflow", "spo2", "ecg", "lat", "rat"]
    ].copy()


def convert_mros(
    path_edf: str | Path,
    path_xml: str | Path,
    *,
    annotation_format: str = "nsrr",
    to_uv: bool = False,
) -> PreparedStudy:
    signals, fs = _read_edf_dataframe(path_edf)

    if annotation_format == "nsrr":
        annotations = load_nsrr_annotations(path_xml, fs, len(signals), cohort="mros")
    elif annotation_format == "native":
        annotations = load_mros_sof_native_xml(path_xml, fs, len(signals))
    else:
        raise ValueError("annotation_format must be 'nsrr' or 'native'")

    signals = _convert_mros_like_signals(signals)

    if to_uv:
        signals = _scale_biopotentials_to_uv(
            signals,
            ["c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin", "ecg", "lat", "rat"],
        )

    annotations = annotations[["stage", "arousal", "resp", "limb"]].copy()
    signals, annotations = _resample_signals_and_annotations(signals, annotations, fs, FS_TARGET)

    return PreparedStudy(
        signals=signals,
        annotations=annotations,
        sampling_rate=FS_TARGET,
        unit_voltage="uV" if to_uv else "V",
        notes=(
            "MrOS distilled from sleep_cognition/preprocessing_and_spectrograms.py, "
            "mros_sof_annotations_parser.py, and dataset_mros.ipynb"
        ),
    )


def convert_sof(
    path_edf: str | Path,
    path_xml: str | Path,
    *,
    to_uv: bool = False,
) -> PreparedStudy:
    """
    Inference from the searched code:

    - sleep_cognition/mros_sof_annotations_parser.py explicitly states it handles both MrOS and SOF.
    - I did not find a dedicated standalone SOF raw->H5 converter.
    - This function therefore reuses the MrOS EDF normalization with the shared native XML parser.
    """
    return convert_mros(path_edf, path_xml, annotation_format="native", to_uv=to_uv)


def convert_koges(
    path_edf: str | Path,
    *,
    to_uv: bool = False,
    trim_to_two_seconds: bool = True,
) -> PreparedStudy:
    signals, fs = _read_edf_dataframe(path_edf)

    signals = _rename_if_present(
        signals,
        {
            "abdomen_pds": "abd",
            "activity_pds": "activity",
            "ekg_pds": "ecg",
            "emg_subm": "chin",
            "eog_r": "e2-m1",
            "thorax_pds": "chest",
            "flow_pds": "resp_airflow",
            "nasal_pds": "ptaf",
            "spo2_pds": "spo2",
        },
    )

    preferred = ["eeg", "ecg", "chin", "e2-m1", "abd", "chest", "resp_airflow", "ptaf", "spo2", "activity"]
    present = [col for col in preferred if col in signals.columns]
    signals = signals[present].copy()

    if to_uv:
        signals = _scale_biopotentials_to_uv(signals, ["eeg", "ecg", "chin", "e2-m1"])

    empty_annotations = pd.DataFrame(index=np.arange(len(signals)))
    signals, _ = _resample_signals_and_annotations(signals, empty_annotations, fs, FS_TARGET)
    if trim_to_two_seconds:
        signals, empty_annotations = _trim_to_multiple_of_seconds(signals, empty_annotations, FS_TARGET, seconds=2)

    return PreparedStudy(
        signals=signals,
        annotations=pd.DataFrame(index=np.arange(len(signals))),
        sampling_rate=FS_TARGET,
        unit_voltage="uV" if to_uv else "V",
        notes="KoGES distilled from sleep_cognition/step2_prepared_data_koges.py and loading_routine_koges()",
    )


def get_penn_labels_wg(path_combined_labels_csv: str | Path) -> pd.DataFrame:
    """Copied in simplified form from caisr_dev/code_from_thijs/Penn_human_IRA.py."""
    labels = pd.read_csv(path_combined_labels_csv, index_col=0, sep="\t")
    stage_map = {3: 1, 2: 2, 1: 3, 5: 4, 0: 5}
    labels = labels.apply(lambda col: col.map(stage_map))
    labels = labels.rename(columns={col: f"{col} sleep_stages" for col in labels.columns})
    return labels


def find_lights_on_lights_off(path_event_list_txt: str | Path) -> tuple[int, int, pd.Timestamp | None]:
    """Simplified from Penn_human_IRA.py and dataset_penn.ipynb."""
    lines = Path(path_event_list_txt).read_text(errors="ignore").splitlines()
    lights_off = None
    lights_on = None
    study_date = None

    for line in lines:
        if "Study Date" in line:
            study_date = pd.Timestamp(line.split("\t")[-1].strip())
        if "Lights Off" in line:
            lights_off = int(line.split("\t")[0].strip())
        elif "Lights On" in line:
            lights_on = int(line.split("\t")[0].strip())

    if lights_off is None or lights_on is None:
        raise ValueError(f"Could not find Lights Off / Lights On in {path_event_list_txt}")

    return lights_off, lights_on, study_date


def find_matching_penn_event_file(event_list_dir: str | Path, file_number: str) -> Path:
    """Readable replacement for the original loop over 'event list (i).txt' files."""
    event_list_dir = Path(event_list_dir)
    for candidate in sorted(event_list_dir.glob("*.txt")):
        first_line = candidate.read_text(errors="ignore").splitlines()[0]
        parts = first_line.split("_")
        if len(parts) > 1 and parts[1].split()[0] == str(file_number):
            return candidate
    raise FileNotFoundError(f"No Penn event-list file matched file_number={file_number}")


def _majority_vote_stage(labels: pd.DataFrame) -> pd.Series:
    mode = labels.mode(axis=1, dropna=True)
    if mode.empty:
        return pd.Series(np.nan, index=labels.index, name="stage")
    return mode.iloc[:, 0].rename("stage")


def convert_penn(
    path_edf: str | Path,
    path_combined_labels_csv: str | Path,
    *,
    path_event_list_txt: str | Path | None = None,
    event_list_dir: str | Path | None = None,
    include_majority_stage: bool = True,
    to_uv: bool = False,
) -> PreparedStudy:
    """
    Penn conversion distilled from dataset_penn.ipynb.

    The original notebook wrote a legacy H5 "Xy" matrix.
    This reference expresses the same logic in the newer grouped prepared-H5 layout.
    """
    signals, fs = _read_edf_dataframe(path_edf)
    studyid = Path(path_edf).stem
    file_number = studyid.split("_")[1]

    if path_event_list_txt is None:
        if event_list_dir is None:
            raise ValueError("Provide path_event_list_txt or event_list_dir for Penn conversion")
        path_event_list_txt = find_matching_penn_event_file(event_list_dir, file_number)

    labels = get_penn_labels_wg(path_combined_labels_csv)
    start_epoch_number = int(labels.index[0] - 1)
    end_epoch_number = int(labels.index[-1])
    labels = labels.reset_index(drop=True)

    lights_off, lights_on, study_date_annotation = find_lights_on_lights_off(path_event_list_txt)
    if end_epoch_number - start_epoch_number != len(labels):
        raise ValueError("Penn label epoch range mismatch")

    meas_date = pd.Timestamp(mne.io.read_info(str(path_edf))["meas_date"])
    if study_date_annotation is not None:
        assert study_date_annotation.hour == meas_date.hour
        assert study_date_annotation.minute == meas_date.minute
        assert study_date_annotation.second == meas_date.second

    signals = _rename_if_present(
        signals,
        {
            "ecg1": "ecg",
            "eog loc-a2": "e1-m2",
            "eog roc-a1": "e2-m1",
            "emg chin": "chin1-chin2",
            "c3-a2": "c3-m2",
            "c4-a1": "c4-m1",
            "flow therm": "airflow",
            "flow pres": "ptaf",
            "effort tho": "chest",
            "effort abd": "abd",
        },
    )
    signals["cflow"] = 0.0

    signals = signals[
        ["c3-m2", "c4-m1", "e1-m2", "chin1-chin2", "abd", "chest", "airflow", "ptaf", "cflow", "spo2", "ecg"]
    ].copy()

    if to_uv:
        signals = _scale_biopotentials_to_uv(
            signals,
            ["c3-m2", "c4-m1", "e1-m2", "e2-m1", "chin1-chin2", "ecg"],
        )

    n_start = int(start_epoch_number * 30 * fs)
    prefix = pd.DataFrame(np.nan, index=np.arange(n_start), columns=labels.columns)
    labels_1s = labels.loc[labels.index.repeat(int(30 * fs))].reset_index(drop=True)
    labels_1s = pd.concat([prefix, labels_1s], ignore_index=True)

    if len(labels_1s) < len(signals):
        suffix = pd.DataFrame(np.nan, index=np.arange(len(signals) - len(labels_1s)), columns=labels.columns)
        labels_1s = pd.concat([labels_1s, suffix], ignore_index=True)
    labels_1s = labels_1s.iloc[: len(signals)].reset_index(drop=True)

    rename_map = {
        "P1 sleep_stages": "stage_0",
        "P2 sleep_stages": "stage_1",
        "L1 sleep_stages": "stage_2",
        "L2 sleep_stages": "stage_3",
        "S1 sleep_stages": "stage_4",
        "S2 sleep_stages": "stage_5",
    }
    labels_1s = labels_1s.rename(columns=rename_map)

    signals, labels_1s = _resample_signals_and_annotations(signals, labels_1s, fs, FS_TARGET)

    annotations = labels_1s[[c for c in labels_1s.columns if c.startswith("stage_")]].copy()
    if include_majority_stage:
        annotations["stage"] = _majority_vote_stage(annotations)

    notes = (
        "Penn distilled from dataset_penn.ipynb and Penn_human_IRA.py. "
        "Original notebook wrote a legacy Xy matrix; this reference writes grouped /signals and /annotations."
    )

    return PreparedStudy(
        signals=signals,
        annotations=annotations,
        sampling_rate=FS_TARGET,
        unit_voltage="uV" if to_uv else "V",
        notes=notes,
    )


def source_summary() -> dict[str, list[str]]:
    """Quick source map for collaborators."""
    return {
        "prepared_h5_reader_writer": [
            "/home/wolfgang/repos/sleep_cognition/load_data.py",
            "/home/wolfgang/repos/CAISR_internal/prepared_data/prepare_data.py",
        ],
        "shhs": [
            "/home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py",
            "/home/wolfgang/repos/sleep_cognition/step2_prepared_data_shhs.py",
            "/home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_shhs.ipynb",
        ],
        "mesa": [
            "/home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py",
            "/home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_mesa.ipynb",
        ],
        "mros": [
            "/home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py",
            "/home/wolfgang/repos/sleep_cognition/mros_sof_annotations_parser.py",
            "/home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_mros.ipynb",
        ],
        "sof": [
            "/home/wolfgang/repos/sleep_cognition/mros_sof_annotations_parser.py",
            "/home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py",
            "No dedicated standalone SOF raw->H5 converter found; SOF handling is inferred from the shared MrOS/SOF parser.",
        ],
        "penn": [
            "/home/wolfgang/repos/CAISR_internal/Sleep Staging/CAISR_Codes/prepare_datasets/dataset_penn.ipynb",
            "/home/wolfgang/repos/caisr_dev/code_from_thijs/Penn_human_IRA.py",
        ],
        "koges": [
            "/home/wolfgang/repos/sleep_cognition/preprocessing_and_spectrograms.py",
            "/home/wolfgang/repos/sleep_cognition/step2_prepared_data_koges.py",
        ],
    }

