import os
import re
import tarfile
import time

import numpy as np
import pandas as pd
import requests

from .validation import InputValidationError, validate_ids

HEADERS = {"User-Agent": "bioconverge/0.2"}


class ApiError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _request(method, url, params=None, timeout=30, retries=3, files=None, budget=100):
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        raise ValueError("retries must be a positive integer")
    if not np.isfinite(timeout) or timeout <= 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("timeout and budget must be positive finite numbers")
    deadline = time.monotonic() + budget
    error = None
    for attempt in range(retries):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        delay = min(2 ** attempt, 10)
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=HEADERS, timeout=min(timeout, remaining))
            else:
                response = requests.post(url, files=files, headers=HEADERS, timeout=min(timeout, remaining))
            if response.status_code == 429 or response.status_code >= 500:
                error = ApiError(f"HTTP {response.status_code}: {url}", response.status_code)
                try:
                    delay = min(max(float(response.headers.get("Retry-After", delay)), 0), 30)
                except (TypeError, ValueError):
                    pass
            else:
                response.raise_for_status()
                return response
        except requests.HTTPError as e:
            raise ApiError(f"HTTP error: {url}: {e}", getattr(e.response, "status_code", None)) from e
        except requests.RequestException as e:
            error = ApiError(f"request failed: {url}: {e}")
        if attempt + 1 < retries:
            time.sleep(max(0, min(delay, deadline - time.monotonic())))
    raise error or ApiError(f"request budget exhausted: {url}")


def _retry_get(url, params=None, timeout=30, retries=3):
    return _request("GET", url, params=params, timeout=timeout, retries=retries)


def api_get(url, params=None, timeout=30, retries=3):
    return _retry_get(url, params=params, timeout=timeout, retries=retries)


def api_post(url, files, timeout=30, retries=3):
    return _request("POST", url, files=files, timeout=timeout, retries=retries)


def _read_table(path, member_suffix=None):
    if isinstance(path, pd.DataFrame):
        return path.copy(deep=True)
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"not found: {path}")
    if path.endswith((".tar.gz", ".tar")):
        with tarfile.open(path) as archive:
            matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(member_suffix or "clinical.tsv")]
            if len(matches) != 1:
                raise InputValidationError("archive must contain exactly one matching table")
            return pd.read_csv(archive.extractfile(matches[0]), sep="\t", low_memory=False)
    return pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".txt")) else ",", low_memory=False)


def _column(df, requested, aliases, required=True):
    if requested is not None:
        if requested not in df:
            raise InputValidationError(f"missing column: {requested}")
        return requested
    names = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in names:
            return names[key]
    if required:
        raise InputValidationError(f"no recognized column for {aliases[0]}")
    return None


def load_survival(path, time_col=None, event_col=None, patient_col=None,
                  time_unit=None, event_mapping=None):
    df = _read_table(path)
    if df.empty or not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise InputValidationError("clinical data must be nonempty with unique columns")
    pid = _column(df, patient_col if patient_col not in (None, "patient_id") or "patient_id" in df else None,
                  ["patient_id", "Patient ID", "cases.submitter_id", "PATIENT_ID", "PatientID"])
    event = _column(df, event_col, ["OS_event", "Overall Survival Status", "OS_status", "vital_status",
                                  "demographic.vital_status", "event", "deceased"])
    raw = df[event].mask(df[event].isin(["'--", "--", "[Not Available]"]))
    if event_mapping is not None:
        values = raw.map(event_mapping)
        if (raw.notna() & values.isna()).any():
            raise InputValidationError("event_mapping does not cover observed values")
    else:
        mapping = {"0": 0, "0.0": 0, "false": 0, "alive": 0, "living": 0,
                   "0:living": 0, "0:alive": 0, "no": 0, "censored": 0,
                   "1": 1, "1.0": 1, "true": 1, "dead": 1, "deceased": 1,
                   "1:deceased": 1, "1:dead": 1, "yes": 1}
        values = raw.astype(str).str.strip().str.lower().map(mapping)
        if (raw.notna() & values.isna()).any():
            raise InputValidationError("unrecognized survival event values; provide event_mapping")
    if not values.dropna().isin([0, 1]).all():
        raise InputValidationError("survival events must be binary")
    duration = _column(df, time_col, ["OS_days", "survival_time", "overall_survival",
                                      "Overall Survival (Months)", "OS_months"], required=False)
    if duration is not None:
        raw_time = df[duration]
        inferred = "months" if "month" in duration.lower() else "days"
    else:
        death = _column(df, None, ["demographic.days_to_death", "days_to_death"], required=False)
        follow = _column(df, None, ["diagnoses.days_to_last_follow_up", "days_to_last_follow_up",
                                     "days_to_last_followup", "days_to_last_follow"], required=False)
        if death is None and follow is None:
            raise InputValidationError("no recognized survival time column")
        raw_time = pd.Series(np.nan, index=df.index, dtype=object)
        if death:
            raw_time.loc[values == 1] = df.loc[values == 1, death]
        if follow:
            raw_time.loc[values == 0] = df.loc[values == 0, follow]
        inferred = "days"
    clean_time = raw_time.mask(raw_time.isin(["'--", "--", "[Not Available]"]))
    times = pd.to_numeric(clean_time, errors="coerce")
    if (clean_time.notna() & times.isna()).any():
        raise InputValidationError("survival durations must be numeric")
    unit = time_unit or inferred
    if unit not in {"days", "months", "years"}:
        raise InputValidationError("time_unit must be days, months, or years")
    times = times * {"days": 1, "months": 30.44, "years": 365.25}[unit]
    if np.isinf(times).any() or (times.dropna() < 0).any():
        raise InputValidationError("survival durations must be finite and nonnegative")
    if df[pid].isna().any() or df[pid].astype(str).str.strip().eq("").any():
        raise InputValidationError("clinical patient IDs must be nonmissing")
    result = pd.DataFrame({"patient_id": df[pid], "OS_days": times, "OS_event": values})
    missing = int(result[["OS_days", "OS_event"]].isna().any(axis=1).sum())
    result = result.dropna(subset=["OS_days", "OS_event"])
    repeats = int(result.duplicated().sum())
    result = result.drop_duplicates().reset_index(drop=True)
    validate_ids(result.patient_id, "clinical patient IDs (conflicting repeated records are unsupported)")
    result["OS_event"] = result.OS_event.astype(int)
    result.attrs.update({"excluded_missing_outcomes": missing, "removed_identical_records": repeats,
                         "time_unit": "days", "source_time_unit": unit})
    return result


def load_clinical_tcga(path, time_col=None, event_col=None, patient_col=None, **kwargs):
    return load_survival(path, time_col, event_col, patient_col, **kwargs)


def load_biospecimen(path):
    return _read_table(path, "sample.tsv")


def load_metabric(path):
    return _read_table(path)


def load_tcga_brca(path):
    return _read_table(path)


def load_maf_tp53(tar_path):
    records = []
    assessed = set()
    needed = ["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"]
    count = 0
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith((".maf.txt", ".maf")):
                continue
            count += 1
            chunk = pd.read_csv(archive.extractfile(member), sep="\t", comment="#", low_memory=False)
            if not set(needed).issubset(chunk.columns) or chunk.Tumor_Sample_Barcode.isna().any():
                raise InputValidationError(f"invalid mutation table: {member.name}")
            barcodes = chunk.Tumor_Sample_Barcode.astype(str)
            if not barcodes.str.match(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-").all():
                raise InputValidationError("TCGA mutation loader requires TCGA sample barcodes")
            assessed.update(barcodes.str[:12])
            selected = chunk.loc[(chunk.Hugo_Symbol == "TP53") & chunk.Variant_Classification.notna() &
                                 (chunk.Variant_Classification != "Silent"), needed].copy()
            selected["patient_id"] = selected.Tumor_Sample_Barcode.str[:12]
            records.append(selected)
    if not count:
        raise InputValidationError("archive contains no mutation tables")
    result = pd.concat(records, ignore_index=True).drop_duplicates("patient_id")
    result.attrs["assessed_patients"] = sorted(assessed)
    return result.reset_index(drop=True)


def parse_gmt(path):
    sets = {}
    with open(path, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3 or not parts[0] or not all(parts[2:]) or parts[0] in sets:
                raise InputValidationError(f"invalid GMT record: {number}")
            sets[parts[0]] = list(dict.fromkeys(parts[2:]))
    if not sets:
        raise InputValidationError("GMT contains no gene sets")
    return sets


def normalize_scores(df, score_cols):
    out = df.copy()
    for col in score_cols:
        vals = pd.to_numeric(out[col], errors="raise").astype(float)
        if np.isinf(vals).any():
            raise InputValidationError("scores cannot contain infinity")
        mn, mx = vals.min(), vals.max()
        out[col] = (vals - mn) / (mx - mn) if mx > mn else vals.where(vals.isna(), 0.0)
    return out
