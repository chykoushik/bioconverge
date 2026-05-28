import os
import tarfile
import time

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "bioconverge/0.1 (research; python-requests)"}


def _retry_get(url, params=None, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 403:
                raise RuntimeError(f"403 forbidden {url}")
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except RuntimeError:
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"api failed {url}: {e}")
            time.sleep(2 ** attempt)


def load_survival(path, time_col=None, event_col=None, patient_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    print("loading survival")
    if path.endswith(".tar.gz") or path.endswith(".tar"):
        with tarfile.open(path) as t:
            names = t.getnames()
            match = next((n for n in names if n.endswith("clinical.tsv")), None)
            if match is None:
                raise ValueError("clinical.tsv not in tar")
            df = pd.read_csv(t.extractfile(match), sep="\t", low_memory=False)
    elif path.endswith(".tsv") or path.endswith(".txt"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    if patient_col and patient_col in df.columns:
        df["patient_id"] = df[patient_col].astype(str).str[:12]
    else:
        pid_col = next((c for c in ["patient_id", "Patient ID", "cases.submitter_id",
                                     "PATIENT_ID", "PatientID"] if c in df.columns), df.columns[0])
        df["patient_id"] = df[pid_col].astype(str).str[:12]
    if time_col and time_col in df.columns:
        df["OS_days"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        time_candidates = [c for c in df.columns if any(x in c.lower() for x in
                          ["days_to_death", "os_days", "survival_time", "overall_survival",
                           "os_months", "days_to_last_follow"])]
        if not time_candidates:
            raise ValueError(f"no time column found. available: {list(df.columns)}")
        best_time = time_candidates[0]
        df["OS_days"] = pd.to_numeric(df[best_time], errors="coerce")
        if "month" in best_time.lower():
            df["OS_days"] = df["OS_days"] * 30.44
        print(f"using time column: {best_time}")
    if event_col and event_col in df.columns:
        df["OS_event"] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    else:
        event_candidates = [c for c in df.columns if any(x in c.lower() for x in
                           ["os_status", "vital_status", "os_event", "event",
                            "overall_survival_status", "deceased"])]
        if not event_candidates:
            raise ValueError(f"no event column found. available: {list(df.columns)}")
        best_event = event_candidates[0]
        col_vals = df[best_event].astype(str).str.lower().str.strip()
        if col_vals.str.match(r"^[01]$").all():
            df["OS_event"] = pd.to_numeric(df[best_event], errors="coerce").fillna(0).astype(int)
        else:
            df["OS_event"] = col_vals.str.contains("dead|deceased|1|yes", case=False).astype(int)
        print(f"using event column: {best_event}")
    return df[["patient_id", "OS_days", "OS_event"]].dropna(subset=["OS_days"]).reset_index(drop=True)


def load_clinical_tcga(path, time_col=None, event_col=None, patient_col=None):
    return load_survival(path, time_col=time_col, event_col=event_col, patient_col=patient_col)


def load_biospecimen(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    print("loading biospecimen")
    if path.endswith(".tar.gz") or path.endswith(".tar"):
        with tarfile.open(path) as t:
            names = t.getnames()
            match = next((n for n in names if n.endswith("sample.tsv")), None)
            if match is None:
                raise ValueError("sample.tsv not in tar")
            return pd.read_csv(t.extractfile(match), sep="\t", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_metabric(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    print("loading metabric")
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_tcga_brca(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    print("loading tcga brca")
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_maf_tp53(tar_path):
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"not found: {tar_path}")
    print("loading mutations")
    records = []
    needed = ["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"]
    with tarfile.open(tar_path) as t:
        for member in t.getmembers():
            name = member.name
            if not (name.endswith(".maf.txt") or name.endswith(".maf")):
                continue
            f = t.extractfile(member)
            if f is None:
                continue
            try:
                chunk = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
                if not all(c in chunk.columns for c in needed):
                    continue
                tp53 = chunk[chunk["Hugo_Symbol"] == "TP53"]
                tp53 = tp53[tp53["Variant_Classification"] != "Silent"]
                if len(tp53) > 0:
                    records.append(tp53[needed].copy())
            except Exception:
                continue
    if not records:
        return pd.DataFrame(columns=["patient_id", "Hugo_Symbol", "Variant_Classification"])
    result = pd.concat(records, ignore_index=True)
    result["patient_id"] = result["Tumor_Sample_Barcode"].str[:12]
    return result.drop_duplicates("patient_id").reset_index(drop=True)


def parse_gmt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    gene_sets = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = [g for g in parts[2:] if g]
    return gene_sets


def api_get(url, params=None, timeout=30, retries=3):
    return _retry_get(url, params=params, timeout=timeout, retries=retries)


def normalize_scores(df, score_cols):
    out = df.copy()
    for col in score_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        mn, mx = vals.min(), vals.max()
        out[col] = (vals - mn) / (mx - mn) if mx > mn else 0.0
    return out