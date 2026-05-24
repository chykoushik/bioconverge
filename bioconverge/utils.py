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


def load_clinical_tcga(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    print("loading clinical")
    if path.endswith(".tar.gz") or path.endswith(".tar"):
        with tarfile.open(path) as t:
            names = t.getnames()
            match = next((n for n in names if n.endswith("clinical.tsv")), None)
            if match is None:
                raise ValueError("clinical.tsv not in tar")
            df = pd.read_csv(t.extractfile(match), sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    pid_col = next((c for c in ["cases.submitter_id", "Patient ID", "patient_id"] if c in df.columns), df.columns[0])
    df = df.drop_duplicates(subset=pid_col)
    df["patient_id"] = df[pid_col].astype(str).str[:12]
    vital_col = next((c for c in df.columns if "vital" in c.lower()), None)
    death_col = next((c for c in df.columns if "days_to_death" in c.lower()), None)
    fu_col = next((c for c in df.columns if "last_follow" in c.lower()), None)
    os_months_col = next((c for c in df.columns if "os_months" in c.lower()), None)
    os_status_col = next((c for c in df.columns if "os_status" in c.lower()), None)
    if vital_col and death_col and fu_col:
        dead_mask = df[vital_col].astype(str).str.lower().str.strip() == "dead"
        df["OS_event"] = dead_mask.astype(int)
        days_death = pd.to_numeric(df[death_col], errors="coerce")
        days_fu = pd.to_numeric(df[fu_col], errors="coerce")
        df["OS_days"] = np.where(dead_mask, days_death, days_fu)
    elif os_months_col and os_status_col:
        df["OS_event"] = df[os_status_col].astype(str).str.contains("1|deceased|dead", case=False).astype(int)
        df["OS_days"] = pd.to_numeric(df[os_months_col], errors="coerce") * 30.44
    else:
        raise ValueError("no survival columns found")
    return df[["patient_id", "OS_days", "OS_event"]].dropna(subset=["OS_days"]).reset_index(drop=True)


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
