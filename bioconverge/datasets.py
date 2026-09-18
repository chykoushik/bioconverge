import re
import tarfile

import numpy as np
import pandas as pd

from .validation import InputValidationError, validate_ids


def load_tcga_rnaseq(path, patient_ids, gene_sets, sample_type="01"):
    validate_ids(patient_ids)
    if not isinstance(gene_sets, dict) or not gene_sets or any(not isinstance(v, (list, tuple)) or not v for v in gene_sets.values()):
        raise InputValidationError("nonempty gene sets are required")
    with tarfile.open(path) as archive:
        matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".data.txt")]
        if len(matches) != 1:
            raise InputValidationError("RNA archive must contain exactly one expression table")
        raw = pd.read_csv(archive.extractfile(matches[0]), sep="\t", index_col=0, skiprows=[1])
    if not all(re.match(r"^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}", c) for c in raw.columns):
        raise InputValidationError("expression table requires TCGA sample barcodes")
    chosen = [c for c in raw if c[:12] in set(patient_ids) and c[13:15] == sample_type]
    if not chosen:
        raise InputValidationError("no RNA samples match the required patient IDs and sample type")
    raw = raw[chosen].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy()).all() or (raw.to_numpy() < 0).any():
        raise InputValidationError("RNA values must be finite and nonnegative")
    raw.index = raw.index.astype(str).str.split("|").str[0]
    duplicates = int(raw.index.duplicated().sum())
    raw = raw.loc[~raw.index.duplicated(keep="first")]
    raw.columns = [c[:12] for c in raw]
    raw = raw.T.groupby(level=0, sort=False).mean().T
    values = {}
    coverage = {}
    for name, genes in gene_sets.items():
        observed = [g for g in dict.fromkeys(genes) if g in raw.index]
        if not observed:
            raise InputValidationError(f"no measured genes for {name}")
        values[name] = raw.loc[observed].mean(axis=0)
        coverage[name] = {"requested": len(set(genes)), "observed": len(observed)}
    result = pd.DataFrame(values).reindex(list(patient_ids))
    result.index.name = "patient_id"
    result = result.reset_index()
    result.attrs = {"sample_type": sample_type, "n_samples": len(chosen), "n_measured_patients": raw.shape[1],
                    "duplicate_gene_symbols_removed": duplicates, "gene_coverage": coverage}
    return result


def load_mutation_counts(path, patient_ids, encoding="utf-8"):
    validate_ids(patient_ids)
    wanted = set(patient_ids)
    counts = pd.Series(np.nan, index=list(patient_ids), name="mutation_count", dtype=float)
    observed = set()
    allowed = {"Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
               "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site"}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith((".maf", ".maf.txt")):
                continue
            name = member.name.rsplit("/", 1)[-1]
            match = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", name)
            if match is None or match.group(1) not in wanted:
                continue
            pid = match.group(1)
            if pid in observed:
                raise InputValidationError(f"multiple mutation files for {pid}")
            frame = pd.read_csv(archive.extractfile(member), sep="\t", comment="#", low_memory=False, encoding=encoding)
            if not {"Tumor_Sample_Barcode", "Variant_Classification"}.issubset(frame):
                raise InputValidationError(f"missing mutation fields: {member.name}")
            if frame.Tumor_Sample_Barcode.isna().any() or not frame.Tumor_Sample_Barcode.astype(str).str[:12].eq(pid).all():
                raise InputValidationError(f"mutation patient mismatch: {member.name}")
            if frame.Variant_Classification.isna().any():
                raise InputValidationError(f"missing mutation classifications: {member.name}")
            counts.loc[pid] = int(frame.Variant_Classification.isin(allowed).sum())
            observed.add(pid)
    if not observed:
        raise InputValidationError("no valid mutation files match patients")
    counts.attrs = {"encoding": encoding, "measure": "nonsynonymous mutation count; not TMB", "n_assessed": len(observed),
                    "n_unassessed": len(wanted - observed)}
    return counts
