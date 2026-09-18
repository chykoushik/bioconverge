import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from bioconverge.datasets import load_tcga_rnaseq, load_mutation_counts
from bioconverge.utils import parse_gmt, load_maf_tp53, normalize_scores
from bioconverge.validation import InputValidationError
import pandas as pd


def archive(path, files):
    with tarfile.open(path, "w:gz") as target:
        for name, text in files.items():
            data = text.encode()
            item = tarfile.TarInfo(name)
            item.size = len(data)
            target.addfile(item, io.BytesIO(data))


def test_rna_excludes_normal_and_metastatic(tmp_path):
    path = tmp_path / "rna.tar.gz"
    archive(path, {"rna.data.txt": "gene\tTCGA-AA-0001-01A\tTCGA-AA-0001-11A\tTCGA-AA-0001-06A\nunits\tnormalized\tnormalized\tnormalized\nTP53|1\t2\t100\t200\n"})
    result = load_tcga_rnaseq(path, ["TCGA-AA-0001", "TCGA-AA-0002"], {"score": ["TP53"]})
    assert result.score.iloc[0] == 2 and np.isnan(result.score.iloc[1])
    assert result.attrs["n_measured_patients"] == 1


def test_mutation_absence_is_missing_not_zero(tmp_path):
    path = tmp_path / "maf.tar.gz"
    archive(path, {"TCGA-AA-0001.maf": "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\nTP53\tTCGA-AA-0001-01A\tSilent\nTP53\tTCGA-AA-0001-01A\tMissense_Mutation\n"})
    counts = load_mutation_counts(path, ["TCGA-AA-0001", "TCGA-AA-0002"])
    assert counts.iloc[0] == 1 and np.isnan(counts.iloc[1])
    assert counts.attrs["n_unassessed"] == 1
    tp53 = load_maf_tp53(path)
    assert tp53.attrs["assessed_patients"] == ["TCGA-AA-0001"]


def test_malformed_mutation_is_error(tmp_path):
    path = tmp_path / "maf.tar.gz"
    archive(path, {"TCGA-AA-0001.maf": "wrong\ncolumn\n"})
    with pytest.raises(InputValidationError): load_maf_tp53(path)
    with pytest.raises(InputValidationError): load_mutation_counts(path, ["TCGA-AA-0001"])


@pytest.mark.parametrize("text", ["", "path\tdescription\n", "path\tdesc\tA\npath\tdesc\tB\n"])
def test_bad_gmt_rejected(tmp_path, text):
    path = tmp_path / "test.gmt"
    path.write_text(text)
    with pytest.raises(InputValidationError): parse_gmt(path)


def test_normalization_preserves_missingness():
    result = normalize_scores(pd.DataFrame({"a": [np.nan, np.nan], "b": [2, np.nan]}), ["a", "b"])
    assert result.a.isna().all()
    assert result.b.iloc[0] == 0 and np.isnan(result.b.iloc[1])
