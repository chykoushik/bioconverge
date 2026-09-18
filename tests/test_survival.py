import numpy as np
import pandas as pd
import pytest

from bioconverge.utils import load_survival
from bioconverge.validation import InputValidationError
from bioconverge.layer4 import ValidationEngine, _adjust_pvalues


@pytest.fixture
def clinical():
    return pd.DataFrame({"patient_id": [f"P{i}" for i in range(12)],
                         "OS_days": np.arange(1, 13, dtype=float), "OS_event": [0, 1] * 6})


def test_followup_and_death_selection():
    frame = pd.DataFrame({"cases.submitter_id": ["P1", "P2"], "demographic.vital_status": ["Dead", "Alive"],
                          "demographic.days_to_death": [3, np.nan], "diagnoses.days_to_last_follow_up": [1, 9]})
    result = load_survival(frame)
    assert result.OS_days.tolist() == [3, 9]
    assert result.OS_event.tolist() == [1, 0]


def test_explicit_text_events_and_months():
    frame = pd.DataFrame({"id": ["abcdefghijklA", "abcdefghijklB"], "months": [1, 2], "status": ["Dead", "Alive"]})
    result = load_survival(frame, "months", "status", "id")
    assert result.patient_id.tolist() == frame.id.tolist()
    assert result.OS_days.tolist() == pytest.approx([30.44, 60.88])
    assert result.OS_event.tolist() == [1, 0]


def test_cbioportal_columns():
    frame = pd.DataFrame({"Patient ID": ["a", "b"], "Overall Survival (Months)": [2, 3],
                          "Overall Survival Status": ["1:DECEASED", "0:LIVING"]})
    assert load_survival(frame).OS_event.tolist() == [1, 0]


@pytest.mark.parametrize("event", [2, -1, "not deceased", "uncertain", np.inf])
def test_bad_events_rejected(clinical, event):
    clinical["OS_event"] = clinical.OS_event.astype(object)
    clinical.loc[0, "OS_event"] = event
    with pytest.raises(InputValidationError): load_survival(clinical)


@pytest.mark.parametrize("duration", [-1, np.inf, -np.inf, "bad"])
def test_bad_durations_rejected(clinical, duration):
    clinical["OS_days"] = clinical.OS_days.astype(object)
    clinical.loc[0, "OS_days"] = duration
    with pytest.raises(InputValidationError): load_survival(clinical)


def test_missing_outcomes_reported(clinical):
    clinical.loc[0, "OS_event"] = np.nan
    result = load_survival(clinical)
    assert len(result) == 11 and result.attrs["excluded_missing_outcomes"] == 1


def test_identical_records_removed_and_conflicts_rejected(clinical):
    result = load_survival(pd.concat([clinical, clinical]))
    assert len(result) == 12 and result.attrs["removed_identical_records"] == 12
    other = clinical.iloc[[0]].copy()
    other["OS_days"] = 999
    with pytest.raises(InputValidationError): load_survival(pd.concat([clinical, other]))


def test_explicit_column_typo_rejected(clinical):
    with pytest.raises(InputValidationError): load_survival(clinical, time_col="typo")


def test_mapping(clinical):
    clinical["OS_event"] = ["c", "e"] * 6
    result = load_survival(clinical, event_mapping={"c": 0, "e": 1})
    assert result.OS_event.tolist() == [0, 1] * 6


def test_survival_structured_status_and_no_fake_mutations(clinical):
    engine = ValidationEngine(clinical_tar_path=clinical)
    archetypes = pd.DataFrame({"patient_id": clinical.patient_id, "archetype": [0] * 6 + [1] * 6})
    engine._run_survival(archetypes)
    result = engine.survival_results()
    assert result["state"] == "completed" and result["n_matched"] == 12
    assert result["results"].n_tp53_mutant_archetype.isna().all()
    assert result["results"].n_tp53_unknown_archetype.eq(6).all()
    assert "logrank_adjusted_pvalue" in result["results"]


def test_no_events_unavailable(clinical):
    clinical["OS_event"] = 0
    engine = ValidationEngine(clinical_tar_path=clinical)
    engine._run_survival(pd.DataFrame({"patient_id": clinical.patient_id, "archetype": [0] * 6 + [1] * 6}))
    assert engine.survival_results()["state"] == "unavailable"


def test_adjustment_known_values():
    assert _adjust_pvalues([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.04, 0.04])
