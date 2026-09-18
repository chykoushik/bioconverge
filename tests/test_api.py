import requests
import pytest
import pandas as pd

from bioconverge import layer3, utils
from bioconverge.layer3 import HypothesisGenerator


class Response:
    def __init__(self, data=None, code=200, headers=None):
        self.data, self.status_code, self.headers = data, code, headers or {}
    def json(self):
        if isinstance(self.data, Exception): raise self.data
        return self.data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


@pytest.mark.parametrize("code", [400, 403, 404])
def test_http_errors_raise(monkeypatch, code):
    monkeypatch.setattr(utils.requests, "get", lambda *a, **k: Response(code=code))
    with pytest.raises(utils.ApiError) as error: utils.api_get("https://example.test")
    assert error.value.status_code == code


@pytest.mark.parametrize("code", [429, 500, 503])
def test_retry_exhaustion_raises(monkeypatch, code):
    calls = []
    monkeypatch.setattr(utils.requests, "get", lambda *a, **k: calls.append(k) or Response(code=code))
    monkeypatch.setattr(utils.time, "sleep", lambda delay: None)
    with pytest.raises(utils.ApiError): utils.api_get("https://example.test", timeout=2, retries=3)
    assert len(calls) == 3 and all(c["timeout"] <= 2 for c in calls)


def test_timeout_and_post_have_bounds(monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("expired")
    monkeypatch.setattr(utils.requests, "post", timeout)
    monkeypatch.setattr(utils.time, "sleep", lambda delay: None)
    with pytest.raises(utils.ApiError): utils.api_post("https://example.test", files={}, timeout=3, retries=2)
    assert len(calls) == 2 and all(c["timeout"] <= 3 for c in calls)


def test_rate_limit_then_success(monkeypatch):
    responses = iter([Response(code=429, headers={"Retry-After": "0"}), Response({"ok": True})])
    monkeypatch.setattr(utils.requests, "get", lambda *a, **k: next(responses))
    assert utils.api_get("https://example.test").json() == {"ok": True}


@pytest.mark.parametrize("query,args", [(layer3._query_reactome, ["immune"]), (layer3._query_string, [["TP53", "BRCA1"]]),
    (layer3._query_gwas, ["immune"]), (layer3._query_pubmed, ["immune cancer"])])
def test_api_failure_not_empty_success(monkeypatch, query, args):
    def fail(*a, **k): raise requests.Timeout("expired")
    monkeypatch.setattr(layer3, "api_get", fail)
    result = query(*args)
    assert result.state == "failed" and result.reason and not result


def test_pubmed_failure_not_zero(monkeypatch):
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: Response({}))
    assert layer3._count_pubmed("query") is None
    assert layer3._pubmed_flag(None) == "unavailable"
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: Response({"esearchresult": {"count": "0"}}))
    assert layer3._count_pubmed("query") == 0


def test_reactome_nested_schema(monkeypatch):
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: Response({"results": [{"entries": [{"name": "Cell cycle", "stId": "R-HSA-1"}]}]}))
    result = layer3._query_reactome("cell cycle")
    assert result.state == "completed" and result[0]["stId"] == "R-HSA-1"


def test_enrichr_uses_post_and_adjusted_significance(monkeypatch):
    calls = []
    monkeypatch.setattr(layer3, "api_post", lambda *a, **k: calls.append(k) or Response({"userListId": 12}))
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: Response({"MSigDB_Hallmark_2020": [
        [1, "significant", 0.001, 1, 1, ["TP53"], 0.01], [2, "not significant", 0.01, 1, 1, ["TP53"], 0.8]]}))
    result = layer3._query_enrichr(["TP53", "BRCA1"])
    assert calls[0]["files"]["list"][1] == "TP53\nBRCA1"
    assert len(result) == 1 and result[0]["term"] == "significant"


def test_enrichr_missing_genes_never_substitutes_score():
    assert layer3._query_enrichr([]).state == "unavailable"


def test_string_single_mapping_not_interaction():
    assert not layer3._query_string(["TP53"])
    assert layer3._query_string(["TP53"]).state == "unavailable"


def test_string_no_gene_truncation(monkeypatch):
    calls = []
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: calls.append(k) or Response([]))
    genes = [f"LONG_GENE_SYMBOL_{i}" for i in range(25)]
    assert layer3._query_string(genes).state == "completed"
    assert calls[0]["params"]["identifiers"].split("\r") == genes


def test_gwas_search_method(monkeypatch):
    calls = []
    monkeypatch.setattr(layer3, "api_get", lambda url, **k: calls.append((url, k)) or Response({"_embedded": {"efoTraits": []}}))
    assert layer3._query_gwas("immune").state == "completed"
    assert calls[0][0].endswith("findByEfoTrait") and calls[0][1]["params"]["trait"] == "immune"


@pytest.mark.parametrize("data", [{}, [], {"results": [{}]}, ValueError("bad JSON")])
def test_malformed_reactome(monkeypatch, data):
    monkeypatch.setattr(layer3, "api_get", lambda *a, **k: Response(data))
    assert layer3._query_reactome("query").state == "failed"


def test_replay_and_query_deduplication(monkeypatch):
    calls = []
    def respond(url, **kwargs):
        calls.append(url)
        if "reactome" in url: return Response({"results": []})
        if "gwas" in url: return Response({"_embedded": {"efoTraits": []}})
        return Response({"esearchresult": {"count": "7"}})
    monkeypatch.setattr(layer3, "api_get", respond)
    archetypes = pd.DataFrame({"patient_id": list("abcdef"), "archetype": [0, 1, 2] * 2})
    cache = {}
    first = HypothesisGenerator({"a": {"process": "immune"}}, archetypes, disease_context="glioblastoma", api_cache=cache).generate()
    assert len(calls) == 3
    second = HypothesisGenerator({"a": {"process": "immune"}}, archetypes, disease_context="glioblastoma", api_cache=cache, replay_only=True).generate()
    assert len(calls) == 3
    pd.testing.assert_frame_equal(first.hypotheses(), second.hypotheses())
    assert first.hypotheses().pubmed_query.str.contains("glioblastoma").all()
    assert not first.hypotheses().empirically_validated.any()
