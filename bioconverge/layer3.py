import copy
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .utils import api_get, api_post
from .validation import check_metadata, validate_ids, status, InputValidationError

REACTOME_URL = "https://reactome.org/ContentService/search/query"
ENRICHR_ADD_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
STRING_URL = "https://version-12-0.string-db.org/api/json/network"
GWAS_URL = "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/search/findByEfoTrait"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


class Evidence(list):
    def __init__(self, records=(), state="completed", reason=None, responses=None):
        super().__init__(records)
        self.state = state
        self.reason = reason
        self.responses = responses or []

    def to_dict(self):
        return {"records": list(self), "state": self.state, "reason": self.reason,
                "responses": self.responses}


def _json_response(response, expected):
    data = response.json()
    if not isinstance(data, expected):
        raise ValueError("unexpected API response type")
    return data


def _failed(error):
    return Evidence(state="failed", reason=f"{type(error).__name__}: {error}")


def _query_reactome(process_term):
    try:
        data = _json_response(api_get(REACTOME_URL, params={"query": process_term, "types": "Pathway",
                                                           "species": "Homo sapiens", "rows": 5}), dict)
        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError("missing Reactome results")
        rows = []
        for group in data["results"]:
            if not isinstance(group, dict) or not isinstance(group.get("entries"), list):
                raise ValueError("invalid Reactome entries")
            for item in group["entries"]:
                if not item.get("stId") or not item.get("name"):
                    raise ValueError("invalid Reactome pathway")
                rows.append({"name": item["name"], "stId": item["stId"],
                             "url": f"https://reactome.org/content/detail/{item['stId']}"})
        return Evidence(rows[:5], responses=[data])
    except Exception as e:
        return _failed(e)


def _query_enrichr(gene_list, description="query"):
    if not gene_list:
        return Evidence(state="unavailable", reason="no genes supplied")
    try:
        genes = list(dict.fromkeys(gene_list))
        data = _json_response(api_post(ENRICHR_ADD_URL, files={"list": (None, "\n".join(genes)),
                                                               "description": (None, description)}), dict)
        if "userListId" not in data or data.get("expired"):
            raise ValueError("Enrichr did not return a usable list ID")
        enriched = _json_response(api_get(ENRICHR_ENRICH_URL, params={"userListId": data["userListId"],
                                                                     "backgroundType": "MSigDB_Hallmark_2020"}), dict)
        if not isinstance(enriched.get("MSigDB_Hallmark_2020"), list):
            raise ValueError("missing Enrichr enrichment results")
        rows = []
        for entry in enriched["MSigDB_Hallmark_2020"]:
            if not isinstance(entry, list) or len(entry) < 7:
                raise ValueError("invalid Enrichr entry")
            pvalue, adjusted = float(entry[2]), float(entry[6])
            if not 0 <= pvalue <= 1 or not 0 <= adjusted <= 1:
                raise ValueError("invalid Enrichr p-value")
            if adjusted < 0.05:
                rows.append({"term": entry[1], "pvalue": pvalue, "adjusted_pvalue": adjusted,
                             "overlap": entry[5]})
        return Evidence(rows[:5], responses=[data, enriched])
    except Exception as e:
        return _failed(e)


def _query_string(gene_list):
    genes = list(dict.fromkeys(gene_list))
    if len(genes) < 2:
        return Evidence(state="unavailable", reason="at least two genes required for interaction evidence")
    try:
        data = _json_response(api_get(STRING_URL, params={"identifiers": "\r".join(genes), "species": 9606,
                                                         "required_score": 400, "caller_identity": "bioconverge"}), list)
        rows = []
        for edge in data:
            if not isinstance(edge, dict) or not edge.get("preferredName_A") or not edge.get("preferredName_B"):
                raise ValueError("invalid STRING interaction")
            score = float(edge.get("score", np.nan))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid STRING score")
            if score >= 0.4:
                rows.append({"protein_a": edge["preferredName_A"], "protein_b": edge["preferredName_B"], "score": score})
        return Evidence(rows[:5], responses=[data])
    except Exception as e:
        return _failed(e)


def _query_gwas(process_term):
    try:
        data = _json_response(api_get(GWAS_URL, params={"trait": process_term, "page": 0, "size": 5}), dict)
        traits = data.get("_embedded", {}).get("efoTraits")
        if not isinstance(traits, list):
            raise ValueError("missing GWAS trait results")
        rows = []
        for item in traits:
            if not isinstance(item, dict) or not item.get("trait"):
                raise ValueError("invalid GWAS trait")
            rows.append({"trait": item["trait"], "uri": item.get("uri", "")})
        return Evidence(rows, responses=[data])
    except Exception as e:
        return _failed(e)


def _query_pubmed(query_term):
    try:
        data = _json_response(api_get(PUBMED_SEARCH_URL, params={"db": "pubmed", "term": query_term,
                                                                "retmax": 0, "retmode": "json"}), dict)
        count = int(data["esearchresult"]["count"])
        if count < 0:
            raise ValueError("invalid PubMed count")
        return Evidence([{"count": count}], responses=[data])
    except Exception as e:
        return _failed(e)


def _count_pubmed(query_term):
    result = _query_pubmed(query_term)
    return result[0]["count"] if result.state == "completed" else None


class HypothesisGenerator:
    def __init__(self, score_metadata, archetypes_df, fragility_df=None, disease_context=None,
                 api_cache=None, replay_only=False):
        if score_metadata is None:
            raise InputValidationError("score_metadata is required")
        check_metadata(score_metadata)
        if not isinstance(archetypes_df, pd.DataFrame) or not {"patient_id", "archetype"}.issubset(archetypes_df):
            raise InputValidationError("archetypes require patient_id and archetype")
        validate_ids(archetypes_df.patient_id)
        if archetypes_df.archetype.isna().any():
            raise InputValidationError("archetype labels cannot be missing")
        if disease_context is not None and (not isinstance(disease_context, str) or not disease_context.strip()):
            raise InputValidationError("disease_context must be a nonempty string")
        if api_cache is not None and not isinstance(api_cache, dict):
            raise InputValidationError("api_cache must be a dictionary")
        self.score_metadata = copy.deepcopy(score_metadata)
        self.archetypes_df = archetypes_df.copy()
        self.fragility_df = fragility_df
        self.disease_context = disease_context
        self.api_cache = api_cache if api_cache is not None else {}
        self.replay_only = replay_only
        self._hypotheses_df = None
        self._repro_log = []
        self.statuses = {}

    def _fetch(self, source, arguments, function):
        key = json.dumps(["0.2", source, arguments], sort_keys=True)
        cached = key in self.api_cache
        if cached:
            item = copy.deepcopy(self.api_cache[key])
            result = Evidence(**item)
            if result.state not in {"completed", "failed", "skipped", "unavailable"}:
                raise InputValidationError("invalid cached evidence status")
        elif self.replay_only:
            result = Evidence(state="unavailable", reason="response absent from replay cache")
        else:
            result = function()
            self.api_cache[key] = copy.deepcopy(result.to_dict())
        self._repro_log.append({"query": key, "timestamp": datetime.now(timezone.utc).isoformat(),
                               "cached": cached, **copy.deepcopy(result.to_dict())})
        return result

    def generate(self):
        self._repro_log = []
        self.statuses = {}
        rows = []
        for score_name, meta in self.score_metadata.items():
            process = meta.get("process", score_name)
            genes = meta.get("genes", [])
            reactome = self._fetch("reactome", [process], lambda: _query_reactome(process))
            enrichr = self._fetch("enrichr", [genes, process], lambda: _query_enrichr(genes, process))
            string = self._fetch("string", [genes], lambda: _query_string(genes))
            gwas = self._fetch("gwas", [process], lambda: _query_gwas(process))
            query = f"{process} {self.disease_context}" if self.disease_context else None
            pubmed = self._fetch("pubmed", [query], lambda: _query_pubmed(query) if query else
                                 Evidence(state="unavailable", reason="disease_context not supplied"))
            sources = {"reactome": reactome, "enrichr": enrichr, "string": string, "gwas": gwas, "pubmed": pubmed}
            for source_name, evidence in sources.items():
                self.statuses[f"{score_name}.{source_name}"] = status(evidence.state, evidence.reason, n_results=len(evidence))
            count = pubmed[0]["count"] if pubmed.state == "completed" and pubmed else None
            for arch in sorted(self.archetypes_df.archetype.unique()):
                rows.append({"archetype": arch, "score": score_name, "process": process,
                             "modality": meta.get("modality", "unknown"),
                             "hypothesis": f"Metadata annotation for {score_name}: {process}; archetype association not tested.",
                             "evidence_scope": "metadata_annotation", "empirically_validated": False,
                             "db_support": sum(bool(v) and v.state == "completed" for v in [reactome, enrichr, string, gwas]),
                             "reactome_pathway": reactome[0]["name"] if reactome else "",
                             "reactome_url": reactome[0]["url"] if reactome else "",
                             "enrichr_term": enrichr[0]["term"] if enrichr else "",
                             "pubmed_count": count, "pubmed_flag": _pubmed_flag(count), "pubmed_query": query,
                             "source_status": {k: status(v.state, v.reason, n_results=len(v)) for k, v in sources.items()},
                             "n_patients_archetype": int((self.archetypes_df.archetype == arch).sum())})
        self._hypotheses_df = pd.DataFrame(rows).sort_values(["db_support", "score", "archetype"],
                                                          ascending=[False, True, True], kind="stable").reset_index(drop=True)
        self.statuses["annotations"] = status("completed", "metadata annotations; archetype associations not tested")
        self.statuses["empirical_hypotheses"] = status("unavailable", "no data-dependent hypothesis test implemented")
        return self

    def hypotheses(self):
        return self._hypotheses_df

    def cross_support(self):
        if self._hypotheses_df is None or self._hypotheses_df.empty:
            return pd.DataFrame()
        return self._hypotheses_df.groupby("archetype")["db_support"].agg(
            mean_support="mean", max_support="max", n_hypotheses="count").reset_index()

    def reproducibility_log(self):
        return pd.DataFrame(copy.deepcopy(self._repro_log))


def _pubmed_flag(count):
    if count is None or pd.isna(count):
        return "unavailable"
    if count > 100:
        return "convergent_with_prior_knowledge"
    if count < 10:
        return "exploratory"
    return "moderate_support"
