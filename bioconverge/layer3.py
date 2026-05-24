import time
from datetime import datetime

import numpy as np
import pandas as pd

from .utils import api_get

REACTOME_URL       = "https://reactome.org/ContentService/search/query"
ENRICHR_ADD_URL    = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
STRING_URL         = "https://string-db.org/api/json/network"
STRING_IDS_URL     = "https://string-db.org/api/json/get_string_ids"
GWAS_URL           = "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/search"
PUBMED_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _query_reactome(process_term):
    try:
        r = api_get(REACTOME_URL, params={
            "query": process_term,
            "types": "Pathway",
            "species": "Homo sapiens",
            "rows": 5,
        })
        results = r.json().get("results", [])
        return [
            {
                "name": item.get("name", ""),
                "stId": item.get("stId", ""),
                "url": f"https://reactome.org/content/detail/{item.get('stId', '')}",
            }
            for item in results[:5]
        ]
    except Exception as e:
        print(f"reactome fail: {e}")
        return []


def _query_enrichr(gene_list, description="query"):
    try:
        r = api_get(ENRICHR_ADD_URL, params={
            "list": "\n".join(gene_list[:200]),
            "description": description,
        })
        user_list_id = r.json().get("userListId", None)
        if user_list_id is None:
            return []
        r2 = api_get(ENRICHR_ENRICH_URL, params={
            "userListId": user_list_id,
            "backgroundType": "MSigDB_Hallmark_2020",
        })
        enrichment = r2.json().get("MSigDB_Hallmark_2020", [])
        return [
            {
                "term": entry[1] if len(entry) > 1 else "",
                "pvalue": entry[2] if len(entry) > 2 else np.nan,
                "overlap": entry[5] if len(entry) > 5 else "",
            }
            for entry in enrichment[:5]
        ]
    except Exception as e:
        print(f"enrichr fail: {e}")
        return []


def _query_string(gene_list):
    # only pass real gene symbols, skip score names
    real_genes = [g for g in gene_list if g and not g.endswith("_score") and len(g) <= 12]
    if not real_genes:
        return []
    try:
        if len(real_genes) == 1:
            r = api_get(STRING_IDS_URL, params={
                "identifiers": real_genes[0],
                "species": 9606,
                "caller_identity": "bioconverge",
            })
            data = r.json()
            if not data:
                return []
            return [{"protein_a": data[0].get("preferredName", ""), "protein_b": "", "score": np.nan}]
        r = api_get(STRING_URL, params={
            "identifiers": "%0d".join(real_genes[:10]),
            "species": 9606,
            "required_score": 400,
            "caller_identity": "bioconverge",
        })
        data = r.json()
        if not data:
            return []
        return [
            {
                "protein_a": edge.get("preferredName_A", ""),
                "protein_b": edge.get("preferredName_B", ""),
                "score": edge.get("score", np.nan),
            }
            for edge in data[:5]
        ]
    except Exception as e:
        print(f"string fail: {e}")
        return []


def _query_gwas(process_term):
    try:
        r = api_get(GWAS_URL, params={"query": process_term, "page": 0, "size": 5})
        traits = r.json().get("_embedded", {}).get("efoTraits", [])
        return [
            {"trait": t.get("trait", ""), "uri": t.get("uri", "")}
            for t in traits[:5]
        ]
    except Exception as e:
        print(f"gwas fail: {e}")
        return []


def _count_pubmed(query_term):
    try:
        r = api_get(PUBMED_SEARCH_URL, params={
            "db": "pubmed",
            "term": query_term,
            "retmax": 0,
            "retmode": "json",
        })
        return int(r.json().get("esearchresult", {}).get("count", 0))
    except Exception as e:
        print(f"pubmed fail: {e}")
        return 0


class HypothesisGenerator:
    def __init__(self, score_metadata, archetypes_df, fragility_df=None):
        self.score_metadata = score_metadata
        self.archetypes_df = archetypes_df
        self.fragility_df = fragility_df
        self._hypotheses_df = None
        self._repro_log = []

    def generate(self):
        print("generating hypotheses")
        rows = []
        for arch in sorted(self.archetypes_df["archetype"].unique()):
            arch_patients = self.archetypes_df[self.archetypes_df["archetype"] == arch]["patient_id"].tolist()
            for score_name, meta in self.score_metadata.items():
                process = meta.get("process", score_name)
                modality = meta.get("modality", "unknown")
                genes = meta.get("genes", [])
                ts = datetime.utcnow().isoformat()

                reactome_hits = _query_reactome(process)
                self._log(f"reactome {process}", ts)
                time.sleep(0.5)

                enrichr_hits = _query_enrichr(genes if genes else [score_name], description=process)
                self._log(f"enrichr {process}", datetime.utcnow().isoformat())
                time.sleep(0.5)

                string_hits = _query_string(genes)
                self._log(f"string {process}", datetime.utcnow().isoformat())
                time.sleep(0.5)

                gwas_hits = _query_gwas(process)
                self._log(f"gwas {process}", datetime.utcnow().isoformat())

                db_support = sum([
                    len(reactome_hits) > 0,
                    len(enrichr_hits) > 0,
                    len(string_hits) > 0,
                    len(gwas_hits) > 0,
                ])

                top_reactome = reactome_hits[0]["name"] if reactome_hits else ""
                top_enrichr  = enrichr_hits[0]["term"] if enrichr_hits else ""
                reactome_url = reactome_hits[0].get("url", "") if reactome_hits else ""

                pubmed_query = f"{process} breast cancer"
                pubmed_count = _count_pubmed(pubmed_query)
                self._log(f"pubmed {pubmed_query}", datetime.utcnow().isoformat())
                time.sleep(0.3)

                rows.append({
                    "archetype": arch,
                    "score": score_name,
                    "process": process,
                    "modality": modality,
                    "hypothesis": (
                        f"Archetype {arch} shows {process} ({modality}) signal. "
                        f"Top pathway: {top_reactome}. "
                        f"Top gene set: {top_enrichr}."
                    ),
                    "db_support": db_support,
                    "reactome_pathway": top_reactome,
                    "reactome_url": reactome_url,
                    "enrichr_term": top_enrichr,
                    "pubmed_count": pubmed_count,
                    "pubmed_flag": _pubmed_flag(pubmed_count),
                    "n_patients_archetype": len(arch_patients),
                })

        if rows:
            self._hypotheses_df = pd.DataFrame(rows).sort_values("db_support", ascending=False).reset_index(drop=True)
        else:
            self._hypotheses_df = pd.DataFrame()
        print("hypotheses done")
        return self

    def _log(self, query, timestamp):
        self._repro_log.append({"query": query, "timestamp": timestamp})

    def hypotheses(self):
        return self._hypotheses_df

    def cross_support(self):
        if self._hypotheses_df is None or self._hypotheses_df.empty:
            return pd.DataFrame()
        return (
            self._hypotheses_df
            .groupby("archetype")["db_support"]
            .agg(["mean", "max", "count"])
            .rename(columns={"mean": "mean_support", "max": "max_support", "count": "n_hypotheses"})
            .reset_index()
        )

    def reproducibility_log(self):
        return pd.DataFrame(self._repro_log)


def _pubmed_flag(count):
    if count > 100:
        return "convergent_with_prior_knowledge"
    if count < 10:
        return "exploratory"
    return "moderate_support"
