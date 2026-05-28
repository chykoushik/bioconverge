import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


class ConcordanceAnalyzer:
    def __init__(self, df, patient_col):
        self.patient_col = patient_col
        self.patients = df[patient_col].values
        self.score_cols = [c for c in df.columns if c != patient_col]
        self.X = df[self.score_cols].apply(pd.to_numeric, errors="coerce").values
        self._concordance_df = None
        self._convergence_df = None
        self._archetypes_df = None
        self._stability_dict = None
        self._km_model = None
        self._scaler = None

    def fit(self, n_archetypes=3, n_bootstrap=1000, random_state=42):
        print("fitting concordance")
        self._compute_concordance(n_bootstrap, random_state)
        self._compute_convergence()
        self._compute_archetypes(n_archetypes, random_state)
        self._compute_stability(n_archetypes, n_bootstrap, random_state)
        print("layer1 done")
        return self

    def _compute_concordance(self, n_bootstrap, random_state):
        rng = np.random.default_rng(random_state)
        n = len(self.score_cols)
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                x = self.X[:, i]
                y = self.X[:, j]
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 5:
                    rows.append({
                        "score_a": self.score_cols[i],
                        "score_b": self.score_cols[j],
                        "spearman_rho": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "n_patients": int(mask.sum()),
                    })
                    continue
                rho, _ = spearmanr(x[mask], y[mask])
                xm, ym = x[mask], y[mask]
                boot = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(xm), size=len(xm))
                    try:
                        rb, _ = spearmanr(xm[idx], ym[idx])
                        boot.append(rb)
                    except Exception:
                        pass
                ci_lo = float(np.percentile(boot, 2.5)) if boot else np.nan
                ci_hi = float(np.percentile(boot, 97.5)) if boot else np.nan
                rows.append({
                    "score_a": self.score_cols[i],
                    "score_b": self.score_cols[j],
                    "spearman_rho": float(rho),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_patients": int(mask.sum()),
                })
        self._concordance_df = pd.DataFrame(rows)

    def _compute_convergence(self):
        global_means = np.nanmean(self.X, axis=0)
        global_stds = np.nanstd(self.X, axis=0)
        global_stds[global_stds == 0] = 1.0
        patient_conv = []
        for p_idx in range(len(self.patients)):
            vals = self.X[p_idx]
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                patient_conv.append(np.nan)
                continue
            z = (vals[valid] - global_means[valid]) / global_stds[valid]
            mean_z = np.mean(z)
            std_z = np.std(z)
            conv = float(np.clip(1.0 - std_z / (abs(mean_z) + 1e-8), -1.0, 1.0))
            patient_conv.append(conv)

        conv_series = pd.Series(patient_conv)
        t33 = conv_series.quantile(0.33)
        t67 = conv_series.quantile(0.67)

        def _stratum(v):
            if pd.isna(v):
                return "unknown"
            if v >= t67:
                return "convergent_high"
            if v >= t33:
                return "convergent_mid"
            return "convergent_low"

        self._convergence_df = pd.DataFrame({
            "patient_id": self.patients,
            "convergence_index": patient_conv,
            "stratum": [_stratum(v) for v in patient_conv],
        })

    def _fill_X(self):
        X_filled = self.X.copy()
        for j in range(X_filled.shape[1]):
            col = X_filled[:, j]
            med = np.nanmedian(col)
            col[np.isnan(col)] = med if not np.isnan(med) else 0.0
        return X_filled

    def _compute_archetypes(self, n_archetypes, random_state):
        X_filled = self._fill_X()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        km = KMeans(n_clusters=n_archetypes, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        self._km_model = km
        self._scaler = scaler
        self._archetypes_df = pd.DataFrame({
            "patient_id": self.patients,
            "archetype": labels,
        })

    def _compute_stability(self, n_archetypes, n_bootstrap, random_state):
        rng = np.random.default_rng(random_state)
        X_filled = self._fill_X()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        base_labels = self._archetypes_df["archetype"].values
        n_iters = min(n_bootstrap, 200)
        aris = []
        for _ in range(n_iters):
            idx = rng.integers(0, len(X_scaled), size=len(X_scaled))
            try:
                km2 = KMeans(n_clusters=n_archetypes, random_state=None, n_init=3, max_iter=100)
                km2.fit(X_scaled[idx])
                pred = km2.predict(X_scaled)
                aris.append(adjusted_rand_score(base_labels, pred))
            except Exception:
                pass
        self._stability_dict = {
            "mean_ari": float(np.mean(aris)) if aris else np.nan,
            "std_ari": float(np.std(aris)) if aris else np.nan,
            "n_bootstrap": len(aris),
            "interpretation": "stable" if (aris and np.mean(aris) > 0.6) else "unstable",
        }

    def concordance(self):
        return self._concordance_df

    def convergence(self):
        return self._convergence_df

    def discordance(self):
        if self._archetypes_df is None or self._convergence_df is None:
            return None
        merged = self._archetypes_df.merge(self._convergence_df, on="patient_id")
        return merged[merged["stratum"] == "convergent_low"].copy().reset_index(drop=True)

    def archetypes(self):
        return self._archetypes_df

    def stability(self):
        return self._stability_dict