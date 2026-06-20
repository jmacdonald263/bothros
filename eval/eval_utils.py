"""Self-contained eval helpers vendored from the BOTHROS development repo so the
oracle harness reproduces the published numbers byte-for-byte. (Sign normalisation
+ confidence intervals; copied verbatim — keep in sync if the dev repo changes them.)
"""
from __future__ import annotations

import math
import random
import re
from typing import Callable

# --- sign normalisation (from the dev pipeline) ---------------------------------
_BCODE_RE = re.compile(r"B\d+[A-Z]?$")
_NUM_RE = re.compile(r"num_(\d+)$")
_DIGIT_RE = re.compile(r"\d+$")

# Ideogram naming equivalences: the corpus transliteration parser renders some
# logograms as English words ("barley", "cloth"); the classifier emits the
# canonical Aegean codes ("hord", "tela"). Canonicalise both ends to agree.
_IDEOGRAM_ALIASES = {
    "man": "vir", "woman": "mul",
    "ram": "ovis_m", "ewe": "ovis_f",
    "bull": "bos_m", "cow": "bos_f",
    "boar": "sus_m", "sow": "sus_f",
    "he-goat": "cap_m", "she-goat": "cap_f",
    "horse": "equ", "stallion": "equ_m", "mare": "equ_f",
    "wheat": "gra", "barley": "hord",
    "olive oil": "ole", "olive_oil": "ole", "wine": "vin",
    "cloth": "tela", "wool": "lana",
    "gold": "aur", "silver": "arg", "bronze": "aes",
    "flour": "far", "arable": "are",
    "cyperus": "cyp", "honey": "meri",
    "horn": "cornu", "spice": "arom",
    "wheel": "rota", "chariot": "bigae", "crescent": "luna",
    "BIG": "bigae", "big": "bigae",
    "ME±RI": "meri", "me±ri": "meri",
    "CAP:f": "cap_f", "cap:f": "cap_f",
    "OVIS:x": "ovis", "ovis:x": "ovis",
    "TELA:1": "tela_1", "tela:1": "tela_1", "TELA_1": "tela_1",
    "TELA:x": "tela", "tela:x": "tela",
    "CUR": "cur", "cur": "cur",
}


def _normalize_sign(s: str) -> str:
    """Canonical sign form for matching: collapses case + numeral variants."""
    if not s:
        return s
    if _BCODE_RE.match(s):
        return s
    m = _NUM_RE.match(s)
    if m:
        return m.group(1)
    if _DIGIT_RE.match(s):
        return s
    if s.startswith("*"):
        return s.lower()
    canon = s.lower().replace(":", "_").replace(";", "_")
    return _IDEOGRAM_ALIASES.get(canon, canon)


# --- confidence intervals (from the dev repo's eval_stats) -----------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. (lo, hi). Assumes iid
    observations — over-tight when signs cluster within tablets; report alongside
    cluster_bootstrap_ci, not instead of it."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(per_cluster: dict, stat: Callable[[list], float],
                         reps: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Cluster bootstrap: resample CLUSTERS (tablets) with replacement, pool their
    items, recompute `stat`. Respects within-tablet correlation (the honest CI)."""
    rng = random.Random(seed)
    keys = sorted(per_cluster)
    if not keys:
        raise ValueError("no clusters")
    point = stat([x for k in keys for x in per_cluster[k]])
    dist = []
    for _ in range(reps):
        sample_keys = [rng.choice(keys) for _ in keys]
        pooled = [x for k in sample_keys for x in per_cluster[k]]
        dist.append(stat(pooled))
    dist.sort()
    lo = dist[int((alpha / 2) * reps)]
    hi = dist[min(reps - 1, int((1 - alpha / 2) * reps))]
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "n_clusters": len(keys), "reps": reps, "seed": seed}
