"""Name normalisation, comparison and character-level corruption primitives."""
from __future__ import annotations

import random
import re
import string
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:
    from rapidfuzz.distance import JaroWinkler, Levenshtein
except ImportError:  # pragma: no cover - rapidfuzz is a declared dependency
    JaroWinkler = None
    Levenshtein = None

# The minimal NCVR extract holds one administrative race/ethnicity code. Small
# cells are collapsed so that subgroup linkage quality can be estimated.
NCVR_OTHER = "Asian / IPI / Mixed / Other / Unknown"
NCVR_GROUP_MAP = {
    "w": "Non-Hispanic White",
    "b": "Non-Hispanic Black",
    "a": NCVR_OTHER,
    "i": NCVR_OTHER,
    "p": NCVR_OTHER,
    "m": NCVR_OTHER,
    "o": NCVR_OTHER,
    "u": NCVR_OTHER,
    "": NCVR_OTHER,
}


def classify_ncvr_ethnicity(value: object) -> str:
    if pd.isna(value):
        return NCVR_OTHER
    return NCVR_GROUP_MAP.get(str(value).strip().lower(), NCVR_OTHER)


def normalise_name(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z\s'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise names and derive the blocking keys used by the linkage model."""
    out = df.copy()
    out["forename_norm"] = out["forename_norm"].map(normalise_name)
    out["surname_norm"] = out["surname_norm"].map(normalise_name)
    out["forename_initial"] = out["forename_norm"].str[:1].fillna("")
    out["surname_initial"] = out["surname_norm"].str[:1].fillna("")
    out["forename_prefix2"] = out["forename_norm"].str[:2].fillna("")
    out["surname_prefix2"] = out["surname_norm"].str[:2].fillna("")
    out["forename_len"] = out["forename_norm"].str.len().fillna(0).astype(int)
    out["surname_len"] = out["surname_norm"].str.len().fillna(0).astype(int)
    out["surname_len_bin"] = pd.cut(
        out["surname_len"], bins=[-1, 3, 6, 10, 100], labels=["0-3", "4-6", "7-10", "11+"]
    ).astype(str)
    return out


def jw_similarity(a: object, b: object) -> float:
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    if JaroWinkler is not None:
        return float(JaroWinkler.normalized_similarity(a, b))
    return SequenceMatcher(None, a, b).ratio()


def lev_distance(a: object, b: object) -> int:
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    if Levenshtein is not None:
        return int(Levenshtein.distance(a, b))
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(dp[j] + 1, ndp[-1] + 1, dp[j - 1] + (ca != cb)))
        dp = ndp
    return dp[-1]


def corrupt_string(value: str, rng: random.Random, max_edits: int = 1) -> str:
    """Apply ``max_edits`` random substitution/deletion/insertion/transposition edits."""
    value = normalise_name(value)
    if not value:
        return value
    chars = list(value)
    for _ in range(max(1, max_edits)):
        op = rng.choice(["substitute", "delete", "insert", "transpose"])
        if op == "substitute" and chars:
            pos = rng.randrange(len(chars))
            old = chars[pos]
            new = rng.choice(string.ascii_uppercase)
            tries = 0
            while new == old and tries < 10:
                new = rng.choice(string.ascii_uppercase)
                tries += 1
            chars[pos] = new
        elif op == "delete" and len(chars) > 1:
            del chars[rng.randrange(len(chars))]
        elif op == "insert":
            chars.insert(rng.randrange(len(chars) + 1), rng.choice(string.ascii_uppercase))
        elif op == "transpose" and len(chars) > 1:
            pos = rng.randrange(len(chars) - 1)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    out = "".join(chars)
    if out == value:
        return value + rng.choice(string.ascii_uppercase)
    return out


def assign_error_flags(n: int, p_forename: float, p_surname: float, p_both: float,
                       p_full_surname: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw per-record error flags that hit the requested marginal and joint rates.

    ``p_both`` is the forename-and-surname co-occurrence rate. ``p_full_surname``
    is a record-level rate, constrained not to exceed the surname-error rate.
    """
    rng = np.random.default_rng(seed)
    p_forename = max(0.0, min(1.0, float(p_forename)))
    p_surname = max(0.0, min(1.0, float(p_surname)))
    p_both = max(0.0, min(float(p_both), p_forename, p_surname))
    p_full_surname = max(0.0, min(float(p_full_surname), p_surname))
    forename_error = rng.random(n) < p_forename
    surname_error = np.zeros(n, dtype=bool)
    fe_idx = np.where(forename_error)[0]
    not_fe_idx = np.where(~forename_error)[0]
    if len(fe_idx):
        surname_error[fe_idx] = rng.random(len(fe_idx)) < p_both / max(p_forename, 1e-12)
    p_surname_given_not_fe = max(p_surname - p_both, 0.0) / max(1 - p_forename, 1e-12)
    if len(not_fe_idx):
        surname_error[not_fe_idx] = rng.random(len(not_fe_idx)) < min(1.0, p_surname_given_not_fe)
    full_replacement = np.zeros(n, dtype=bool)
    surname_idx = np.where(surname_error)[0]
    if len(surname_idx) and p_surname > 0:
        full_replacement[surname_idx] = rng.random(len(surname_idx)) < min(1.0, p_full_surname / p_surname)
    return forename_error, surname_error, full_replacement


def corrupt_names(df: pd.DataFrame, surname_pool_source: pd.DataFrame, seed: int,
                  forename_error_rate: float, surname_error_rate: float,
                  multi_field_error_rate: float, full_surname_replacement_rate: float) -> pd.DataFrame:
    """Corrupt forename/surname with known flags; full replacements draw from ``surname_pool_source``.

    Used both to build the observed task (pool = the clean reference itself) and
    to corrupt synthetic replicas (pool = the observed clean reference).
    """
    rng = random.Random(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    fe, se, fsr = assign_error_flags(n, forename_error_rate, surname_error_rate,
                                     multi_field_error_rate, full_surname_replacement_rate, seed)
    out["forename_was_corrupted"] = fe
    out["surname_was_corrupted"] = se
    out["full_surname_replacement_flag"] = fsr
    surnames_pool = list(surname_pool_source["surname_norm"].dropna().map(normalise_name).unique())
    for i in range(n):
        if fe[i]:
            out.at[i, "forename_norm"] = corrupt_string(out.at[i, "forename_norm"], rng)
        if se[i]:
            if fsr[i] and surnames_pool:
                current = out.at[i, "surname_norm"]
                replacement = rng.choice(surnames_pool)
                tries = 0
                while replacement == current and tries < 50:
                    replacement = rng.choice(surnames_pool)
                    tries += 1
                out.at[i, "surname_norm"] = replacement
            else:
                out.at[i, "surname_norm"] = corrupt_string(out.at[i, "surname_norm"], rng)
    return out
