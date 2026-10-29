"""
Preprocessing of the BRFSS 2015 dataset.

Pipeline:
    1. clean_features    : survey codes -> NaN / 0, recoding, skip patterns.
    2. fit_preprocessing : statistics computed on the training data only.
    3. transform         : clipping, imputation, one-hot, standardization.

Usage:
    from helpers import load_csv_data
    from preprocessing import load_feature_names, preprocess

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")
    names = load_feature_names("data/dataset/x_train.csv")
    x_tr, x_te, out_names, params = preprocess(x_train, x_test, names)

Code references: BRFSS 2015 Codebook (CDC).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Rules per variable
# ---------------------------------------------------------------------------
# type:
#   "num" : continuous           -> median imputation + standardization
#   "ord" : ordered categories   -> median imputation + standardization
#   "bin" : yes/no as 1/0        -> mode imputation
#   "nom" : unordered categories -> one-hot (NaN is its own category)
# nan  : codes set to NaN (don't know / refused / missing)
# zero : codes meaning "none" -> 0
# map  : recoding of the remaining values
# log  : log(1 + x) for skewed variables

YES_NO = {"type": "bin", "nan": [7, 9], "map": {1: 1, 2: 0}}

RULES = {
    # --- General health ------------------------------------------------------
    "GENHLTH":  {"type": "ord", "nan": [7, 9]},                 # 1 excellent ... 5 poor
    "PHYSHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # days of poor physical health
    "MENTHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # days of poor mental health
    "POORHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # skip pattern (see below)
    # --- Access to care ------------------------------------------------------
    "HLTHPLN1": YES_NO,
    "PERSDOC2": {"type": "bin", "nan": [7, 9], "map": {1: 1, 2: 1, 3: 0}},  # has a personal doctor
    "MEDCOST":  YES_NO,
    "CHECKUP1": {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 2, 3: 3, 4: 4, 8: 5}},  # 8 = never -> 5
    # --- Blood pressure / cholesterol ----------------------------------------
    "BPHIGH4":  {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 0, 3: 0, 4: 0.5}},  # 2 = pregnancy only, 4 = borderline
    "BPMEDS":   YES_NO,                                          # skip pattern on BPHIGH4
    "BLOODCHO": YES_NO,
    "CHOLCHK":  {"type": "ord", "nan": [7, 9]},                  # skip pattern on BLOODCHO
    "TOLDHI2":  YES_NO,                                          # skip pattern on BLOODCHO
    # --- Chronic diseases ----------------------------------------------------
    "CVDSTRK3": YES_NO,
    "_ASTHMS1": {"type": "nom", "nan": [9]},                     # current / former / never
    "CHCSCNCR": YES_NO,
    "CHCOCNCR": YES_NO,
    "CHCCOPD1": YES_NO,
    "HAVARTH3": YES_NO,
    "ADDEPEV2": YES_NO,
    "CHCKIDNY": YES_NO,
    "DIABETE3": {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 0, 3: 0, 4: 0.5}},  # 4 = prediabetes
    # --- Demographics --------------------------------------------------------
    "SEX":      {"type": "bin", "nan": [], "map": {1: 1, 2: 0}},  # 1 = male
    "_AGE80":   {"type": "num", "nan": []},                       # age (capped at 80)
    "MARITAL":  {"type": "nom", "nan": [9]},
    "EDUCA":    {"type": "ord", "nan": [9]},
    "RENTHOM1": {"type": "nom", "nan": [7, 9]},
    "VETERAN3": YES_NO,
    "EMPLOY1":  {"type": "nom", "nan": [9]},
    "CHILDREN": {"type": "num", "nan": [77, 99], "zero": [88]},
    "INCOME2":  {"type": "ord", "nan": [77, 99]},
    "INTERNET": YES_NO,
    "_RACE":    {"type": "nom", "nan": [9]},
    # --- Body size (computed by the CDC) -------------------------------------
    "HTM4":     {"type": "num", "nan": []},                       # height (m)
    "WTKG3":    {"type": "num", "nan": []},                       # weight (kg)
    "_BMI5":    {"type": "num", "nan": []},
    # --- Disability ----------------------------------------------------------
    "QLACTLM2": YES_NO,
    "USEEQUIP": YES_NO,
    "BLIND":    YES_NO,
    "DECIDE":   YES_NO,
    "DIFFWALK": YES_NO,
    "DIFFDRES": YES_NO,
    "DIFFALON": YES_NO,
    # --- Tobacco (computed variable, no skip pattern) ------------------------
    "_SMOKER3": {"type": "ord", "nan": [9]},                      # 1 daily ... 4 never
    "USENOW3":  {"type": "ord", "nan": [7, 9]},                   # smokeless tobacco
    # --- Alcohol (computed variables, non-drinkers = 0) ----------------------
    "DRNKANY5": YES_NO,
    "_DRNKWEK": {"type": "num", "nan": [99900], "log": True},     # drinks per week
    "_RFBING5": {"type": "bin", "nan": [9], "map": {1: 0, 2: 1}},  # binge drinking
    "_RFDRHV5": {"type": "bin", "nan": [9], "map": {1: 0, 2: 1}},  # heavy drinking
    # --- Diet (computed, times per day) --------------------------------------
    "_FRUTSUM": {"type": "num", "nan": []},
    "_VEGESUM": {"type": "num", "nan": []},
    # --- Physical activity ---------------------------------------------------
    "EXERANY2": YES_NO,
    "PA1MIN_":  {"type": "num", "nan": [], "log": True},          # minutes of activity per week
    "PA1VIGM_": {"type": "num", "nan": [], "log": True},          # vigorous minutes per week
    "STRFREQ_": {"type": "num", "nan": [99]},                     # strength training per week
    "_PACAT1":  {"type": "ord", "nan": [9]},                      # 1 very active ... 4 inactive
    "_PAINDX1": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},
    "_PASTRNG": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},
    # --- Arthritis (computed, 3 = no arthritis) ------------------------------
    "_LMTACT1": {"type": "nom", "nan": [9]},
    "_LMTWRK1": {"type": "nom", "nan": [9]},
    "JOINPAIN": {"type": "num", "nan": [77, 99]},                 # pain 0-10, skip pattern
    # --- Prevention ----------------------------------------------------------
    "_RFSEAT2": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},  # always wears seatbelt
    "FLUSHOT6": YES_NO,
    "PNEUVAC3": YES_NO,
    "HIVTST6":  YES_NO,
}

# High-cardinality variable, only used if use_state=True.
OPTIONAL_RULES = {
    "_STATE": {"type": "nom", "nan": []},
}

# ---------------------------------------------------------------------------
# 1b. Feature engineering (enabled with the `features` argument)
# ---------------------------------------------------------------------------
# Positive rate in x_train:
#   _AGEG5YR : 0.5 % (18-24) -> 22 % (80+), non-linear -> one-hot.
#   GENHLTH  : 2 % (excellent) -> 32 % (poor), irregular steps -> one-hot.
#   _BMI5CAT : 8.7 / 6.7 / 9.3 / 11.4 %, U-shaped -> one-hot + BMI^2.

AGE_ONEHOT_RULE = {"_AGEG5YR": {"type": "nom", "nan": [14]}}   # 14 = don't know
BMI_ONEHOT_RULE = {"_BMI5CAT": {"type": "nom", "nan": []}}
GENHLTH_ONEHOT_RULE = {"GENHLTH_CAT": {"type": "nom", "nan": [7, 9],
                                       "source": "GENHLTH"}}

# Optional survey modules (asked in some states only, 70-99 % NaN).
# One-hot with NaN as its own category, kept despite the NaN threshold.
# Note: daily aspirin (CVDASPRN) is likely a consequence of heart disease.
EXTRA_RULES = {
    "CVDASPRN": {"type": "nom", "nan": [7, 9], "keep_nan": True},  # daily aspirin
    "CIMEMLOS": {"type": "nom", "nan": [7, 9], "keep_nan": True},  # memory loss
    "INSULIN":  {"type": "nom", "nan": [9], "keep_nan": True},     # takes insulin
    "PREDIAB1": {"type": "nom", "nan": [7, 9], "keep_nan": True},  # prediabetes
}

# Squared terms: (A, A) -> (A - median)^2
SQUARES = [("_AGE80", "_AGE80"), ("_BMI5", "_BMI5")]

# Targeted interactions between cardiovascular risk factors.
# A*B = (A - median_A) * (B - median_B), binary variables are not centered.
INTERACTIONS = [
    ("_AGE80", "SEX"),
    ("_AGE80", "DIABETE3"),
    ("_AGE80", "BPHIGH4"),
    ("_AGE80", "TOLDHI2"),
    ("_AGE80", "_SMOKER3"),
    ("_AGE80", "CVDSTRK3"),
    ("_AGE80", "GENHLTH"),
    ("SEX", "_SMOKER3"),
    ("SEX", "DIABETE3"),
    ("DIABETE3", "BPHIGH4"),
    ("BPHIGH4", "TOLDHI2"),
    ("_BMI5", "DIABETE3"),
]

# All pairwise products of these 15 risk factors (105 interactions).
TOP_RISK_FEATURES = [
    "_AGE80", "SEX", "GENHLTH", "DIABETE3", "BPHIGH4", "TOLDHI2", "CVDSTRK3",
    "_SMOKER3", "CHCKIDNY", "DIFFWALK", "CHCCOPD1", "HAVARTH3", "_BMI5",
    "INCOME2", "PHYSHLTH",
]

# All options False = baseline preprocessing
FEATURE_DEFAULTS = {
    "age_onehot": False,      # one-hot of the 13 age groups (_AGEG5YR)
    "bmi_onehot": False,      # one-hot of the 4 BMI categories (_BMI5CAT)
    "genhlth_onehot": False,  # one-hot of GENHLTH (in addition to the ordinal)
    "squares": False,         # age^2 and BMI^2
    "interactions": False,    # products in INTERACTIONS
    "extra_modules": False,   # columns of EXTRA_RULES
    "pairwise_top": False,    # all pairwise products of TOP_RISK_FEATURES
}

# Final configuration, chosen by cross-validation
# (see experiments_features.py and results/feature_ablation.md).
RECOMMENDED_FEATURES = {
    "age_onehot": True,
    "bmi_onehot": True,
    "genhlth_onehot": True,
    "squares": True,
    "interactions": True,
    "extra_modules": True,
    "pairwise_top": True,
}
RECOMMENDED_USE_STATE = False    # _STATE one-hot: no gain in CV
RECOMMENDED_LAMBDA = 1e-4        # L2 regularization of the logistic regression


def _feature_options(features):
    opts = dict(FEATURE_DEFAULTS)
    if features:
        unknown = set(features) - set(FEATURE_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown options: {unknown}")
        opts.update(features)
    return opts


def _engineered_pairs(opts):
    pairs = []
    if opts["squares"]:
        pairs += SQUARES
    if opts["interactions"]:
        pairs += INTERACTIONS
    if opts["pairwise_top"]:
        top = TOP_RISK_FEATURES
        pairs += [(top[i], top[j]) for i in range(len(top))
                  for j in range(i + 1, len(top))]
    # remove duplicates (A*B and B*A), keeping the order
    seen, unique = set(), []
    for a, b in pairs:
        key = frozenset((a, b)) if a != b else (a,)
        if key not in seen:
            seen.add(key)
            unique.append((a, b))
    return unique


# Target leakage: only asked to people who already had a heart attack.
LEAKAGE = ["HAREHAB1"]

# Columns not used as features, by group.
DROPPED_GROUPS = {
    "target leakage": LEAKAGE,
    "administrative / identifiers": [
        "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
        "SEQNO", "_PSU", "QSTVER", "QSTLANG", "MSCODE"],
    "sampling weights": [
        "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_CLLCPWT",
        "_DUALCOR", "_DUALUSE"],
    "phone call logistics (landline / cell)": [
        "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
        "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT",
        "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "NUMHHOL2",
        "NUMPHON2", "CPDEMO1"],
}


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def load_feature_names(x_train_path):
    """Column names (without 'Id'), in the same order as load_csv_data."""
    with open(x_train_path, "r") as f:
        header = f.readline().strip().split(",")
    return header[1:]


def _col(x, names, name):
    return x[:, names.index(name)].astype(float)


# ---------------------------------------------------------------------------
# 3. Cleaning (same for train and test, no fitted statistics)
# ---------------------------------------------------------------------------
def _apply_rule(raw, rule):
    col = raw.copy()
    for code in rule.get("nan", []):
        col[raw == code] = np.nan
    for code in rule.get("zero", []):
        col[raw == code] = 0.0
    if "map" in rule:
        # values not listed in "map" become NaN
        mapped = np.full_like(col, np.nan)
        for old, new in rule["map"].items():
            mapped[raw == old] = new
        col = mapped
    if rule.get("log", False):
        col = np.log1p(np.maximum(col, 0))
    return col


def _apply_skip_patterns(cols, x, names):
    """Fill NaN of questions skipped because of a previous answer."""
    def raw(n):
        return _col(x, names, n)

    def fill(child, mask, value):
        if child in cols:
            c = cols[child]
            c[np.isnan(c) & mask] = value

    # POORHLTH only asked if PHYSHLTH > 0 or MENTHLTH > 0
    fill("POORHLTH", (raw("PHYSHLTH") == 88) & (raw("MENTHLTH") == 88), 0.0)
    # BPMEDS only asked if hypertensive
    fill("BPMEDS", np.isin(raw("BPHIGH4"), [2, 3, 4]), 0.0)
    # CHOLCHK / TOLDHI2 only asked if cholesterol was ever checked
    fill("CHOLCHK", raw("BLOODCHO") == 2, 5.0)   # 5 = never checked
    fill("TOLDHI2", raw("BLOODCHO") == 2, 0.0)
    # JOINPAIN only asked if arthritis
    fill("JOINPAIN", raw("HAVARTH3") == 2, 0.0)
    # no exercise -> 0 minutes
    no_exercise = raw("EXERANY2") == 2
    fill("PA1MIN_", no_exercise, 0.0)
    fill("PA1VIGM_", no_exercise, 0.0)
    fill("STRFREQ_", no_exercise, 0.0)
    return cols


def _build_rules(use_state=False, features=None):
    """Active rules given the options (RULES + optional additions)."""
    opts = _feature_options(features)
    rules = dict(RULES)
    if use_state:
        rules.update(OPTIONAL_RULES)
    if opts["age_onehot"]:
        rules.update(AGE_ONEHOT_RULE)
    if opts["bmi_onehot"]:
        rules.update(BMI_ONEHOT_RULE)
    if opts["genhlth_onehot"]:
        rules.update(GENHLTH_ONEHOT_RULE)
    if opts["extra_modules"]:
        rules.update(EXTRA_RULES)
    return rules


def clean_features(x, names, use_state=False, features=None):
    """Apply the codebook rules; columns without a rule are dropped.

    Returns:
        x_clean, clean_names, types
    """
    rules = _build_rules(use_state, features)

    leaks = [n for n in rules if n in LEAKAGE or rules[n].get("source") in LEAKAGE]
    if leaks:
        raise ValueError(f"Target leakage variables in RULES: {leaks}")

    cols = {}
    for name, rule in rules.items():
        source = rule.get("source", name)   # original column (can be copied)
        if source not in names:
            continue
        cols[name] = _apply_rule(_col(x, names, source), rule)

    cols = _apply_skip_patterns(cols, x, names)

    clean_names = list(cols.keys())
    types = [rules[n]["type"] for n in clean_names]
    x_clean = np.column_stack([cols[n] for n in clean_names])
    return x_clean, clean_names, types


# ---------------------------------------------------------------------------
# 4. Fit on train / transform train and test
# ---------------------------------------------------------------------------
def fit_preprocessing(x_clean, names, types, nan_threshold=0.4,
                      add_nan_indicators=True, indicator_min_ratio=0.05,
                      clip_percentiles=(1, 99), pairs=None, force_keep=()):
    """Compute all preprocessing parameters on the training data.

    Args:
        pairs: list of (A, B) -> adds the feature (A - c_A) * (B - c_B),
               c = median for num/ord, 0 for binary. Pairs with an
               unavailable variable are ignored.
        force_keep: variables kept even above the NaN threshold.
    """
    nan_ratio = np.mean(np.isnan(x_clean), axis=0)
    keep = (nan_ratio <= nan_threshold) | np.isin(names, list(force_keep))

    params = {"names_in": names, "types": types, "keep": keep,
              "nan_ratio": nan_ratio, "fill": {}, "categories": {},
              "clip": {}, "indicators": [], "mean": None, "std": None,
              "pairs": [], "center": {}}

    for j, (n, t) in enumerate(zip(names, types)):
        if not keep[j]:
            continue
        c = x_clean[:, j]
        valid = c[~np.isnan(c)]
        # outlier clipping bounds for continuous variables
        if t == "num" and clip_percentiles is not None:
            lo, hi = np.percentile(valid, clip_percentiles)
            params["clip"][n] = (lo, hi)
            valid = np.clip(valid, lo, hi)
        if t in ("num", "ord"):
            params["fill"][n] = np.median(valid)
        elif t == "bin":
            params["fill"][n] = 1.0 if np.mean(valid) >= 0.5 else 0.0
        elif t == "nom":
            params["categories"][n] = np.unique(valid)
        if add_nan_indicators and t != "nom" and nan_ratio[j] >= indicator_min_ratio:
            params["indicators"].append(n)
        if t in ("num", "ord"):
            params["center"][n] = params["fill"][n]
        elif t == "bin":
            params["center"][n] = 0.0

    # squared terms / interactions between available variables only
    for a, b in (pairs or []):
        if a in params["center"] and b in params["center"]:
            params["pairs"].append((a, b))

    # mean / std of the scaled columns (after imputation)
    x_tmp, out_names, scale_mask = _build_matrix(x_clean, params)
    params["scale_mask"] = scale_mask
    params["mean"] = np.mean(x_tmp[:, scale_mask], axis=0)
    std = np.std(x_tmp[:, scale_mask], axis=0)
    params["std"] = np.where(std == 0, 1.0, std)
    params["names_out"] = out_names
    return params


def _build_matrix(x_clean, params):
    names, types, keep = params["names_in"], params["types"], params["keep"]
    blocks, out_names, scale = [], [], []
    imputed = {}   # columns after clipping + imputation (for the interactions)
    for j, (n, t) in enumerate(zip(names, types)):
        if not keep[j]:
            continue
        c = x_clean[:, j].copy()
        is_nan = np.isnan(c)
        if t == "nom":
            for cat in params["categories"][n]:
                blocks.append((c == cat).astype(float))
                out_names.append(f"{n}={cat:g}")
                scale.append(False)
            blocks.append(is_nan.astype(float))          # "missing" category
            out_names.append(f"{n}=NaN")
            scale.append(False)
        else:
            if n in params["clip"]:
                c = np.clip(c, *params["clip"][n])   # NaN stay NaN
            c[is_nan] = params["fill"][n]
            imputed[n] = c
            blocks.append(c)
            out_names.append(n)
            scale.append(t in ("num", "ord"))
            if n in params["indicators"]:
                blocks.append(is_nan.astype(float))
                out_names.append(f"{n}_was_nan")
                scale.append(False)

    # squared terms and interactions (centered, then standardized)
    for a, b in params.get("pairs", []):
        ca = imputed[a] - params["center"][a]
        cb = imputed[b] - params["center"][b]
        blocks.append(ca * cb)
        out_names.append(f"{a}^2" if a == b else f"{a}*{b}")
        scale.append(True)
    return np.column_stack(blocks), out_names, np.array(scale)


def transform(x_clean, params):
    x_out, _, _ = _build_matrix(x_clean, params)
    m = params["scale_mask"]
    x_out[:, m] = (x_out[:, m] - params["mean"]) / params["std"]
    return x_out


# ---------------------------------------------------------------------------
# 5. All-in-one functions
# ---------------------------------------------------------------------------
def fit_from_clean(x_clean, names_c, types, nan_threshold=0.4,
                   add_nan_indicators=True, use_state=False,
                   clip_percentiles=(1, 99), features=None):
    """fit_preprocessing with the pairs / kept columns given by the options."""
    opts = _feature_options(features)
    rules = _build_rules(use_state, features)
    force_keep = [n for n, r in rules.items() if r.get("keep_nan", False)]
    return fit_preprocessing(x_clean, names_c, types, nan_threshold,
                             add_nan_indicators,
                             clip_percentiles=clip_percentiles,
                             pairs=_engineered_pairs(opts),
                             force_keep=force_keep)


def preprocess(x_train, x_test, feature_names, nan_threshold=0.4,
               add_nan_indicators=True, use_state=False,
               clip_percentiles=(1, 99), verbose=True, features=None):
    """Clean, fit on x_train and transform x_train and x_test.

    Args:
        x_train, x_test: outputs of load_csv_data (without the Id column).
        features: feature engineering options (None = baseline,
                  RECOMMENDED_FEATURES = final configuration).
    Returns:
        x_train_proc, x_test_proc, output_names, params
    """
    xtr_c, names_c, types = clean_features(x_train, feature_names, use_state, features)
    xte_c, _, _ = clean_features(x_test, feature_names, use_state, features)

    params = fit_from_clean(xtr_c, names_c, types, nan_threshold,
                            add_nan_indicators, use_state,
                            clip_percentiles, features)
    x_tr = transform(xtr_c, params)
    x_te = transform(xte_c, params)

    if verbose:
        dropped = [n for n, k in zip(names_c, params["keep"]) if not k]
        print(f"Original columns          : {len(feature_names)}")
        print(f"After codebook rules      : {len(names_c)}")
        print(f"Dropped (NaN > {nan_threshold:.0%})     : {dropped if dropped else 'none'}")
        print(f"Final features            : {x_tr.shape[1]} "
              f"(including one-hot and NaN indicators)")
    return x_tr, x_te, params["names_out"], params


def nan_report(x_clean, names, top=15):
    """Print the cleaned variables with the most remaining NaN."""
    r = np.mean(np.isnan(x_clean), axis=0)
    for i in np.argsort(-r)[:top]:
        print(f"{names[i]:10s} {r[i]:6.1%}")


# ---------------------------------------------------------------------------
# 6. Train / validation split, then preprocessing
# ---------------------------------------------------------------------------
def split_data(x, y, ratio=0.8, seed=1, stratify=True):
    """Split into training and validation sets.

    stratify=True keeps the same positive rate in both sets.
    Returns x_tr, x_va, y_tr, y_va.
    """
    rng = np.random.default_rng(seed)
    if stratify:
        tr_idx, va_idx = [], []
        for label in np.unique(y):
            idx = rng.permutation(np.where(y == label)[0])
            cut = int(ratio * len(idx))
            tr_idx.append(idx[:cut])
            va_idx.append(idx[cut:])
        tr_idx = rng.permutation(np.concatenate(tr_idx))
        va_idx = rng.permutation(np.concatenate(va_idx))
    else:
        idx = rng.permutation(len(y))
        cut = int(ratio * len(y))
        tr_idx, va_idx = idx[:cut], idx[cut:]
    return x[tr_idx], x[va_idx], y[tr_idx], y[va_idx]


def preprocess_split(x_tr, x_va, x_te, feature_names, nan_threshold=0.4,
                     add_nan_indicators=True, use_state=False,
                     clip_percentiles=(1, 99), verbose=True, features=None):
    """Preprocessing after the split: all statistics are computed on x_tr
    only, then applied to x_va and x_te.

    Returns X_tr, X_va, X_te, output_names, params.
    """
    tr_c, names_c, types = clean_features(x_tr, feature_names, use_state, features)
    va_c, _, _ = clean_features(x_va, feature_names, use_state, features)
    te_c, _, _ = clean_features(x_te, feature_names, use_state, features)

    params = fit_from_clean(tr_c, names_c, types, nan_threshold,
                            add_nan_indicators, use_state,
                            clip_percentiles, features)
    X_tr = transform(tr_c, params)
    X_va = transform(va_c, params)
    X_te = transform(te_c, params)

    if verbose:
        print(f"train {X_tr.shape} | validation {X_va.shape} | test {X_te.shape}")
    return X_tr, X_va, X_te, params["names_out"], params


def add_bias(*arrays):
    """Add a column of ones (bias term) at the start of each matrix."""
    out = tuple(np.c_[np.ones(len(a)), a] for a in arrays)
    return out if len(out) > 1 else out[0]
