import numpy as np 
"""
Preprocessing du dataset BRFSS 2015 (CS433 - Projet 1), numpy uniquement.

Pipeline :
    1. clean_features   : codes spéciaux du questionnaire (7/9/77/99... -> NaN,
                          88/888 -> 0), recodage des réponses, résolution des
                          questions conditionnelles (skip patterns).
                          Les colonnes absentes de RULES sont supprimées
                          (administratif, poids, doublons, modules optionnels).
    2. fit_preprocessing : sur le TRAIN uniquement -> taux de NaN, bornes des outliers, médianes,
                          modes, catégories du one-hot, moyennes/écarts-types.
    3. transform        : applique ces paramètres au train ET au test :
                          écrêtage des outliers (percentiles 1-99), imputation,
                          one-hot, standardisation (moyenne 0, écart-type 1).

Utilisation :
    from helpers import load_csv_data
    from preprocessing import load_feature_names, preprocess

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")
    names = load_feature_names("data/dataset/x_train.csv")
    x_tr, x_te, out_names, params = preprocess(x_train, x_test, names)

Référence des codes : BRFSS 2015 Codebook (CDC).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Règles par variable
# ---------------------------------------------------------------------------
# type :
#   "num" : quantité continue            -> imputation médiane + standardisation
#   "ord" : catégories ordonnées         -> imputation médiane + standardisation
#   "bin" : oui/non recodé en 1/0        -> imputation par le mode
#   "nom" : catégories sans ordre        -> one-hot (NaN = sa propre catégorie)
# nan  : codes à transformer en NaN (ne sait pas / refus / manquant)
# zero : codes qui signifient "aucun" -> 0
# map  : recodage des valeurs restantes
# log  : applique log(1 + x) (variables très asymétriques)

YES_NO = {"type": "bin", "nan": [7, 9], "map": {1: 1, 2: 0}}

RULES = {
    # --- Santé générale -----------------------------------------------------
    "GENHLTH":  {"type": "ord", "nan": [7, 9]},                 # 1 excellent ... 5 mauvais
    "PHYSHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # jours mauvaise santé physique
    "MENTHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # jours mauvaise santé mentale
    "POORHLTH": {"type": "num", "nan": [77, 99], "zero": [88]},  # conditionnelle (voir plus bas)
    # --- Accès aux soins ----------------------------------------------------
    "HLTHPLN1": YES_NO,
    "PERSDOC2": {"type": "bin", "nan": [7, 9], "map": {1: 1, 2: 1, 3: 0}},  # a un médecin traitant
    "MEDCOST":  YES_NO,
    "CHECKUP1": {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 2, 3: 3, 4: 4, 8: 5}},  # 1 <1 an ... 4 5+ ans, 8 jamais -> 5
    # --- Hypertension / cholestérol ----------------------------------------
    "BPHIGH4":  {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 0, 3: 0, 4: 0.5}},  # 2 = grossesse seulement, 4 = limite
    "BPMEDS":   YES_NO,                                          # conditionnelle à BPHIGH4
    "BLOODCHO": YES_NO,
    "CHOLCHK":  {"type": "ord", "nan": [7, 9]},                  # conditionnelle à BLOODCHO
    "TOLDHI2":  YES_NO,                                          # conditionnelle à BLOODCHO
    # --- Maladies chroniques -----------------------------------------------
    "CVDSTRK3": YES_NO,
    "_ASTHMS1": {"type": "nom", "nan": [9]},                     # asthme actuel / ancien / jamais
    "CHCSCNCR": YES_NO,
    "CHCOCNCR": YES_NO,
    "CHCCOPD1": YES_NO,
    "HAVARTH3": YES_NO,
    "ADDEPEV2": YES_NO,
    "CHCKIDNY": YES_NO,
    "DIABETE3": {"type": "ord", "nan": [7, 9], "map": {1: 1, 2: 0, 3: 0, 4: 0.5}},  # 4 = prédiabète
    # --- Démographie ---------------------------------------------------------
    "SEX":      {"type": "bin", "nan": [], "map": {1: 1, 2: 0}},  # 1 = homme
    "_AGE80":   {"type": "num", "nan": []},                       # âge (plafonné à 80)
    "MARITAL":  {"type": "nom", "nan": [9]},
    "EDUCA":    {"type": "ord", "nan": [9]},
    "RENTHOM1": {"type": "nom", "nan": [7, 9]},
    "VETERAN3": YES_NO,
    "EMPLOY1":  {"type": "nom", "nan": [9]},                      # 8 = incapable de travailler
    "CHILDREN": {"type": "num", "nan": [77, 99], "zero": [88]},
    "INCOME2":  {"type": "ord", "nan": [77, 99]},
    "INTERNET": YES_NO,
    "_RACE":    {"type": "nom", "nan": [9]},
    # --- Corpulence (versions déjà converties par le CDC) --------------------
    "HTM4":     {"type": "num", "nan": []},                       # taille (m)
    "WTKG3":    {"type": "num", "nan": []},                       # poids (kg)
    "_BMI5":    {"type": "num", "nan": []},
    # --- Handicap -----------------------------------------------------------
    "QLACTLM2": YES_NO,
    "USEEQUIP": YES_NO,
    "BLIND":    YES_NO,
    "DECIDE":   YES_NO,
    "DIFFWALK": YES_NO,
    "DIFFDRES": YES_NO,
    "DIFFALON": YES_NO,
    # --- Tabac (variable calculée : résout le skip pattern SMOKE100 -> SMOKDAY2)
    "_SMOKER3": {"type": "ord", "nan": [9]},                      # 1 quotidien, 2 parfois, 3 ancien, 4 jamais
    "USENOW3":  {"type": "ord", "nan": [7, 9]},                   # tabac sans fumée
    # --- Alcool (variables calculées : les non-buveurs valent déjà 0) --------
    "DRNKANY5": YES_NO,
    "_DRNKWEK": {"type": "num", "nan": [99900], "log": True},     # verres / semaine
    "_RFBING5": {"type": "bin", "nan": [9], "map": {1: 0, 2: 1}},  # binge drinking
    "_RFDRHV5": {"type": "bin", "nan": [9], "map": {1: 0, 2: 1}},  # forte consommation
    # --- Alimentation (sommes calculées, fois / jour) ------------------------
    "_FRUTSUM": {"type": "num", "nan": []},
    "_VEGESUM": {"type": "num", "nan": []},
    # --- Activité physique ---------------------------------------------------
    "EXERANY2": YES_NO,
    "PA1MIN_":  {"type": "num", "nan": [], "log": True},          # minutes d'activité / semaine
    "PA1VIGM_": {"type": "num", "nan": [], "log": True},          # minutes vigoureuses / semaine
    "STRFREQ_": {"type": "num", "nan": [99]},                     # renforcement musculaire / semaine
    "_PACAT1":  {"type": "ord", "nan": [9]},                      # 1 très actif ... 4 inactif
    "_PAINDX1": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},
    "_PASTRNG": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},
    # --- Arthrite (calculées : 3 = pas d'arthrite, résout le skip pattern) --
    "_LMTACT1": {"type": "nom", "nan": [9]},
    "_LMTWRK1": {"type": "nom", "nan": [9]},
    "JOINPAIN": {"type": "num", "nan": [77, 99]},                 # douleur 0-10, conditionnelle
    # --- Prévention ----------------------------------------------------------
    "_RFSEAT2": {"type": "bin", "nan": [9], "map": {1: 1, 2: 0}},  # ceinture toujours
    "FLUSHOT6": YES_NO,
    "PNEUVAC3": YES_NO,
    "HIVTST6":  YES_NO,
}

# Variables nominales à haute cardinalité, désactivées par défaut.
OPTIONAL_RULES = {
    "_STATE": {"type": "nom", "nan": []},   # 53 États -> 53 colonnes one-hot
}

# FUITE DE LA CIBLE : ces questions ne sont posées qu'aux personnes ayant
# déjà eu une crise cardiaque (100 % de positifs parmi les répondants).
# Elles ne doivent JAMAIS être utilisées comme features.
LEAKAGE = ["HAREHAB1"]

# Pour le rapport : pourquoi les autres colonnes sont supprimées.
DROPPED_GROUPS = {
    "fuite de la cible": LEAKAGE,
    "administratif / identifiants": [
        "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
        "SEQNO", "_PSU", "QSTVER", "QSTLANG", "MSCODE"],
    "poids d'échantillonnage": [
        "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_CLLCPWT",
        "_DUALCOR", "_DUALUSE"],
    "logistique de l'appel (bloc fixe / portable)": [
        "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
        "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT",
        "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "NUMHHOL2",
        "NUMPHON2", "CPDEMO1"],
}


# ---------------------------------------------------------------------------
# 2. Utilitaires
# ---------------------------------------------------------------------------
def load_feature_names(x_train_path):
    """Noms des colonnes (sans 'Id'), dans le même ordre que load_csv_data."""
    with open(x_train_path, "r") as f:
        header = f.readline().strip().split(",")
    return header[1:]


def _col(x, names, name):
    return x[:, names.index(name)].astype(float)


# ---------------------------------------------------------------------------
# 3. Nettoyage (sans état : identique pour train et test)
# ---------------------------------------------------------------------------
def _apply_rule(raw, rule):
    col = raw.copy()
    for code in rule.get("nan", []):
        col[raw == code] = np.nan
    for code in rule.get("zero", []):
        col[raw == code] = 0.0
    if "map" in rule:
        # Les valeurs non listées dans "map" deviennent NaN
        mapped = np.full_like(col, np.nan)
        for old, new in rule["map"].items():
            mapped[raw == old] = new
        col = mapped
    if rule.get("log", False):
        col = np.log1p(np.maximum(col, 0))
    return col


def _apply_skip_patterns(cols, x, names):
    """
    Questions conditionnelles : la question n'est posée que si la question
    filtre a une certaine réponse. Si la question a été sautée, le NaN a un
    sens connu, donc on le remplace par la valeur logique.
    """
    def raw(n):
        return _col(x, names, n)

    def fill(child, mask, value):
        if child in cols:
            c = cols[child]
            c[np.isnan(c) & mask] = value

    # POORHLTH n'est posée que si PHYSHLTH > 0 ou MENTHLTH > 0
    fill("POORHLTH", (raw("PHYSHLTH") == 88) & (raw("MENTHLTH") == 88), 0.0)
    # BPMEDS n'est posée qu'aux hypertendus (BPHIGH4 == 1)
    fill("BPMEDS", np.isin(raw("BPHIGH4"), [2, 3, 4]), 0.0)
    # CHOLCHK / TOLDHI2 seulement si le cholestérol a déjà été mesuré
    fill("CHOLCHK", raw("BLOODCHO") == 2, 5.0)   # 5 = jamais mesuré
    fill("TOLDHI2", raw("BLOODCHO") == 2, 0.0)
    # JOINPAIN seulement en cas d'arthrite
    fill("JOINPAIN", raw("HAVARTH3") == 2, 0.0)
    # Minutes d'activité : pas de sport -> 0 minute
    no_exercise = raw("EXERANY2") == 2
    fill("PA1MIN_", no_exercise, 0.0)
    fill("PA1VIGM_", no_exercise, 0.0)
    fill("STRFREQ_", no_exercise, 0.0)
    return cols


def clean_features(x, names, use_state=False):
    """
    Applique les règles du codebook. Renvoie (x_clean, clean_names, types).
    Les colonnes sans règle sont supprimées.
    """
    rules = dict(RULES)
    if use_state:
        rules.update(OPTIONAL_RULES)

    leaks = [n for n in rules if n in LEAKAGE]
    if leaks:
        raise ValueError(f"Variables à fuite de cible dans RULES : {leaks}")

    cols = {}
    for name, rule in rules.items():
        if name not in names:
            continue
        cols[name] = _apply_rule(_col(x, names, name), rule)

    cols = _apply_skip_patterns(cols, x, names)

    clean_names = list(cols.keys())
    types = [rules[n]["type"] for n in clean_names]
    x_clean = np.column_stack([cols[n] for n in clean_names])
    return x_clean, clean_names, types


# ---------------------------------------------------------------------------
# 4. Fit sur le train / transform sur train et test
# ---------------------------------------------------------------------------
def fit_preprocessing(x_clean, names, types, nan_threshold=0.4,
                      add_nan_indicators=True, indicator_min_ratio=0.05,
                      clip_percentiles=(1, 99)):
    """Calcule tous les paramètres sur le TRAIN uniquement."""
    nan_ratio = np.mean(np.isnan(x_clean), axis=0)
    keep = nan_ratio <= nan_threshold

    params = {"names_in": names, "types": types, "keep": keep,
              "nan_ratio": nan_ratio, "fill": {}, "categories": {},
              "clip": {}, "indicators": [], "mean": None, "std": None}

    for j, (n, t) in enumerate(zip(names, types)):
        if not keep[j]:
            continue
        c = x_clean[:, j]
        valid = c[~np.isnan(c)]
        # Outliers : bornes d'écrêtage (winsorization) des variables continues
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

    # Moyenne / écart-type des colonnes continues et ordinales (après imputation)
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
            blocks.append(is_nan.astype(float))          # catégorie "manquant"
            out_names.append(f"{n}=NaN")
            scale.append(False)
        else:
            if n in params["clip"]:
                c = np.clip(c, *params["clip"][n])   # NaN restent NaN
            c[is_nan] = params["fill"][n]
            blocks.append(c)
            out_names.append(n)
            scale.append(t in ("num", "ord"))
            if n in params["indicators"]:
                blocks.append(is_nan.astype(float))
                out_names.append(f"{n}_was_nan")
                scale.append(False)
    return np.column_stack(blocks), out_names, np.array(scale)


def transform(x_clean, params):
    x_out, _, _ = _build_matrix(x_clean, params)
    m = params["scale_mask"]
    x_out[:, m] = (x_out[:, m] - params["mean"]) / params["std"]
    return x_out


# ---------------------------------------------------------------------------
# 5. Fonction tout-en-un
# ---------------------------------------------------------------------------
def preprocess(x_train, x_test, feature_names, nan_threshold=0.4,
               add_nan_indicators=True, use_state=False,
               clip_percentiles=(1, 99), verbose=True):
    """
    x_train, x_test : sorties de load_csv_data (sans la colonne Id).
    Renvoie x_train_proc, x_test_proc, output_names, params.
    """
    xtr_c, names_c, types = clean_features(x_train, feature_names, use_state)
    xte_c, _, _ = clean_features(x_test, feature_names, use_state)

    params = fit_preprocessing(xtr_c, names_c, types, nan_threshold,
                               add_nan_indicators,
                               clip_percentiles=clip_percentiles)
    x_tr = transform(xtr_c, params)
    x_te = transform(xte_c, params)

    if verbose:
        dropped = [n for n, k in zip(names_c, params["keep"]) if not k]
        print(f"Colonnes d'origine        : {len(feature_names)}")
        print(f"Après règles du codebook  : {len(names_c)}")
        print(f"Supprimées (NaN > {nan_threshold:.0%})  : {dropped if dropped else 'aucune'}")
        print(f"Features finales          : {x_tr.shape[1]} "
              f"(one-hot + indicateurs NaN inclus)")
    return x_tr, x_te, params["names_out"], params


def nan_report(x_clean, names, top=15):
    """Affiche les variables nettoyées avec le plus de NaN restants."""
    r = np.mean(np.isnan(x_clean), axis=0)
    for i in np.argsort(-r)[:top]:
        print(f"{names[i]:10s} {r[i]:6.1%}")


# ---------------------------------------------------------------------------
# 6. Split train / validation puis preprocessing (pour le main)
# ---------------------------------------------------------------------------
def split_data(x, y, ratio=0.8, seed=1, stratify=True):
    """
    Sépare x_train en un set d'entraînement et un set de validation.
    stratify=True garde la même proportion de malades (~9 %) dans les deux.
    Renvoie x_tr, x_va, y_tr, y_va.
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
                     clip_percentiles=(1, 99), verbose=True):
    """
    À appeler APRÈS le split. Toutes les statistiques (taux de NaN, bornes
    des outliers, médianes, moyennes/écarts-types de la standardisation)
    sont calculées sur x_tr uniquement, puis appliquées à x_va et x_te.
    x_va et x_te ne "voient" donc jamais leurs propres statistiques.

    Renvoie X_tr, X_va, X_te, output_names, params.
    """
    tr_c, names_c, types = clean_features(x_tr, feature_names, use_state)
    va_c, _, _ = clean_features(x_va, feature_names, use_state)
    te_c, _, _ = clean_features(x_te, feature_names, use_state)

    params = fit_preprocessing(tr_c, names_c, types, nan_threshold,
                               add_nan_indicators,
                               clip_percentiles=clip_percentiles)
    X_tr = transform(tr_c, params)
    X_va = transform(va_c, params)
    X_te = transform(te_c, params)

    if verbose:
        print(f"train {X_tr.shape} | validation {X_va.shape} | test {X_te.shape}")
    return X_tr, X_va, X_te, params["names_out"], params


def add_bias(*arrays):
    """Ajoute une colonne de 1 (terme constant w0) au début de chaque matrice."""
    out = tuple(np.c_[np.ones(len(a)), a] for a in arrays)
    return out if len(out) > 1 else out[0]
