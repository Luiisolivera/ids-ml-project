"""Única validación de features, compartida por preparación e inferencia."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["proto", "service", "state"]
FEATURES = """dur proto service state spkts dpkts sbytes dbytes rate sttl dttl
sload dload sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt
synack ackdat smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl
ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login
ct_ftp_cmd ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports""".split()
CONTINUOUS = "dur rate sload dload sinpkt dinpkt sjit djit tcprtt synack ackdat".split()
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
INTEGER = [name for name in NUMERIC if name not in CONTINUOUS]
ATTACKS = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
           "Reconnaissance", "Shellcode", "Worms"]
CLASSES = ["Normal", *ATTACKS]


def validate_features(data) -> pd.DataFrame:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, (pd.DataFrame, list)):
        raise ValueError("Se requiere DataFrame, diccionario o lista de diccionarios")
    if isinstance(data, list) and not all(isinstance(row, dict) for row in data):
        raise ValueError("Cada fila debe ser un diccionario con nombres de features")
    frame = pd.DataFrame(data).copy()
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError("Se requieren filas no vacías y nombres de columnas únicos")
    missing, extra = set(FEATURES) - set(frame), set(frame) - set(FEATURES)
    if missing or extra:
        raise ValueError(f"Contrato inválido: faltan={sorted(missing)}, sobran={sorted(extra)}")
    frame = frame[FEATURES]
    if frame.isna().any().any():
        raise ValueError("No se permiten nulos; no se imputan features desconocidas")
    for name in CATEGORICAL:
        if not frame[name].map(lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all():
            raise ValueError(f"{name}: se requiere texto no vacío sin espacios exteriores")
    for name in NUMERIC:
        if pd.api.types.is_bool_dtype(frame[name]) or not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"{name}: se requieren números, no texto ni booleanos")
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}: se requieren números finitos no negativos")
        if name in INTEGER and (values != np.floor(values)).any():
            raise ValueError(f"{name}: se requieren valores enteros")
        if name == "is_sm_ips_ports" and not np.isin(values, [0, 1]).all():
            raise ValueError("is_sm_ips_ports debe ser 0 o 1")
    return frame


def feature_contract(preprocessor, train: pd.DataFrame) -> dict:
    encoder = preprocessor.named_transformers_["categorical"]
    categories = dict(zip(CATEGORICAL, [v.tolist() for v in encoder.categories_]))
    return {
        "schema_version": 1,
        "input": "Una fila por flujo preparado según UNSW-NB15; no paquetes crudos",
        "features": [{
            "name": name, "position": i, "required": True,
            "type": "string" if name in CATEGORICAL else ("integer" if name in INTEGER else "number"),
            **({"known_categories": categories[name], "unknown": "one-hot all zeros; reported in output"}
               if name in CATEGORICAL else {
                   "minimum": 0,
                   "observed_train_min": float(train[name].min()),
                   "observed_train_max": float(train[name].max()),
                   "maximum_is_constraint": False,
               }),
        } for i, name in enumerate(FEATURES)],
        "extra_fields": "rejected, including id, label and attack_cat",
        "nulls": "rejected", "column_order": "aligned by name internally",
        "transformations": "one-hot categorical, numeric passthrough; no scaling, clipping or imputation",
        "classes": CLASSES,
        "output": {"label": "0 normal / 1 attack", "attack_cat": CLASSES,
                   "probability_attack": "RF score [0,1], not calibrated",
                   "unknown_categories": "list of categorical feature names unknown at fit"},
    }
