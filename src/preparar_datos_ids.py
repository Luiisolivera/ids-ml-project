"""Prepara UNSW-NB15 para un IDS en dos etapas.

Uso desde la carpeta del proyecto:
    python src/preparar_datos_ids.py

La prueba oficial se transforma con el preprocesador ajustado solo con
entrenamiento. No se aplica sobremuestreo ni se entrena ningún clasificador.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from ids_ml.contract import FEATURES, CATEGORICAL, NUMERIC, CLASSES, validate_features, feature_contract


REQUIRED = {"id", "label", "attack_cat"}
RANDOM_STATE = 42


def feature_signatures(frame: pd.DataFrame) -> np.ndarray:
    """Stable row keys for detecting identical model inputs (excluding labels/id)."""
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")


def duplicate_summary(frame: pd.DataFrame, signatures: np.ndarray) -> dict:
    grouped = pd.DataFrame(
        {
            "signature": signatures,
            "label": frame["label"].astype(int).to_numpy(),
            "attack_cat": frame["attack_cat"].astype(str).to_numpy(),
        }
    ).groupby("signature", sort=False)
    binary_conflicts = grouped["label"].nunique()
    category_conflicts = grouped["attack_cat"].nunique()
    binary_conflict_groups = set(binary_conflicts[binary_conflicts > 1].index)
    category_conflict_groups = set(category_conflicts[category_conflicts > 1].index)
    return {
        "filas": int(len(frame)),
        "grupos_caracteristicas": int(len(grouped)),
        "filas_repetidas_por_caracteristicas": int(len(frame) - len(grouped)),
        "grupos_con_label_binario_conflictivo": int(len(binary_conflict_groups)),
        "filas_en_grupos_con_label_binario_conflictivo": int(
            np.isin(signatures, list(binary_conflict_groups)).sum()
        ),
        "grupos_con_attack_cat_multiple": int(len(category_conflict_groups)),
        "filas_en_grupos_con_attack_cat_multiple": int(
            np.isin(signatures, list(category_conflict_groups)).sum()
        ),
    }


def read_and_check(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {sorted(missing)}")
    if df["label"].isna().any() or df["attack_cat"].isna().any():
        raise ValueError(f"{name}: hay etiquetas vacías")
    if not df["label"].isin([0, 1]).all() or not df["attack_cat"].isin(CLASSES).all():
        raise ValueError(f"{name}: etiquetas fuera del contrato")
    validate_features(df.drop(columns=["id", "label", "attack_cat"]))
    inconsistent = ((df["label"] == 0) != (df["attack_cat"] == "Normal"))
    if inconsistent.any():
        raise ValueError(f"{name}: label y attack_cat no coinciden en {int(inconsistent.sum())} filas")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/UNSW_NB15_training-set.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/UNSW_NB15_testing-set.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    train = read_and_check(args.train, "Entrenamiento")
    test = read_and_check(args.test, "Prueba")
    train_original = train.copy()
    if set(train.columns) != set(test.columns):
        raise ValueError("Los archivos de entrenamiento y prueba no tienen las mismas columnas")
    # Alinear el orden de las columnas de prueba con el archivo de entrenamiento.
    test = test[train.columns]

    # id identifica una fila, no describe el tráfico; se excluye de las entradas.
    feature_cols = FEATURES
    X = validate_features(train[feature_cols])
    X_test = validate_features(test[feature_cols])

    # Evitar que una fila del conjunto oficial de entrenamiento comparta
    # exactamente las mismas entradas que una fila del test reservado.
    train_signatures = feature_signatures(X)
    test_signatures = feature_signatures(X_test)
    test_signature_set = set(test_signatures.tolist())
    overlap_mask = np.isin(train_signatures, np.fromiter(test_signature_set, dtype="uint64"))
    overlap_rows = int(overlap_mask.sum())
    overlap_groups = int(pd.Series(train_signatures[overlap_mask]).nunique())

    train = train.loc[~overlap_mask].reset_index(drop=True)
    X = X.loc[~overlap_mask].reset_index(drop=True)
    y = train[["label", "attack_cat"]].reset_index(drop=True)
    train_signatures = train_signatures[~overlap_mask]
    if len(train) == 0:
        raise ValueError("No quedan filas de entrenamiento después de retirar el solapamiento con test")

    # Mantener todas las filas con las mismas entradas dentro de una sola
    # partición. La estratificación sigue usando la categoría de ataque.
    grouped_split = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    idx_train, idx_val = next(
        grouped_split.split(X, y["attack_cat"], groups=train_signatures)
    )
    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]
    train_groups = train_signatures[idx_train]
    val_groups = train_signatures[idx_val]

    if set(train_groups) & set(val_groups):
        raise RuntimeError("Una firma de características quedó en train y validación")
    if set(train_groups) & test_signature_set or set(val_groups) & test_signature_set:
        raise RuntimeError("Una firma de características quedó en train/validación y test")
    y_test = test[["label", "attack_cat"]].reset_index(drop=True)

    cat_cols = CATEGORICAL
    num_cols = NUMERIC
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("numeric", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # fit_transform aprende categorías solo del entrenamiento; validación y prueba
    # reciben transform(), evitando que influyan en el preprocesamiento.
    matrices = {
        "X_train.npz": preprocessor.fit_transform(X_train),
        "X_val.npz": preprocessor.transform(X_val),
        "X_test.npz": preprocessor.transform(X_test),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, matrix in matrices.items():
        save_npz(args.out / filename, csr_matrix(matrix))

    # Guardar entradas crudas permite evaluar la interfaz pública tras recargar.
    for name, indices in [("train", idx_train), ("val", idx_val)]:
        train.iloc[indices].to_csv(args.out / f"raw_{name}.csv", index=False)
    test.to_csv(args.out / "raw_test.csv", index=False)
    pd.DataFrame({"source_id": train["id"], "signature": train_signatures,
                  "partition": np.where(np.isin(np.arange(len(train)), idx_train), "train", "val")}).to_csv(
                      args.out / "split_manifest.csv", index=False)

    y_train.to_csv(args.out / "y_train.csv", index=False)
    y_val.to_csv(args.out / "y_val.csv", index=False)
    y_test.to_csv(args.out / "y_test.csv", index=False)
    joblib.dump(preprocessor, args.out / "preprocesador.joblib")

    feature_names = preprocessor.get_feature_names_out().tolist()
    (args.out / "columnas_transformadas.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contract = feature_contract(preprocessor, X_train)
    (args.out / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    train.iloc[idx_val[:3]][FEATURES].to_json(args.out / "example_input.json", orient="records", indent=2)
    report = {
        "archivos": {"train": str(args.train), "test": str(args.test)},
        "filas": {
            "train_oficial_original": int(len(train_original)),
            "train_retiradas_por_solapamiento_con_test": overlap_rows,
            "train_restantes": int(len(train)),
            "train_modelo": len(idx_train),
            "validacion": len(idx_val),
            "prueba_oficial_sin_modificar": len(test),
        },
        "grupos_caracteristicas_train_retirados_por_solapamiento": overlap_groups,
        "duplicados_train_original": duplicate_summary(
            train_original, feature_signatures(train_original[feature_cols])
        ),
        "duplicados_train_despues_de_purgar_test": duplicate_summary(
            train, train_signatures
        ),
        "duplicados_test_oficial": duplicate_summary(test, test_signatures),
        "grupos_caracteristicas_por_particion": {
            "train": int(pd.Series(train_groups).nunique()),
            "validacion": int(pd.Series(val_groups).nunique()),
            "test": int(pd.Series(test_signatures).nunique()),
        },
        "sin_solapamiento_de_caracteristicas": True,
        "caracteristicas_originales": feature_cols,
        "columnas_categoricas": cat_cols,
        "columnas_numericas": num_cols,
        "columnas_transformadas": len(feature_names),
        "seed": RANDOM_STATE,
        "split": "first fold StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "raw_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train, args.test]},
        "calidad": {
            "nulos_train": int(train_original.isna().sum().sum()),
            "nulos_test": int(test.isna().sum().sum()),
            "infinitos": 0,
            "tipos_originales": {c: str(t) for c, t in train_original.dtypes.items()},
            "cuantiles_numericos_train": X_train[NUMERIC].quantile([0, .01, .5, .99, 1]).to_dict(),
            "outliers": "No se recortan extremos: pueden representar ataques; estadísticas solo de train.",
            "categorias_desconocidas": {name: {c: int((~frame[c].isin(X_train[c].unique())).sum())
                for c in CATEGORICAL} for name, frame in [("val", X_val), ("test", X_test)]},
        },
        "clases_entrenamiento": y_train["attack_cat"].value_counts().to_dict(),
        "clases_validacion": y_val["attack_cat"].value_counts().to_dict(),
        "clases_prueba": y_test["attack_cat"].value_counts().to_dict(),
        "nota": (
            "Las filas del train oficial cuyas características aparecen en test se excluyen. "
            "El train restante se divide por grupos de características, de modo que filas "
            "idénticas no crucen train/validación. Test se conserva completo; sus etiquetas "
            "no se usan para ajustar el preprocesador ni los modelos."
        ),
    }
    (args.out / "resumen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.out.iterdir() if p.is_file() and p.name != "manifest.json"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nArchivos preparados en: {args.out.resolve()}")


if __name__ == "__main__":
    main()
