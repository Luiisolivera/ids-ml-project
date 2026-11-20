"""Simulador de tráfico para demostraciones del IDS.

Toma flujos reales etiquetados del test oficial de UNSW-NB15, les retira `id`,
`label` y `attack_cat`, y los envía uno a uno a `POST /predict` del backend en
ejecución, como si llegaran de la red. Muestra en consola la etiqueta real frente
a la predicción y un resumen final. Las alertas aparecen solas en el dashboard.

Aviso: el test oficial fue consultado históricamente para decidir el umbral, por
lo que estos resultados son demostrativos/diagnósticos, no una evaluación
independiente. La muestra es aleatoria (con semilla) e incluye los errores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from ids_ml.contract import ATTACKS, CONTINUOUS, FEATURES, INTEGER

RED, GREEN, YELLOW, BOLD, RESET = "\033[31m", "\033[32m", "\033[33m", "\033[1m", "\033[0m"


def load_sample(csv_path: Path, per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    parts = []
    for cls in ["Normal", *ATTACKS]:
        subset = df[df["attack_cat"] == cls]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(per_class, len(subset)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def to_payload(row: pd.Series) -> dict:
    """Convierte una fila a los tipos JSON exigidos por el contrato (42 features)."""
    out = {}
    for name in FEATURES:
        value = row[name]
        if name in INTEGER:
            out[name] = int(value)
        elif name in CONTINUOUS:
            out[name] = float(value)
        else:
            out[name] = str(value)
    return out


def predict(url: str, payload: dict) -> dict:
    req = Request(
        f"{url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="URL del backend")
    ap.add_argument("--csv", type=Path, default=Path("data/raw/UNSW_NB15_testing-set.csv"))
    ap.add_argument("--per-class", type=int, default=3, help="Flujos por clase (10 clases)")
    ap.add_argument("--delay", type=float, default=1.5, help="Segundos entre flujos")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sample = load_sample(args.csv, args.per_class, args.seed)
    total = len(sample)
    print(f"{BOLD}Simulando {total} flujos reales de UNSW-NB15 hacia {args.url}{RESET}\n")

    detect_ok = class_ok = attacks = false_alarms = normals = 0
    for i, row in sample.iterrows():
        real_cat = str(row["attack_cat"])
        real_attack = int(row["label"]) == 1
        try:
            pred = predict(args.url, to_payload(row))["predictions"][0]
        except HTTPError as exc:
            print(f"{RED}Error HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}{RESET}")
            return 1
        except URLError as exc:
            print(f"{RED}No se pudo conectar con el backend ({exc.reason}). ¿Está en ejecución?{RESET}")
            return 1

        pred_attack = int(pred["label"]) == 1
        pred_cat = pred["attack_cat"]
        if real_attack:
            attacks += 1
            detect_ok += pred_attack
            class_ok += pred_cat == real_cat
        else:
            normals += 1
            false_alarms += pred_attack

        if pred_cat == real_cat:
            mark = f"{GREEN}✔ correcto{RESET}"
        elif real_attack and pred_attack:
            mark = f"{YELLOW}~ detectado, tipo distinto{RESET}"
        elif real_attack:
            mark = f"{RED}✘ ataque no detectado{RESET}"
        else:
            mark = f"{RED}✘ falsa alarma{RESET}"

        verdict = f"{RED}ALERTA {pred_cat}{RESET}" if pred_attack else f"{GREEN}Normal{RESET}"
        print(f"[{i + 1:>2}/{total}] {row['proto']:>5}/{row['service']:<6} "
              f"Real: {real_cat:<15} → IDS: {verdict:<26} {mark}")
        time.sleep(args.delay)

    print(f"\n{BOLD}Resumen{RESET}")
    if attacks:
        print(f"  Ataques detectados:      {detect_ok}/{attacks} ({detect_ok / attacks:.0%})")
        print(f"  Tipo de ataque correcto: {class_ok}/{attacks} ({class_ok / attacks:.0%})")
    if normals:
        print(f"  Falsas alarmas (Normal): {false_alarms}/{normals}")
    print("  Nota: muestra pequeña y test históricamente expuesto; resultado demostrativo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
