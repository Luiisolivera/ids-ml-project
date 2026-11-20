"""Predicción offline desde JSON con features; no ajusta ningún modelo."""
import argparse
import json
from pathlib import Path

from ids_ml.artifacts import load_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=Path('models/final'))
    ap.add_argument('--input', type=Path, default=Path('models/final/example_input.json'))
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    result = load_pipeline(args.model).predict(json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
