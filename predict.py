#!/usr/bin/env python
"""CLI: score a single student for early at-risk status and explain why.

Examples
--------
    python predict.py --student-id 605939
    python predict.py --student-id 605939 --module GGG --presentation 2013J
    python predict.py --list-high 10        # show 10 high-risk students to try
    python predict.py --student-id 605939 --calibrated

Run from the project root with the project venv active.
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.features import load_features
from src.model import (
    add_risk_columns,
    find_student,
    load_model,
    top_shap_drivers,
)


def _print_enrollment(shap_model, row, top_n: int) -> None:
    enroll = row.iloc[0]
    prob = float(enroll["risk_probability"])
    tier = enroll["risk_tier"]
    bar = "#" * int(round(prob * 30))

    print("-" * 60)
    print(f"  {enroll['code_module']} / {enroll['code_presentation']}  "
          f"(student {enroll['id_student']})")
    print(f"  Risk probability : {prob:.1%}  [{bar:<30}]")
    print(f"  Risk tier        : {tier}")
    print(f"  Actual outcome   : {enroll.get('final_result', 'n/a')}")
    print()

    try:
        drivers = top_shap_drivers(shap_model, row, top_n=top_n)
        print(f"  Top {top_n} drivers (SHAP):")
        for _, d in drivers.iterrows():
            arrow = "^" if d["shap_value"] >= 0 else "v"
            print(f"    {arrow} {d['feature']:<32} {d['shap_value']:+.4f}  ({d['direction']})")
    except Exception as exc:  # SHAP optional / model may not be a tree
        print(f"  (SHAP explanation unavailable: {exc})")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a student for at-risk likelihood.")
    parser.add_argument("--student-id", type=int, help="OULAD id_student to score")
    parser.add_argument("--module", type=str, default=None, help="code_module filter (e.g. GGG)")
    parser.add_argument("--presentation", type=str, default=None, help="code_presentation filter (e.g. 2013J)")
    parser.add_argument("--top-n", type=int, default=6, help="number of SHAP drivers to show")
    parser.add_argument("--calibrated", action="store_true", help="use the calibrated model if present")
    parser.add_argument("--list-high", type=int, metavar="N", help="list N high-risk students and exit")
    args = parser.parse_args(argv)

    df = load_features()
    model_path = config.CALIBRATED_MODEL_PATH if args.calibrated else config.DEFAULT_MODEL_PATH
    model = load_model(model_path)
    scored = add_risk_columns(model, df)

    # SHAP needs the raw tree pipeline; the calibrated wrapper hides named_steps.
    shap_model = load_model(config.DEFAULT_MODEL_PATH) if args.calibrated else model

    if args.list_high is not None:
        top = (scored.sort_values("risk_probability", ascending=False)
               .head(args.list_high))
        print(f"Top {args.list_high} high-risk students (model: {model_path.name}):")
        for _, r in top.iterrows():
            print(f"  id={r['id_student']:>8}  {r['code_module']}/{r['code_presentation']}  "
                  f"risk={r['risk_probability']:.1%}  ({r['final_result']})")
        return 0

    if args.student_id is None:
        parser.error("provide --student-id, or use --list-high N to find ids")

    matches = find_student(scored, args.student_id, args.module, args.presentation)
    if matches.empty:
        print(f"No student found with id={args.student_id}"
              f"{f' in {args.module}/{args.presentation}' if args.module else ''}.")
        print("Tip: run  python predict.py --list-high 10  to see valid ids.")
        return 1

    print(f"\nModel: {model_path.name}   |   Enrollments found: {len(matches)}")
    for _, row in matches.groupby(config.ID_COLS, sort=False):
        _print_enrollment(shap_model, row, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
