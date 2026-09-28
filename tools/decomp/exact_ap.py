"""Thin re-export of evaluator/exact_ap.py, so the scripts in this directory can keep
`from exact_ap import ap_summaries`. Writes no report.
Validation gates: python evaluator/exact_ap.py [--full]  (or this file, which forwards to it).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evaluator.exact_ap import *  # noqa: F401,F403
from evaluator.exact_ap import (  # noqa: F401  (private helpers, re-exported explicitly)
    _pr_env_at_thresholds, _all_scores, _pad_env, _toy_annos)

if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "evaluator", "exact_ap.py"), run_name="__main__")
