"""
Master pipeline. Runs every phase in order with audit logging.
Usage: python scripts/run_all.py
"""
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.audit import setup_audit, log_warning


def run_phase(label, fn):
    print(f"\n[{label}]")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s: {e}")
        log_warning(f"{label} failed: {e}")
        traceback.print_exc()
        return False


def phase1_data():
    import subprocess
    subprocess.run([sys.executable, "data/fetch_data.py"], check=True)


def phase2_deterministic():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.deterministic_model"], check=True)


def phase3_forecast():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.demand_forecasting"], check=True)


def phase4_stochastic():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.stochastic_model"], check=True)


def phase5_decision_focused():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.decision_focused"], check=True)


def phase6_sensitivity():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.sensitivity"], check=True)


def phase7_visualizations():
    import subprocess
    subprocess.run([sys.executable, "-m", "src.visualization"], check=True)


def main():
    setup_audit(".")
    print("=" * 60)
    print("PUNE BUS OPTIMIZATION - FULL PIPELINE")
    print("=" * 60)

    phases = [
        ("1/7 data",                    phase1_data),
        ("2/7 deterministic MIP",       phase2_deterministic),
        ("3/7 demand forecasting",      phase3_forecast),
        ("4/7 stochastic and robust",   phase4_stochastic),
        ("5/7 decision-focused",        phase5_decision_focused),
        ("6/7 sensitivity",             phase6_sensitivity),
        ("7/7 visualizations",          phase7_visualizations),
    ]
    failures = []
    for label, fn in phases:
        ok = run_phase(label, fn)
        if not ok:
            failures.append(label)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED PHASES: {failures}")
        print("See audit.log for details")
        sys.exit(1)
    print("ALL PHASES COMPLETE")
    print("See audit.log for full trace")
    print("Run 'streamlit run dashboard/app.py' for interactive dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
