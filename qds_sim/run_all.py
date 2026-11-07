"""Run every experiment in order.

    python run_all.py            # full runs (a few minutes)
    python run_all.py --quick    # fast check (under a minute)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = ["e0_reanalysis.py", "e1_zeno_schedule.py", "e2_reminders.py", "e3_delay_qq.py",
           "e4_feedback.py", "e5_model_recovery.py", "e6_agent_memory.py", "e7_token_benchmark.py", "e8_orchestrated.py", "e9_hierarchy.py", "e10_noise_stability.py", "e11_accuracy_eval.py"]


def main():
    extra = sys.argv[1:]
    print("== tests ==")
    for t in ["test_core.py", "test_pipeline.py", "test_hierarchy.py", "test_e11.py"]:
        if subprocess.run([sys.executable, str(HERE / "tests" / t)]).returncode:
            sys.exit(f"{t} failed; fix before running experiments")
    for s in SCRIPTS:
        print(f"\n== {s} ==")
        t0 = time.time()
        r = subprocess.run([sys.executable, s, *extra], cwd=HERE / "experiments")
        if r.returncode:
            sys.exit(f"{s} failed")
        print(f"({time.time() - t0:.0f}s)")
    print("\nAll results are in ./results/")


if __name__ == "__main__":
    main()
