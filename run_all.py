"""
run_all.py - run the full pipeline end to end
"""
import subprocess, sys

steps = [
    ("Generating dataset...",  ["python", "generate_dataset.py"]),
    ("Preprocessing...",       ["python", "preprocess.py"]),
    ("Training model...",      ["python", "train_model.py"]),
    ("Evaluating...",          ["python", "evaluate.py"]),
]

for msg, cmd in steps:
    print(f"\n>>> {msg}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Pipeline failed."); sys.exit(1)

print("\nDone! Check loss_curve.png and pred_vs_actual.png")
