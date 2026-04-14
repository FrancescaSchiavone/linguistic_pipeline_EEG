import os
import numpy as np
import pandas as pd
import pickle

# =====================
# PARAMETRI
# =====================
fs = 100
trial_sec = 60
trial_len = trial_sec * fs     # 6000
trials_per_story = 3

# percorsi
excel_orders = r"doc\stories_order_D(A)_CI.xlsx"
PRED_ROOT = r"data\Predictors_D"
OUT_ROOT  = r"data\SubjectPredictors_D_CI"

metrics = ["Dissimilarity", "Entropy", "Surprisal", "WordFreq"]

# =====================
# LEGGI EXCEL ORDINI
# =====================
df = pd.read_excel(excel_orders)
print(df[["Subj", "NamestoryA"]])

os.makedirs(OUT_ROOT, exist_ok=True)

# =====================
# CICLO SOGGETTI
# =====================
for _, row in df.iterrows():
    subj = int(row["Subj"])
    stories_str = str(row["NamestoryA"]).strip()
    story_list = stories_str.split()

    subj_folder = os.path.join(OUT_ROOT, f"Subject{subj:02d}_D")
    os.makedirs(subj_folder, exist_ok=True)

    for metric in metrics:
        pred_total = np.array([], dtype=float)

        for st in story_list:
            in_file = os.path.join(PRED_ROOT, st, f"{metric}_{st}_pred.csv")
            pred = pd.read_csv(in_file, header=None).values.flatten()

            pred_total = np.concatenate((pred_total, pred))

        # salva CSV concatenato per soggetto+metrica
        out_csv = os.path.join(subj_folder, f"{metric}_Subject{subj:02d}_D_pred.csv")
        np.savetxt(out_csv, pred_total, delimiter=",")
        print("saved:", out_csv)

        # salva anche in trials
        num_trials = len(story_list) * trials_per_story
        trials = []
        start_idx = 0

        for i in range(num_trials):
            end_idx = start_idx + trial_len
            if end_idx <= len(pred_total):
                trial = pred_total[start_idx:end_idx]
            else:
                trial = np.zeros(trial_len)
                remaining = len(pred_total) - start_idx
                if remaining > 0:
                    trial[:remaining] = pred_total[start_idx:]
            trials.append(trial)
            start_idx = end_idx

        trials_array = np.stack(trials)

        out_pkl = os.path.join(subj_folder, f"trials_Subject{subj:02d}_D_{metric}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump({"trials": trials_array, "fs": fs}, f)
        print("saved:", out_pkl)
