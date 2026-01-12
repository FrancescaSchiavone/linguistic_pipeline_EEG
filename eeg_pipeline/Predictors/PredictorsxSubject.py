import numpy as np
import pandas as pd
import pickle
#per generare un nuovo predittore cambiare nomi variabli dove c'è 🫖
# --- Parametri ---
fs = 100 # Hz
trial_sec = 60 # secondi per trial
trial_len = trial_sec * fs # 6000 campioni per trial
num_trials = 15
# Lista dei nell'ordine ascoltato dal singolo soggetto
pred_files = ["diss2C.csv", "diss9C.csv", "diss4C.csv", "diss6C.csv", "diss5C.csv"] #🫖
# Taglia ogni storia ai primi 3 trial e concatena
pred_total = np.array([])
for f in pred_files:
    pred = pd.read_csv(f).values.flatten()
    max_len = trial_len * 3 # 3 trial per storia
    if len(pred) > max_len:
        pred = pred[:max_len]
    pred_total = np.concatenate((pred_total, pred))

#  Creazione dei 15 trial da 60s ciascuno 
trials = []
start_idx = 0
for i in range(num_trials):
    end_idx = start_idx + trial_len
    if end_idx <= len(pred_total):
        trial = pred_total[start_idx:end_idx]
        trials.append(trial)
        start_idx = end_idx
    else:
 # Se non ci sono abbastanza campioni, riempi con zeri
        trial = np.zeros(trial_len)
        remaining = len(pred_total) - start_idx
        if remaining > 0:
            trial[:remaining] = pred_total[start_idx:]
        trials.append(trial)
        start_idx = len(pred_total)

#  Converti in array 2D (trial × campioni così da poter essere usati in Eelbrain)
trials_array = np.stack(trials) # shape = (15, 6000)

# Salva in un unico CSV 
#pd.DataFrame(trials_array).to_csv("trials_Subject19_WOS.csv", index=False) #🫖

# Salva in pickle
with open("trials_Subject19_DissC.pkl", "wb") as f: #🫖
    pickle.dump({"trials": trials_array, "fs": fs}, f)