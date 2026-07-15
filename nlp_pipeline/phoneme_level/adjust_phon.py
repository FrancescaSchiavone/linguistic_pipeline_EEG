import pandas as pd

# Leggi il file
#df = pd.read_csv("file.csv")  

# Se è Excel usa:
df = pd.read_excel(r"C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\IMT\TESI\linguistic_pipeline_EEG\data\phoneme_onset_D\St08_D.xlsx")

# Crea il token progressivo
df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum()
df
# Salva il risultato
df.to_excel(r"C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\IMT\TESI\linguistic_pipeline_EEG\data\phoneme_onset_D\St08_D.xlsx", index=False)

# Per Excel:
# df.to_excel("file_con_token.xlsx", index=False)

print(df)