import pandas as pd

# Leggi il file
#df = pd.read_csv("file.csv")  

# Se è Excel usa:
df = pd.read_excel(r"C:\Users\schia\Downloads\St07_D_CORRETTO.xlsx")

# Crea il token progressivo
df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum()
df
# Salva il risultato
df.to_excel(r"C:\Users\schia\Downloads\St07_D_CORRETTO.xlsx", index=False)

# Per Excel:
# df.to_excel("file_con_token.xlsx", index=False)

print(df)