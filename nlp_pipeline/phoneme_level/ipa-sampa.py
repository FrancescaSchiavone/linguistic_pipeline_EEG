#ADDING COLUMN FOR SAMPA
import pandas as pd

phonitalia = r"datasets\phonitalia (1).xlsx"
sampa_ipa = r'doc\SAMPA.xlsx'

df_phonitalia = pd.read_excel(phonitalia)
df_conv = pd.read_excel(sampa_ipa)

df_conv = df_conv.dropna(subset=["IPA", "SAMPA"])
df_conv["IPA"] = df_conv["IPA"].astype(str)
df_conv["SAMPA"] = df_conv["SAMPA"].astype(str)

ipa_to_sampa = dict(zip(df_conv["IPA"], df_conv["SAMPA"]))

ipa_keys = sorted(ipa_to_sampa.keys(), key=len, reverse=True)

def ipa_to_sampa_convert(ipa_text):
    if pd.isna(ipa_text):
        return ""

    result = str(ipa_text)

    for ipa_symbol in ipa_keys:
        sampa_symbol = ipa_to_sampa[ipa_symbol]
        result = result.replace(ipa_symbol, sampa_symbol)

    return result

df_phonitalia["SAMPA"] = df_phonitalia["IPA"].apply(ipa_to_sampa_convert)


output_file =r'C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\IMT\TESI\linguistic_pipeline_EEG\datasets\phonitalia.xlsx'
df_phonitalia.to_excel(output_file, index=False)

print(f"File salvato: {output_file}")

#PHONITALIA MISSING WORDS
import pandas as pd
import os

phonitalia = r'C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\IMT\TESI\linguistic_pipeline_EEG\doc\phonitalia.csv'

df_phon = pd.read_csv(phonitalia, sep=';')
parole_db = set(df_phon["WORD"].astype(str).str.lower())

numbers = range(1, 11)  

missing_words = []

for n in numbers:

    codice = f"{n:02d}"

    text_path = rf'output_nlp\output_B\St{codice}_B\St{codice}_B.csv'

    print(f"\nAnalizzo: {text_path}")

    df_text = pd.read_csv(text_path, sep=';')

    # ===== TOKEN =====
    word_text = (
        df_text["tokens_no_punct"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    non_match = word_text[~word_text.isin(parole_db)]

    # aggiungi alla lista totale
    missing_words.extend(non_match.tolist())

    print("Totale token:", len(word_text))
    print("Totale non match:", len(non_match))

df_missing = pd.DataFrame({
    "missing_word": missing_words
})

# rimuove duplicati
df_missing = df_missing.drop_duplicates()

# ===== SALVA CSV =====
output_csv = "missing_words2.csv"
df_missing.to_csv(output_csv, sep=';', index=False)

print(f"\nFile salvato: {output_csv}")
print("Totale missing words uniche:", len(df_missing))