import pandas as pd
from collections import defaultdict, Counter


CSV_FILE = "datasets\\phonitalia.csv"

df = pd.read_csv(CSV_FILE, sep=';')

# ============================================================
# 2. TOKENIZZAZIONE SAMPA
# ============================================================

# Assumiamo:
# colonna = "SAMPA"
# fonemi separati da spazio
#
# esempio:
# "k o n t a d i n o"

def tokenize(s):
    return str(s).strip().split()


corpus = []

for s in df["SAMPA"].dropna():

    phonemes = tokenize(s)

    if len(phonemes) > 0:
        corpus.append(phonemes)

# ============================================================
# 3. BIGRAMMI
# ============================================================

bigram_counts = defaultdict(Counter)

for seq in corpus:

    for i in range(len(seq) - 1):

        current_ph = seq[i]
        next_ph = seq[i + 1]

        bigram_counts[current_ph][next_ph] += 1

# ============================================================
# 4. TRIGRAMMI
# ============================================================

trigram_counts = defaultdict(Counter)

for seq in corpus:

    for i in range(len(seq) - 2):

        ph1 = seq[i]
        ph2 = seq[i + 1]
        ph3 = seq[i + 2]

        trigram_counts[(ph1, ph2)][ph3] += 1

# ============================================================
# 5. PROBABILITÀ BIGRAMMI
# ============================================================

bigram_probs = defaultdict(dict)

for current_ph in bigram_counts:

    total = sum(bigram_counts[current_ph].values())

    for next_ph in bigram_counts[current_ph]:

        prob = (
            bigram_counts[current_ph][next_ph]
            / total
        )

        bigram_probs[current_ph][next_ph] = prob

# ============================================================
# 6. PROBABILITÀ TRIGRAMMI
# ============================================================

trigram_probs = defaultdict(dict)

for context in trigram_counts:

    total = sum(trigram_counts[context].values())

    for next_ph in trigram_counts[context]:

        prob = (
            trigram_counts[context][next_ph]
            / total
        )

        trigram_probs[context][next_ph] = prob

# ============================================================
# 7. FUNZIONI DI QUERY
# ============================================================

def bigram_probability(previous_phoneme,
                        current_phoneme):

    if previous_phoneme in bigram_probs:

        if current_phoneme in bigram_probs[previous_phoneme]:

            return bigram_probs[previous_phoneme][current_phoneme]

    return 0.0


def trigram_probability(ph1,
                        ph2,
                        ph3):

    context = (ph1, ph2)

    if context in trigram_probs:

        if ph3 in trigram_probs[context]:

            return trigram_probs[context][ph3]

    return 0.0

# ============================================================
# SALVA MATRICE BIGRAMMI
# ============================================================

import pandas as pd

all_phonemes = sorted(bigram_probs.keys())

bigram_matrix = pd.DataFrame(
    index=all_phonemes,
    columns=all_phonemes
).fillna(0)

for ph1 in bigram_probs:

    for ph2 in bigram_probs[ph1]:

        bigram_matrix.loc[ph1, ph2] = (
            bigram_probs[ph1][ph2]
        )

bigram_matrix.to_csv(
    "bigram_matrix.csv"
)

print("\nBigram matrix salvata.")


# ============================================================
# SALVA TRIGRAMMI
# ============================================================

trigram_rows = []

for context in trigram_probs:

    ph1, ph2 = context

    for ph3 in trigram_probs[context]:

        trigram_rows.append({

            "ph1": ph1,
            "ph2": ph2,
            "ph3": ph3,

            "probability":
            trigram_probs[context][ph3]
        })

trigram_df = pd.DataFrame(trigram_rows)

trigram_df.to_csv(
    "trigram_probabilities.csv",
    index=False
)

print("Trigram probabilities salvate.")


import matplotlib.pyplot as plt

# ============================================================
# HEATMAP BIGRAMMI
# ============================================================

plt.figure(figsize=(12, 10))

plt.imshow(
    bigram_matrix,
    aspect='auto'
)

plt.colorbar(label="Probability")

plt.xticks(
    range(len(bigram_matrix.columns)),
    bigram_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(bigram_matrix.index)),
    bigram_matrix.index
)

plt.xlabel("Next phoneme")
plt.ylabel("Current phoneme")

plt.title("Bigram Phonotactic Matrix")

plt.tight_layout()

plt.show()