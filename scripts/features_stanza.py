import stanza
import pandas as pd

nlp = stanza.Pipeline("it") #modello per l'italiano

with open ("data\\Il contadino astrologo.txt", "r", encoding="utf-8") as infile: #apriamo il file (da sostituire con ciclo for)
    text = infile.read()

doc = nlp(text)

sentence_ids = []
sentences = []
tokens = []
PoS = []
lemma = []
depparse = []
head = []
ner = [] #da vedere
constituency = []
    
for sent_id, sentence in enumerate(doc.sentences):
    for word in sentence.words:
        sentence_ids.append(sent_id)
        sentences.append(sentence.text)
        tokens.append(word.text)
        lemma.append(word.lemma)
        PoS.append(word.pos)
        depparse.append(word.deprel)
        head.append(sentence.words[word.head - 1].text if word.head > 0 else "ROOT")
        constituency.append(sentence.constituency if hasattr(sentence, "constituency") else None)

# se da errore 'All arrays must be of the same lenght" --> rigenere il codice a partire da sentence_id =[]
df = pd.DataFrame ({
    "sentence_ids": sentence_ids,
    "tokens": tokens,
    "lemma" : lemma,
    "PoS": PoS,
    "depparse": depparse,
    "head": head,
    "constituency": constituency})

df.head()
        
df.to_csv("output\\extracted_features_stanza1.csv", index=False)

from collections import Counter

lemma_freq = Counter(lemma)

sorted_lemma = lemma_freq.most_common()

for lem, freq in sorted_lemma:
    print(f"{lem}: {freq}")


import matplotlib.pyplot as plt

top_lemmi = dict(sorted_lemma)
labels, values = zip(*sorted_lemma)

plt.style.use("seaborn-v0_8")
plt.figure(figsize=(10, 6))
plt.bar(top_lemmi.keys(), top_lemmi.values(),color = "powderblue" )
plt.plot(labels, values, color='lightsteelblue', linestyle='-', linewidth=2)


plt.title("Frequenza dei lemmi più comuni")
plt.xticks([])
plt.xlabel("Lemma")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.show()
