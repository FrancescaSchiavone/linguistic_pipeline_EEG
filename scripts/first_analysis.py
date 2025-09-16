import spacy
import pandas as pd


import numpy as np

#da aggiungere: ciclo for per leggere file in una cartella e fare lo stesso per tutti
#aggiungere una funzione che nomina il file csv in base al nome del file in input

with open ("data\\Il contadino astrologo.txt", "r", encoding="utf-8") as infile: 
    text = infile.read()

nlp = spacy.load("it_core_news_lg") 

print(nlp.pipe_names) #elements in the pipeline
doc = nlp(text.lower())
tokens = []
lemmas = []
pos_tags = []
morphs = []
dep_rels = []
heads = []
ent_types = []
sent_ids = []
token_positions = []

for sent_id, sent in enumerate(doc.sents):
    for token_pos, token in enumerate(sent):
        if not token.is_punct and not token.is_space:
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)
            morphs.append(token.morph.to_dict())  
            dep_rels.append(token.dep_)
            heads.append(token.head.text)  
            ent_types.append(token.ent_type_ if token.ent_type_ else "O")  #da rivedere: sto usando .lower()
            sent_ids.append(sent_id)
            token_positions.append(token_pos)

df = pd.DataFrame({
    "sentence_id": sent_ids,
    "token_position": token_positions,
    "token": tokens,
    "lemma": lemmas,
    "pos": pos_tags,
    "morphology": morphs,
    "dependency_relation": dep_rels,
    "head_token": heads,
    "entity_type": ent_types
})

df.head()

df.to_csv("output\\extracted_features1.csv", index=False)

import stanza
stanza.download("it")

nlp = stanza.Pipeline(lang='it', processors='tokenize,pos,lemma,depparse,ner')
with open ("data\\Il contadino astrologo.txt", "r", encoding="utf-8") as infile: 
    text = infile.read()

doc =nlp(text)

for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text:<12} {word.lemma:<12} {word.pos:<6} {word.deprel:<10} → {word.head}")
tokens = []
lemmas = []
pos_tags = []
dep_rels = []
heads = []
ent_types = []
sent_ids = []
token_positions = []

# Estrazione dei dati
for sent_id, sentence in enumerate(doc.sentences):
    for token_pos, word in enumerate(sentence.words):
        tokens.append(word.text)
        lemmas.append(word.lemma)
        pos_tags.append(word.pos)
        dep_rels.append(word.deprel)
        heads.append(sentence.words[word.head - 1].text if word.head > 0 else "ROOT")
        ent_types.append("O")  # Stanza gestisce NER separatamente
        sent_ids.append(sent_id)
        token_positions.append(token_pos)

# Costruzione del DataFrame
df = pd.DataFrame({
    "sentence_id": sent_ids,
    "token_position": token_positions,
    "token": tokens,
    "lemma": lemmas,
    "pos": pos_tags,
    "dependency_relation": dep_rels,
    "head_token": heads,
    "entity_type": ent_types
})

# Salvataggio CSV
df.to_csv("output\\extracted_features_stanza.csv", index=False)
