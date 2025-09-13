import spacy
import pandas as pd

with open ("data\\Il contadino astrologo.txt", "r", encoding="utf-8") as infile: 
    text = infile.read()

nlp = spacy.load("it_core_news_lg") 
doc = nlp(text)
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
            ent_types.append(token.ent_type_ if token.ent_type_ else "O")  
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

df.to_csv("extracted_features1.csv", index=False)
print("Estrazione completata, file salvato.")
