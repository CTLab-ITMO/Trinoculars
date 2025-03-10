import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

generated_essays_path = 'datasets/per_dataset/scientific_texts/generated_scientific.json'
original_essay_path = 'datasets/per_dataset/scientific_texts/orig_scientific.json'
paraphrased_essays_path = 'datasets/per_dataset/scientific_texts/paraphrased_scientific.json'

with open(generated_essays_path, 'r', encoding='utf-8') as f:
    generated_essays = json.load(f)

with open(original_essay_path, 'r', encoding='utf-8') as f:
    original_essays = json.load(f)

with open(paraphrased_essays_path, 'r', encoding='utf-8') as f:
    paraphrased_essays = json.load(f)

essays_by_id = {}

for essay in generated_essays:
    essay_id = essay['id']
    if essay_id not in essays_by_id:
        essays_by_id[essay_id] = {}
    essays_by_id[essay_id]['generated'] = essay['text']

for essay in original_essays:
    essay_id = essay['id']
    if essay_id not in essays_by_id:
        essays_by_id[essay_id] = {}
    essays_by_id[essay_id]['original'] = essay['text']

for essay in paraphrased_essays:
    essay_id = essay['id']
    if essay_id not in essays_by_id:
        essays_by_id[essay_id] = {}
    essays_by_id[essay_id]['paraphrased'] = essay['text']

def compute_cosine_similarity(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)

threshold = 0.1

dissimilar_texts = []

for essay_id, texts in essays_by_id.items():
    if len(texts) < 3:
        continue
    
    text_list = [texts['original'], texts['generated'], texts['paraphrased']]
    similarity_matrix = compute_cosine_similarity(text_list)
    
    df = pd.DataFrame(
        similarity_matrix,
        index=['original', 'generated', 'paraphrased'],
        columns=['original', 'generated', 'paraphrased']
    )
    
    pairs = [
        ('original', 'generated'),
        ('original', 'paraphrased'),
        ('generated', 'paraphrased')
    ]
    
    for text1, text2 in pairs:
        sim = df.loc[text1, text2]
        if sim < threshold:
            dissimilar_texts.append({
                'id': essay_id,
                'pair': f"{text1} vs {text2}",
                'similarity': sim
            })

if dissimilar_texts:
    print("IDs of texts that are dissimilar (cosine similarity < 0.1):")
    for item in dissimilar_texts:
        print(f"ID: {item['id']}, {item['pair']}, similarity: {item['similarity']:.4f}")
else:
    print("All texts have similarity above the threshold of 0.1")

if dissimilar_texts:
    unique_ids = sorted(set(item['id'] for item in dissimilar_texts))
    print("\nUnique IDs with dissimilar texts:")
    print(unique_ids)
