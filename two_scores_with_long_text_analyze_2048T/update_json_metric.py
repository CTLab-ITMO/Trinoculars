import os
import json
import spacy
from collections import Counter
import math
import numpy as np
import glob

nlp = spacy.load("ru_core_news_lg")

def analyze_text_for_json(text):
    doc = nlp(text)

    tokens = [token.text for token in doc]
    words = [token.text for token in doc if token.is_alpha]
    unique_words = set(words)
    stop_words = [token.text for token in doc if token.is_stop]
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    pos_counts = dict(Counter([token.pos_ for token in doc if token.is_alpha]))
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    unique_lemmas = set(lemmas)
    
    dependencies = dict(Counter([token.dep_ for token in doc if token.dep_ != ""]))
    
    noun_chunks_count = 0
    try:
        noun_chunks_count = len(list(doc.noun_chunks))
    except NotImplementedError:
        pass
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_counts = dict(Counter([ent.label_ for ent in doc.ents]))
    
    ttr = len(unique_words) / len(words) if words else 0
    
    def calculate_simplified_mtld(text_tokens, ttr_threshold=0.72):
        if len(text_tokens) < 10:
            return 0
        
        segments = []
        current_segment = []
        
        for token in text_tokens:
            current_segment.append(token)
            current_ttr = len(set(current_segment)) / len(current_segment)
            
            if current_ttr <= ttr_threshold and len(current_segment) >= 10:
                segments.append(current_segment)
                current_segment = []
        
        if current_segment:
            segments.append(current_segment)
            
        if not segments:
            return 0
        
        return len(text_tokens) / len(segments)
    
    mtld = calculate_simplified_mtld(words)
    
    sentences = list(doc.sents)
    sentence_lengths = [len(sent) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentences) if sentences else 0
    
    question_count = sum(1 for sent in sentences if sent.text.strip().endswith('?'))
    exclamation_count = sum(1 for sent in sentences if sent.text.strip().endswith('!'))
    
    words_per_sentence = len(words) / len(sentences) if sentences else 0
    
    def count_syllables_ru(word):
        return len([c for c in word.lower() if c in 'аеёиоуыэюя'])
    
    syllables = sum(count_syllables_ru(word) for word in words)
    syllables_per_word = syllables / len(words) if words else 0
    flesh_kincaid = 206.835 - 1.3 * words_per_sentence - 60.1 * syllables_per_word
    
    long_words = [word for word in words if count_syllables_ru(word) > 4]
    long_words_percent = len(long_words) / len(words) * 100 if words else 0
    
    word_freq = Counter(words)
    most_common_words = dict(word_freq.most_common(10))
    
    coherence_score = 0
    if len(sentences) > 1:
        coherence_scores = []
        for i in range(len(sentences)-1):
            if len(sentences[i]) > 0 and len(sentences[i+1]) > 0:
                try:
                    if sentences[i].vector_norm > 0 and sentences[i+1].vector_norm > 0:
                        sim = sentences[i].similarity(sentences[i+1])
                        coherence_scores.append(sim)
                except:
                    pass
        
        if coherence_scores:
            coherence_score = sum(coherence_scores) / len(coherence_scores)

    result = {
        "basic_stats": {
            "total_tokens": len(tokens),
            "total_words": len(words),
            "unique_words": len(unique_words),
            "stop_words": len(stop_words),
            "avg_word_length": round(avg_word_length, 2)
        },
        "morphological_analysis": {
            "pos_distribution": pos_counts,
            "unique_lemmas": len(unique_lemmas),
            "lemma_word_ratio": round(len(unique_lemmas)/len(unique_words), 2) if unique_words else 0
        },
        "syntactic_analysis": {
            "dependencies": dependencies,
            "noun_chunks": noun_chunks_count
        },
        "named_entities": {
            "total_entities": len(entities),
            "entity_types": entity_counts
        },
        "lexical_diversity": {
            "ttr": round(ttr, 3),
            "mtld": round(mtld, 2)
        },
        "text_structure": {
            "sentence_count": len(sentences),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "question_sentences": question_count,
            "exclamation_sentences": exclamation_count
        },
        "readability": {
            "words_per_sentence": round(words_per_sentence, 2),
            "syllables_per_word": round(syllables_per_word, 2),
            "flesh_kincaid_score": round(flesh_kincaid, 2),
            "long_words_percent": round(long_words_percent, 2)
        },
        "frequency_analysis": {
            "most_common_words": most_common_words
        },
        "semantic_coherence": {
            "avg_coherence_score": round(coherence_score, 3)
        }
    }
    
    return result

def update_json_files(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'data' not in data:
                print(f"Skipping file {file_path}: structure does not contain 'data' field")
                continue
            
            for item in data['data']:
                if 'text' in item:
                    text = item['text']
                    analysis_results = analyze_text_for_json(text)
                    item['text_analysis'] = analysis_results
                    
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"File {file_path} successfully updated")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

if __name__ == "__main__":
    directory = "res_full_text_tests"
    update_json_files(directory)