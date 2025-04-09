import spacy
from collections import Counter

nlp = spacy.load("ru_core_news_lg")

def analyze_text(text):
    doc = nlp(text)

    tokens = [token.text for token in doc]
    words = [token.text for token in doc if token.is_alpha]
    unique_words = set(words)
    stop_words = [token.text for token in doc if token.is_stop]
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    unique_lemmas = set(lemmas)
    
    dependencies = Counter([token.dep_ for token in doc if token.dep_ != ""])
    
    has_noun_chunks = False
    try:
        next(doc.noun_chunks, None)
        has_noun_chunks = True
    except NotImplementedError:
        pass
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_counts = Counter([ent.label_ for ent in doc.ents])
    
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
    
    words_per_sentence = len(words) / len(sentences) if sentences else 0
    
    def count_syllables_ru(word):
        return len([c for c in word.lower() if c in 'аеёиоуыэюя'])
    
    syllables = sum(count_syllables_ru(word) for word in words)
    syllables_per_word = syllables / len(words) if words else 0
    flesh_kincaid = 206.835 - 1.3 * words_per_sentence - 60.1 * syllables_per_word
    
    long_words = [word for word in words if count_syllables_ru(word) > 4]
    long_words_percent = len(long_words) / len(words) * 100 if words else 0
    
    sentence_count = len(sentences)
    question_count = sum(1 for sent in sentences if sent.text.strip().endswith('?'))
    exclamation_count = sum(1 for sent in sentences if sent.text.strip().endswith('!'))
    
    coherence_scores = []
    if len(sentences) > 1:
        for i in range(len(sentences)-1):
            if len(sentences[i]) > 0 and len(sentences[i+1]) > 0:
                try:
                    if sentences[i].vector_norm > 0 and sentences[i+1].vector_norm > 0:
                        sim = sentences[i].similarity(sentences[i+1])
                        coherence_scores.append(sim)
                except:
                    pass
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    
    analysis_results = {
        'basic_stats': {
            'total_tokens': len(tokens),
            'total_words': len(words),
            'unique_words': len(unique_words),
            'stop_words': len(stop_words),
            'avg_word_length': avg_word_length
        },
        'morphological_analysis': {
            'pos_distribution': {pos: count for pos, count in pos_counts.items()},
            'unique_lemmas': len(unique_lemmas),
            'lemma_word_ratio': len(unique_lemmas) / len(unique_words) if unique_words else 0
        },
        'syntactic_analysis': {
            'dependencies': {dep: count for dep, count in dependencies.most_common(10)},
            'noun_chunks': has_noun_chunks
        },
        'named_entities': {
            'total_entities': len(entities),
            'entity_types': {label: count for label, count in entity_counts.items()}
        },
        'lexical_diversity': {
            'ttr': ttr,
            'mtld': mtld
        },
        'text_structure': {
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'question_sentences': question_count,
            'exclamation_sentences': exclamation_count
        },
        'readability': {
            'words_per_sentence': words_per_sentence,
            'syllables_per_word': syllables_per_word,
            'flesh_kincaid_score': flesh_kincaid,
            'long_words_percent': long_words_percent
        },
        'semantic_coherence': {
            'avg_coherence_score': avg_coherence
        }
    }
    
    return analysis_results

def show_text_analysis(analysis):
    print("\n📊 TEXT ANALYSIS")
    
    print("\n=== BASIC STATISTICS ===")
    print(f"- Total tokens: {analysis['basic_stats']['total_tokens']}")
    print(f"- Total words: {analysis['basic_stats']['total_words']}")
    print(f"- Unique words: {analysis['basic_stats']['unique_words']}")
    print(f"- Stop words: {analysis['basic_stats']['stop_words']}")
    print(f"- Average word length: {analysis['basic_stats']['avg_word_length']:.2f} characters")
    
    print("\n=== MORPHOLOGICAL ANALYSIS ===")
    print("- POS distribution:")
    for pos, count in analysis['morphological_analysis']['pos_distribution'].items():
        print(f"  • {pos}: {count}")
    print(f"- Unique lemmas: {analysis['morphological_analysis']['unique_lemmas']}")
    
    print("\n=== SYNTACTIC ANALYSIS ===")
    print("- Syntactic dependencies (top-5):")
    for i, (dep, count) in enumerate(analysis['syntactic_analysis']['dependencies'].items()):
        if i >= 5:
            break
        print(f"  • {dep}: {count}")
    
    print("\n=== NAMED ENTITIES ===")
    print(f"- Total entities: {analysis['named_entities']['total_entities']}")
    
    print("\n=== LEXICAL DIVERSITY ===")
    print(f"- TTR (type-token ratio): {analysis['lexical_diversity']['ttr']:.3f}")
    print(f"- MTLD (simplified): {analysis['lexical_diversity']['mtld']:.2f}")
    
    print("\n=== TEXT STRUCTURE ===")
    print(f"- Sentence count: {analysis['text_structure']['sentence_count']}")
    print(f"- Average sentence length: {analysis['text_structure']['avg_sentence_length']:.2f} tokens")
    
    print("\n=== READABILITY ===")
    print(f"- Flesch-Kincaid score: {analysis['readability']['flesh_kincaid_score']:.2f}")
    print(f"- Long words percentage: {analysis['readability']['long_words_percent']:.2f}%")
    
    print(f"\n=== SEMANTIC COHERENCE ===")
    print(f"- Average coherence between sentences: {analysis['semantic_coherence']['avg_coherence_score']:.3f}") 