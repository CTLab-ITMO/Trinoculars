import pandas as pd
from text_analysis import analyze_text

def extract_features(text, feature_config=None, scores=None):
    if feature_config is None:
        feature_config = {
            'basic_scores': True, 
            'basic_text_stats': ['total_tokens', 'total_words', 'unique_words', 'stop_words', 'avg_word_length'],
            'morphological': ['pos_distribution', 'unique_lemmas', 'lemma_word_ratio'],
            'syntactic': ['dependencies', 'noun_chunks'],
            'entities': ['total_entities', 'entity_types'],
            'diversity': ['ttr', 'mtld'],
            'structure': ['sentence_count', 'avg_sentence_length', 'question_sentences', 'exclamation_sentences'],
            'readability': ['words_per_sentence', 'syllables_per_word', 'flesh_kincaid_score', 'long_words_percent'],
            'semantic': True
        }
    
    text_analysis = analyze_text(text)
    
    features_df = pd.DataFrame(index=[0])
    
    if scores:
        features_df['score_chat'] = scores.get('score_chat', 0)
        features_df['score_coder'] = scores.get('score_coder', 0)
    else:
        features_df['score_chat'] = 0
        features_df['score_coder'] = 0
        print("Warning: No scores provided, using zeros for score_chat and score_coder")
    
    if feature_config.get('basic_text_stats'):
        for feature in feature_config['basic_text_stats']:
            features_df[f'basic_{feature}'] = text_analysis.get('basic_stats', {}).get(feature, 0)
    
    if feature_config.get('morphological'):
        for feature in feature_config['morphological']:
            if feature == 'pos_distribution':
                pos_types = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'DET', 'ADP', 'PRON', 'CCONJ', 'SCONJ']
                for pos in pos_types:
                    features_df[f'pos_{pos}'] = text_analysis.get('morphological_analysis', {}).get('pos_distribution', {}).get(pos, 0)
            else:
                features_df[f'morph_{feature}'] = text_analysis.get('morphological_analysis', {}).get(feature, 0)
    
    if feature_config.get('syntactic'):
        for feature in feature_config['syntactic']:
            if feature == 'dependencies':
                dep_types = ['nsubj', 'obj', 'amod', 'nmod', 'ROOT', 'punct', 'case']
                for dep in dep_types:
                    features_df[f'dep_{dep}'] = text_analysis.get('syntactic_analysis', {}).get('dependencies', {}).get(dep, 0)
            else:
                features_df[f'synt_{feature}'] = text_analysis.get('syntactic_analysis', {}).get(feature, 0)
    
    if feature_config.get('entities'):
        for feature in feature_config['entities']:
            if feature == 'entity_types':
                entity_types = ['PER', 'LOC', 'ORG']
                for ent in entity_types:
                    features_df[f'ent_{ent}'] = text_analysis.get('named_entities', {}).get('entity_types', {}).get(ent, 0)
            else:
                features_df[f'ent_{feature}'] = text_analysis.get('named_entities', {}).get(feature, 0)
    
    if feature_config.get('diversity'):
        for feature in feature_config['diversity']:
            features_df[f'div_{feature}'] = text_analysis.get('lexical_diversity', {}).get(feature, 0)
    
    if feature_config.get('structure'):
        for feature in feature_config['structure']:
            features_df[f'struct_{feature}'] = text_analysis.get('text_structure', {}).get(feature, 0)
    
    if feature_config.get('readability'):
        for feature in feature_config['readability']:
            features_df[f'read_{feature}'] = text_analysis.get('readability', {}).get(feature, 0)
    
    if feature_config.get('semantic'):
        features_df['semantic_coherence'] = text_analysis.get('semantic_coherence', {}).get('avg_coherence_score', 0)
    
    return features_df, text_analysis 