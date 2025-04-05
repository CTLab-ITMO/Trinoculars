__all__ = ["binary_app"]

import gradio as gr
import torch
import os

from model_utils import load_model, classify_text
from binoculars_utils import initialize_binoculars, compute_scores

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MINIMUM_TOKENS = 200

SAMPLE_TEXT = """Привет! Я хотел бы рассказать вам о своём опыте путешествия по Петербургу. Невероятный город с богатой историей и красивой архитектурой. Особенно запомнился Эрмитаж с его огромной коллекцией произведений искусства. Также понравилась прогулка по каналам города, где можно увидеть множество старинных мостов и зданий."""

css = """
.human-text { 
    color: black !important;
    line-height: 1.9em; 
    padding: 0.5em; 
    background: #ccffcc; 
    border-radius: 0.5rem;
    font-weight: bold;
}
.ai-text { 
    color: black !important;
    line-height: 1.9em; 
    padding: 0.5em; 
    background: #ffad99; 
    border-radius: 0.5rem;
    font-weight: bold;
}
.analysis-block {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}
.scores {
    font-size: 1.1em;
    padding: 10px;
    background: #e6f7ff;
    border-radius: 5px;
    margin: 10px 0;
}
"""

def run_binary_classifier(text, show_analysis=False):
    if len(text.strip()) < MINIMUM_TOKENS:
        return gr.Markdown(f"Текст слишком короткий. Требуется минимум {MINIMUM_TOKENS} символов."), None, None
    
    # Initialize Binoculars models
    bino_chat, bino_coder = initialize_binoculars()

    # Load binary classifier model
    model, scaler, label_encoder, imputer = load_model()

    # Compute scores using binoculars
    scores = compute_scores(text, bino_chat, bino_coder)
    
    # Run classification
    result = classify_text(text, model, scaler, label_encoder, imputer=imputer, scores=scores)
    
    # Format results
    predicted_class = result['predicted_class']
    probabilities = result['probabilities']
    
    # Format probabilities
    prob_str = ""
    for cls, prob in probabilities.items():
        prob_str += f"- {cls}: {prob:.4f}\n"
    
    # Format scores
    scores_str = ""
    if scores:
        scores_str = "### Binoculars Scores\n"
        if 'score_chat' in scores:
            scores_str += f"- Score Chat: {scores['score_chat']:.4f}\n"
        if 'score_coder' in scores:
            scores_str += f"- Score Coder: {scores['score_coder']:.4f}\n"
    
    # Result markdown
    class_style = "human-text" if predicted_class == "Human" else "ai-text"
    result_md = f"""
## Результат классификации

Предсказанный класс: <span class="{class_style}">{predicted_class}</span>

### Вероятности классов:
{prob_str}

{scores_str}
"""
    
    # Analysis markdown
    analysis_md = None
    if show_analysis:
        features = result['features']
        text_analysis = result['text_analysis']
        
        basic_stats_dict = {
            'total_tokens': 'Количество токенов',
            'total_words': 'Количество слов',
            'unique_words': 'Количество уникальных слов',
            'stop_words': 'Количество стоп-слов',
            'avg_word_length': 'Средняя длина слова (символов)'
        }
        
        morph_dict = {
            'pos_distribution': 'Распределение частей речи',
            'unique_lemmas': 'Количество уникальных лемм',
            'lemma_word_ratio': 'Отношение лемм к словам'
        }
        
        synt_dict = {
            'dependencies': 'Зависимости между словами',
            'noun_chunks': 'Количество именных групп'
        }
        
        entities_dict = {
            'total_entities': 'Общее количество именованных сущностей',
            'entity_types': 'Типы именованных сущностей'
        }
        
        diversity_dict = {
            'ttr': 'TTR (отношение типов к токенам)',
            'mtld': 'MTLD (мера лексического разнообразия)'
        }
        
        structure_dict = {
            'sentence_count': 'Количество предложений',
            'avg_sentence_length': 'Средняя длина предложения (токенов)',
            'question_sentences': 'Количество вопросительных предложений',
            'exclamation_sentences': 'Количество восклицательных предложений'
        }
        
        readability_dict = {
            'words_per_sentence': 'Слов на предложение',
            'syllables_per_word': 'Слогов на слово',
            'flesh_kincaid_score': 'Индекс читабельности Флеша-Кинкейда',
            'long_words_percent': 'Процент длинных слов'
        }
        
        semantic_dict = {
            'avg_coherence_score': 'Средняя связность между предложениями'
        }
        
        analysis_md = "## Анализ текста\n\n"
        
        # Basic statistics
        analysis_md += "### Основная статистика\n"
        for key, value in text_analysis.get('basic_stats', {}).items():
            label = basic_stats_dict.get(key, key)
            if isinstance(value, float):
                analysis_md += f"- {label}: {value:.2f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Morphological analysis
        analysis_md += "### Морфологический анализ\n"
        morph_analysis = text_analysis.get('morphological_analysis', {})
        for key, value in morph_analysis.items():
            label = morph_dict.get(key, key)
            if key == 'pos_distribution':
                analysis_md += f"- {label}:\n"
                for pos, count in value.items():
                    pos_name = pos
                    if pos == 'NOUN': pos_name = 'Существительные'
                    elif pos == 'VERB': pos_name = 'Глаголы'
                    elif pos == 'ADJ': pos_name = 'Прилагательные'
                    elif pos == 'ADV': pos_name = 'Наречия'
                    elif pos == 'PROPN': pos_name = 'Имена собственные'
                    elif pos == 'DET': pos_name = 'Определители'
                    elif pos == 'ADP': pos_name = 'Предлоги'
                    elif pos == 'PRON': pos_name = 'Местоимения'
                    elif pos == 'CCONJ': pos_name = 'Сочинительные союзы'
                    elif pos == 'SCONJ': pos_name = 'Подчинительные союзы'
                    elif pos == 'NUM': pos_name = 'Числительные'
                    elif pos == 'PART': pos_name = 'Частицы'
                    elif pos == 'PUNCT': pos_name = 'Знаки препинания'
                    elif pos == 'AUX': pos_name = 'Вспомогательные глаголы'
                    elif pos == 'SYM': pos_name = 'Символы'
                    elif pos == 'INTJ': pos_name = 'Междометия'
                    elif pos == 'X': pos_name = 'Другое (X)'
                    analysis_md += f"  - {pos_name}: {count}\n"
            elif isinstance(value, float):
                analysis_md += f"- {label}: {value:.3f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Syntactic analysis
        analysis_md += "### Синтаксический анализ\n"
        synt_analysis = text_analysis.get('syntactic_analysis', {})
        for key, value in synt_analysis.items():
            label = synt_dict.get(key, key)
            if key == 'dependencies':
                analysis_md += f"- {label}:\n"
                for dep, count in value.items():
                    dep_name = dep
                    if dep == 'nsubj': dep_name = 'Подлежащие'
                    elif dep == 'obj': dep_name = 'Дополнения'
                    elif dep == 'amod': dep_name = 'Определения'
                    elif dep == 'nmod': dep_name = 'Именные модификаторы'
                    elif dep == 'ROOT': dep_name = 'Корневые узлы'
                    elif dep == 'punct': dep_name = 'Пунктуация'
                    elif dep == 'case': dep_name = 'Падежные маркеры'
                    elif dep == 'dep': dep_name = 'Общие зависимости'
                    elif dep == 'appos': dep_name = 'Приложения'
                    elif dep == 'flat:foreign': dep_name = 'Иностранные выражения'
                    elif dep == 'conj': dep_name = 'Сочинительные конструкции'
                    elif dep == 'obl': dep_name = 'Косвенные дополнения'
                    analysis_md += f"  - {dep_name}: {count}\n"
            elif key == 'noun_chunks':
                if isinstance(value, bool):
                    analysis_md += f"- {label}: {0 if value is False else value}\n"
                else:
                    analysis_md += f"- {label}: {value}\n"
            elif isinstance(value, float):
                analysis_md += f"- {label}: {value:.3f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Named entities
        analysis_md += "### Именованные сущности\n"
        entities = text_analysis.get('named_entities', {})
        for key, value in entities.items():
            label = entities_dict.get(key, key)
            if key == 'entity_types':
                analysis_md += f"- {label}:\n"
                for ent, count in value.items():
                    ent_name = ent
                    if ent == 'PER': ent_name = 'Люди'
                    elif ent == 'LOC': ent_name = 'Локации'
                    elif ent == 'ORG': ent_name = 'Организации'
                    analysis_md += f"  - {ent_name}: {count}\n"
            elif isinstance(value, float):
                analysis_md += f"- {label}: {value:.3f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Lexical diversity
        analysis_md += "### Лексическое разнообразие\n"
        for key, value in text_analysis.get('lexical_diversity', {}).items():
            label = diversity_dict.get(key, key)
            if isinstance(value, float):
                analysis_md += f"- {label}: {value:.3f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Text structure
        analysis_md += "### Структура текста\n"
        for key, value in text_analysis.get('text_structure', {}).items():
            label = structure_dict.get(key, key)
            if isinstance(value, float):
                analysis_md += f"- {label}: {value:.2f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Readability
        analysis_md += "### Читабельность\n"
        for key, value in text_analysis.get('readability', {}).items():
            label = readability_dict.get(key, key)
            if isinstance(value, float):
                analysis_md += f"- {label}: {value:.2f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
        analysis_md += "\n"
        
        # Semantic coherence
        analysis_md += "### Семантическая связность\n"
        for key, value in text_analysis.get('semantic_coherence', {}).items():
            label = semantic_dict.get(key, key)
            if isinstance(value, float):
                analysis_md += f"- {label}: {value:.3f}\n"
            else:
                analysis_md += f"- {label}: {value}\n"
    
    return gr.Markdown(result_md), gr.Markdown(analysis_md) if analysis_md else None, text

def reset_outputs():
    return None, None, ""

with gr.Blocks(css=css, theme=gr.themes.Base()) as binary_app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<h1>Бинарный классификатор: Human vs AI Detection</h1>")
            gr.HTML("<p>В этой демонстрации используется нейронная сеть для классификации текста как написанного человеком или сгенерированного искусственным интеллектом.</p>")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(value=SAMPLE_TEXT, placeholder="Введите текст для анализа", 
                                   lines=10, label="Текст для анализа")
            
            with gr.Row():
                analysis_checkbox = gr.Checkbox(label="Показать детальный анализ текста", value=False)
                submit_button = gr.Button("Классифицировать", variant="primary")
                clear_button = gr.Button("Очистить")
            
    with gr.Row():
        with gr.Column():
            result_output = gr.Markdown(label="Результат")
    
    with gr.Row():
        with gr.Column():
            analysis_output = gr.Markdown(label="Анализ")
            
    with gr.Accordion("О модели", open=False):
        gr.Markdown("""
        ### О бинарном классификаторе
        
        Эта демонстрация использует нейронную сеть для классификации текста как написанного человеком или сгенерированного ИИ.
        
        #### Архитектура модели:
        - Входной слой: Количество признаков (зависит от анализа текста)
        - Скрытые слои: [256, 192, 128, 64]
        - Выходной слой: 2 класса (Human, AI)
        - Dropout: 0.3
        
        #### Особенности:
        - Используется анализ текста и оценки качества текста с помощью Binoculars
        - Анализируются морфологические, синтаксические и семантические особенности текста
        - Вычисляются показатели лексического разнообразия и читабельности
        
        #### Рекомендации:
        - Для более точной классификации рекомендуется использовать тексты длиннее 200 слов
        - Модель обучена на русскоязычных текстах
        """)
    
    # Set up event handlers
    submit_button.click(
        fn=run_binary_classifier,
        inputs=[input_text, analysis_checkbox],
        outputs=[result_output, analysis_output, input_text]
    )
    
    clear_button.click(
        fn=reset_outputs,
        inputs=[],
        outputs=[result_output, analysis_output, input_text]
    ) 