__all__ = ["binary_app"]

import gradio as gr
import torch
import os

from model_utils import load_model, classify_text
from binoculars_utils import initialize_binoculars, compute_scores
from text_analysis import show_text_analysis

# Initialize Binoculars models
bino_chat, bino_coder = initialize_binoculars()

# Load binary classifier model
model, scaler, label_encoder, imputer = load_model()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MINIMUM_TOKENS = 50

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
        
        analysis_md = "## Анализ текста\n\n"
        
        # Basic statistics
        analysis_md += "### Основная статистика\n"
        analysis_md += f"- Всего токенов: {text_analysis['basic_stats']['total_tokens']}\n"
        analysis_md += f"- Всего слов: {text_analysis['basic_stats']['total_words']}\n"
        analysis_md += f"- Уникальных слов: {text_analysis['basic_stats']['unique_words']}\n"
        analysis_md += f"- Стоп-слов: {text_analysis['basic_stats']['stop_words']}\n"
        analysis_md += f"- Средняя длина слова: {text_analysis['basic_stats']['avg_word_length']:.2f} символов\n\n"
        
        # Lexical diversity
        analysis_md += "### Лексическое разнообразие\n"
        analysis_md += f"- TTR (Type-Token Ratio): {text_analysis['lexical_diversity']['ttr']:.3f}\n"
        analysis_md += f"- MTLD (упрощенный): {text_analysis['lexical_diversity']['mtld']:.2f}\n\n"
        
        # Text structure
        analysis_md += "### Структура текста\n"
        analysis_md += f"- Количество предложений: {text_analysis['text_structure']['sentence_count']}\n"
        analysis_md += f"- Средняя длина предложения: {text_analysis['text_structure']['avg_sentence_length']:.2f} токенов\n\n"
        
        # Readability
        analysis_md += "### Читабельность\n"
        analysis_md += f"- Flesch-Kincaid score: {text_analysis['readability']['flesh_kincaid_score']:.2f}\n"
        analysis_md += f"- Процент длинных слов: {text_analysis['readability']['long_words_percent']:.2f}%\n\n"
        
        # Semantic coherence
        analysis_md += "### Семантическая связность\n"
        analysis_md += f"- Средняя связность между предложениями: {text_analysis['semantic_coherence']['avg_coherence_score']:.3f}\n"
    
    return gr.Markdown(result_md), gr.Markdown(analysis_md) if analysis_md else None, text

def reset_outputs():
    return None, None, ""

with gr.Blocks(css=css, theme=gr.themes.Base()) as binary_app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<h1>Binary Classifier: Human vs AI Text Detection</h1>")
            gr.HTML("<p>This demo uses a neural network (Medium_Binary_Network) to classify text as either written by a human or generated by AI.</p>")
    
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
        
        Эта демонстрация использует нейронную сеть Medium_Binary_Network для классификации текста как написанного человеком или сгенерированного ИИ.
        
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
        - Для более точной классификации рекомендуется использовать тексты длиннее 100 слов
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