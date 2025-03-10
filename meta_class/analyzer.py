import spacy
from collections import Counter
import math
import numpy as np

nlp = spacy.load("ru_core_news_lg")

def analyze_text(text):
    doc = nlp(text)

    tokens = [token.text for token in doc]
    words = [token.text for token in doc if token.is_alpha]
    unique_words = set(words)
    stop_words = [token.text for token in doc if token.is_stop]
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    return {
        "total_tokens": len(tokens),
        "total_words": len(words),
        "unique_words": len(unique_words),
        "stop_words": len(stop_words),
        "avg_word_length": round(avg_word_length, 2)
    }

if __name__ == "__main__":
    text = """
    В современной лингвистике наблюдается увеличение интереса к вопросам языкового моделирования текстового мира художественного произведения на основе дейксиса (Н.А. Серебрянская [1], С.А. Пушмина [2], О.Г. Мельник [3]). Дейксис рассматривается как одно из основных понятий прагматики. Взаимосвязь прагматического значения высказывания с контекстом также имеет большое значение для понимания ситуации общения. Определение дейксиса, предложенное Дж. Лайонзом, подчеркивает эту взаимосвязь: «Под дейксисом подразумевается локализация и идентификация лиц, объектов, событий, процессов и действий, о которых говорится или на которые ссылаются относительно пространственно-временного контекста, создаваемого и поддерживаемого актом высказывания и участием в нем, как правило, одного говорящего и, по крайней мере, одного адресата [4. С. 539]. 
    """
    
    analysis = analyze_text(text)
    
    print(f"\n📊 ПОЛНЫЙ АНАЛИЗ ТЕКСТА")
    print(f"\n=== БАЗОВАЯ СТАТИСТИКА ===")
    print(f"- Всего токенов: {analysis['total_tokens']}")
    print(f"- Всего слов: {analysis['total_words']}")
    print(f"- Уникальных слов: {analysis['unique_words']}")
    print(f"- Стоп-слов: {analysis['stop_words']}")
    print(f"- Средняя длина слова: {analysis['avg_word_length']} символов")
