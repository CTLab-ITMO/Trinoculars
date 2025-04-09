import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SUMMARY_SYSTEM_MESSAGE = "Ты — профессиональный ученый, который создает краткие и информативные аннотации научных статей на русском языке."
GENERATION_SYSTEM_MESSAGE = "Ты — профессиональный ученый, который пишет подробные научные статьи на русском языке в академическом стиле."

def create_summary_prompt(article_text: str) -> str:
    return f"""
    Создай аннотацию следующей научной статьи. Аннотация должна содержать ключевую информацию, основные научные факты и выводы.
    Информации должно быть достаточно для того, чтобы понять, о чем идет речь в статье и была возможность на основе аннотации создать новую статью.
    Напиши только текст аннотации, без лишних символов или выделений.
    Статья: {article_text}
    """

def create_generation_prompt(summary: str, target_length: int) -> str:
    return f"""
    На основе следующей аннотации научной статьи, напиши полную научную статью.
    Используй факты из аннотации, но добавь больше научного контекста, деталей, аргументов и оценок.
    Статья должна быть написана академическим научным стилем, с анализом научных концепций, методологий и результатов.
    Статья должна быть примерно {target_length} слов в длину.
    В ответе должен быть только полный текст статьи.

    Аннотация: {summary}
    """

def count_words(text: str) -> int:
    """Подсчет количества слов в тексте."""
    return len(text.split())

def load_articles(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        
    # Проверка структуры JSON
    required_fields = ["id", "text", "source", "dataset"]
    for i, article in enumerate(articles):
        missing_fields = [field for field in required_fields if field not in article]
        if missing_fields:
            print(f"Предупреждение: статья #{i+1} не содержит поля: {', '.join(missing_fields)}")
    
    return articles

def summarize_with_gemini(article_text: str) -> str:
    prompt = SUMMARY_SYSTEM_MESSAGE + " " + create_summary_prompt(article_text)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing with Gemini API: {e}")
        return "[Summarization Error]"

def generate_with_gemini(summary: str, target_length: int) -> str:
    prompt = GENERATION_SYSTEM_MESSAGE + " " + create_generation_prompt(summary, target_length)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating with Gemini API: {e}")
        return "[Generation Error]"

def process_articles(input_file: str, output_file: str) -> None:
    articles = load_articles(input_file)
    print(f"Loaded {len(articles)} articles")
    
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    for article in tqdm(articles, desc="Processing articles", unit="article"):
        article_id = article.get("id")
        
        if article_id is None:
            print(f"Пропуск статьи без ID")
            continue
            
        if any(result.get("original_id") == article_id for result in all_results):
            print(f"Skipping article with ID {article_id} - already processed")
            continue
        
        article_text = article.get("text", "")
        if not article_text:
            print(f"Пропуск статьи с ID {article_id} - отсутствует текст")
            continue
        
        # Подсчитываем количество слов в оригинальной статье
        original_word_count = count_words(article_text)
        #print(f"Статья ID {article_id}: {original_word_count} слов")
        
        # Generate summary
        summary = summarize_with_gemini(article_text)
        time.sleep(5)  # Avoid rate limiting
        
        # Generate new article based on summary with target length
        generated_text = generate_with_gemini(summary, original_word_count)
        time.sleep(5)  # Avoid rate limiting
        
        # Подсчитываем слова в сгенерированной статье
        generated_word_count = count_words(generated_text)
        
        result = {
            "original_id": article_id,
            "summary": summary,
            "generated_content": generated_text,
            "model": "gemini-2.0-flash",
            "dataset": "Rus_sc_gemini",
            "original_word_count": original_word_count,
            "generated_word_count": generated_word_count
        }
        
        all_results.append(result)
        
        # Save after each article to prevent data loss
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        #print(f"Сгенерирована статья: {generated_word_count} слов (целевая длина: {original_word_count})")
    
    print(f"Completed! Total items processed: {len(all_results)}.")

def main():
    input_file = "datasets/long_sc_valid/rus_scientific_articles.json"
    output_file = "datasets/long_sc_valid/generated_articles.json"
    
    process_articles(input_file, output_file)

if __name__ == "__main__":
    main()