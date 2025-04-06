import time
import os
import json
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Prompt for paraphrasing
DP_REWRITE = "Ты — научный редактор, который переписывает научные тексты, сохраняя их структуру, смысл и терминологию. Перепиши следующий научный текст, значительно изменив синтаксическую структуру и стилистику, но при этом сохрани все научные термины, факты и аргументацию. Конечная версия должна быть академически корректна и отлична от исходной."

def load_articles(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_with_openai(text: str) -> str:
    full_prompt = f"{DP_REWRITE}\n\nНаучный текст:\n{text}"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API (gpt-4o-mini): {e}")
        return "[Paraphrasing Error]"

def process_articles(input_file: str, output_file: str) -> None:
    articles = load_articles(input_file)
    print(f"Loaded {len(articles)} generated articles for paraphrasing")
    
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    for article in tqdm(articles, desc="Paraphrasing with gpt-4o-mini", unit="article"):
        article_id = article.get("id")
        generated_text = article.get("text")
        
        if article_id is None or not generated_text:
            print(f"Skipping article - missing ID or content")
            continue
            
        if any(result.get("id") == article_id for result in all_results):
            print(f"Skipping article with ID {article_id} - already processed")
            continue
        
        try:
            paraphrased_text = generate_with_openai(generated_text)
            
            result = {
                "id": article_id,
                "text": paraphrased_text,
                "model": "chatgpt-4o-mini",
                "paraphrasing_type": "DP",
                "dataset": "Rus scientific articles"
            }
            
            all_results.append(result)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error paraphrasing article with ID {article_id}: {e}")
    
    print(f"Completed! Total processed items: {len(all_results)}.")

def main():
    input_file = "datasets/long_sc_valid/rus_scientific_articles.json"
    output_file = "datasets/long_sc_valid/paraphrased_generated_articles.json"
    
    process_articles(input_file, output_file)

if __name__ == "__main__":
    main() 