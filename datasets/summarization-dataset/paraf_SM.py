import time
import os
import json
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI
from google import genai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DP_REWRITE = "Ты — научный редактор, который переписывает научные тексты, сохраняя их структуру, смысл и терминологию. Перепиши следующий научный текст, значительно изменив синтаксическую структуру и стилистику, но при этом сохрани все научные термины, факты и аргументацию. Конечная версия должна быть академически корректна и отлична от исходной."
LP_REWRITE = "Ты — научный редактор, который улучшает читаемость научных текстов. Замени некоторые слова синонимами, где это возможно без потери точности научной терминологии, и перефразируй предложения для улучшения читаемости. Не изменяй научное содержание, методологию и результаты. Сохрани все специальные термины."

MODELS = [
    {"name": "chatgpt-o1-mini", "type": "DP"},
    {"name": "chatgpt-4o-mini", "type": "LP"},
    {"name": "deepseek-reasoner", "type": "DP"},
    {"name": "deepseek-chat", "type": "LP"},
    {"name": "gemini-2.0-flash", "type": "LP"}
]

def load_articles(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def distribute_articles(articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    model_articles = {model["name"]: [] for model in MODELS}
    
    for i, article in enumerate(articles):
        model_index = i % len(MODELS)
        model_name = MODELS[model_index]["name"]
        model_articles[model_name].append(article)
    
    return model_articles

def generate_with_openai(client, model_name: str, prompt_type: str, text: str) -> str:
    instruction = DP_REWRITE if prompt_type == "DP" else LP_REWRITE
    full_prompt = f"{instruction}\n\nНаучный текст:\n{text}"
    
    model = "o1-mini" if model_name == "chatgpt-o1-mini" else "gpt-4o-mini"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API ({model}): {e}")
        return "[Paraphrasing Error]"

def generate_with_deepseek(client, model_name: str, prompt_type: str, text: str) -> str:
    instruction = DP_REWRITE if prompt_type == "DP" else LP_REWRITE
    full_prompt = f"{instruction}\n\nНаучный текст:\n{text}"
    
    model = "deepseek-reasoner" if model_name == "deepseek-reasoner" else "deepseek-chat"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying DeepSeek API ({model}): {e}")
        return "[Paraphrasing Error]"

def generate_with_gemini(prompt_type: str, text: str) -> str:
    instruction = DP_REWRITE if prompt_type == "DP" else LP_REWRITE
    full_prompt = f"{instruction}\n\nНаучный текст:\n{text}"
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "[Paraphrasing Error]"

def process_articles(input_file: str, output_file: str) -> None:
    articles = load_articles(input_file)
    print(f"Loaded {len(articles)} scientific articles")
    
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    model_articles = distribute_articles(articles)
    
    for model_name, article_list in model_articles.items():
        model_type = next(model["type"] for model in MODELS if model["name"] == model_name)
        print(f"\nGenerating content using model {model_name} (type: {model_type})...")
        
        for article in tqdm(article_list, desc=model_name, unit="article"):
            article_id = article["id"]
            original_text = article["text"]
            dataset = article.get("dataset", "scientific_articles")
            
            if any(result.get("original_id") == article_id and result.get("model") == model_name for result in all_results):
                print(f"Skipping article with ID {article_id} - already processed by model {model_name}")
                continue
            
            try:
                text_for_processing = original_text
                
                if model_name == "chatgpt-o1-mini" or model_name == "chatgpt-4o-mini":
                    paraphrased_text = generate_with_openai(openai_client, model_name, model_type, text_for_processing)
                    delay = 1
                elif model_name == "deepseek-reasoner" or model_name == "deepseek-chat":
                    paraphrased_text = generate_with_deepseek(deepseek_client, model_name, model_type, text_for_processing)
                    delay = 15
                elif model_name == "gemini-2.0-flash":
                    paraphrased_text = generate_with_gemini(model_type, text_for_processing)
                    delay = 15
                
                result = {
                    "original_id": article_id,
                    "paraphrased_text": paraphrased_text,
                    "model": model_name,
                    "paraphrasing_type": model_type,
                    "dataset": f"{dataset}_{model_type}_{model_name}",
                    "source": 'ai'
                }
                
                all_results.append(result)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error paraphrasing article with ID {article_id}: {e}")
    
    print(f"Completed! Total processed items: {len(all_results)}.")

def main():
    input_file = "datasets/summarization-dataset/orig_texts.json"
    output_file = "datasets/summarization-dataset/paraphrased_articles.json"
    
    process_articles(input_file, output_file)

if __name__ == "__main__":
    main() 