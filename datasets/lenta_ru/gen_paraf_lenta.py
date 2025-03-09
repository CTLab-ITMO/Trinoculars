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

DP_REWRITE = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Перепиши следующий текст, значительно изменив структуру, стиль и формулировки. Сохрани основную мысль и фактическое содержание, но переформулируй каждое предложение так, чтобы конечная версия была достаточно отлична от исходной."
LP_REWRITE = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Замени некоторые слова синонимами и перефразируй предложения, но не сокращай и не добавляй новую информацию. Количество слов в изменённом тексте должно остаться таким же."

MODELS = [
    {"name": "chatgpt-o1-mini", "type": "DP"},
    {"name": "chatgpt-4o-mini", "type": "LP"},
    {"name": "deepseek-reasoner", "type": "DP"},
    {"name": "deepseek-chat", "type": "LP"},
    {"name": "gemini-2.0-flash", "type": "LP"}
]

def load_news(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def distribute_news(news_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    model_news = {model["name"]: [] for model in MODELS}
    
    for i, news_item in enumerate(news_items):
        model_index = i % len(MODELS)
        model_name = MODELS[model_index]["name"]
        model_news[model_name].append(news_item)
    
    return model_news

def generate_with_openai(client, model_name: str, prompt_type: str, text: str) -> str:
    instruction = DP_REWRITE if prompt_type == "DP" else LP_REWRITE
    full_prompt = f"{instruction}\n\nТекст:\n{text}"
    
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
    full_prompt = f"{instruction}\n\nТекст:\n{text}"
    
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
    full_prompt = f"{instruction}\n\nТекст:\n{text}"
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "[Paraphrasing Error]"

def process_news(input_file: str, output_file: str) -> None:
    news_items = load_news(input_file)
    print(f"Loaded {len(news_items)} news items")
    
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    model_news = distribute_news(news_items)
    
    for model_name, news_list in model_news.items():
        model_type = next(model["type"] for model in MODELS if model["name"] == model_name)
        print(f"\nGenerating content using model {model_name} (type: {model_type})...")
        
        for news_item in tqdm(news_list, desc=model_name, unit="article"):
            news_id = news_item["id"]
            original_text = news_item["text"]
            dataset = news_item.get("dataset", "lenta_ru")
            
            if any(result.get("original_id") == news_id and result.get("model") == model_name for result in all_results):
                print(f"Skipping news with ID {news_id} - already processed by model {model_name}")
                continue
            
            try:
                if model_name == "chatgpt-o1-mini" or model_name == "chatgpt-4o-mini":
                    paraphrased_text = generate_with_openai(openai_client, model_name, model_type, original_text)
                    delay = 1
                elif model_name == "deepseek-reasoner" or model_name == "deepseek-chat":
                    paraphrased_text = generate_with_deepseek(deepseek_client, model_name, model_type, original_text)
                    delay = 1
                elif model_name == "gemini-2.0-flash":
                    paraphrased_text = generate_with_gemini(model_type, original_text)
                    delay = 1
                
                result = {
                    "original_id": news_id,
                    "original_text": original_text,
                    "paraphrased_text": paraphrased_text,
                    "model": model_name,
                    "paraphrasing_type": model_type,
                    "dataset": f"{dataset}_{model_type}_{model_name}"
                }
                
                all_results.append(result)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error paraphrasing news with ID {news_id}: {e}")
    
    print(f"Completed! Total processed items: {len(all_results)}.")

def main():
    input_file = "datasets/lenta_ru/lenta_480.json"
    output_file = "datasets/lenta_ru/paraphrased_news.json"
    
    process_news(input_file, output_file)

if __name__ == "__main__":
    main() 