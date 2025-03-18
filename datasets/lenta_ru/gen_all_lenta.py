import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI
from google import genai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY_2 = os.getenv("DEEPSEEK_API_KEY_2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

deepseek_r1_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

deepseek_chat_client = OpenAI(
    api_key=DEEPSEEK_API_KEY_2 or DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

MODELS = [
    "chatgpt-o1-mini",
    "chatgpt-4o-mini",
    "deepseek-reasoner",
    "deepseek-chat",
    "gemini-2.0-flash"
]

SUMMARY_SYSTEM_MESSAGE = "Ты — профессиональный журналист, который создает краткие и информативные суммаризации новостей на русском языке."
GENERATION_SYSTEM_MESSAGE = "Ты — профессиональный журналист, который пишет подробные новостные статьи на русском языке."

def create_summary_prompt(news_text: str) -> str:
    return f"""
    Создай краткую суммаризацию следующей новости. Суммаризация должна содержать ключевую информацию и основные факты.
    Напиши только текст суммаризации, без лишних символов или выделений.
    Новость: {news_text}
    """

def create_generation_prompt(summary: str) -> str:
    return f"""
    На основе следующей суммаризации новости, напиши подробную новостную статью.
    Используй факты из суммаризации, но добавь контекст, детали или цитаты, чтобы статья была интересной и информативной.
    Статья должна быть написана журналистским стилем, без лишних символов или выделений.
    В ответе должен быть только полный текст новости.

    Суммаризация: {summary}
    """

def load_news(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_with_openai(news_text: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_MESSAGE},
                {"role": "user", "content": create_summary_prompt(news_text)}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing with OpenAI API: {e}")
        return "[Summarization Error]"

def distribute_news(news_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    model_news = {model: [] for model in MODELS}
    
    for i, news_item in enumerate(news_items):
        model_index = i % len(MODELS)
        model = MODELS[model_index]
        model_news[model].append(news_item)
    
    return model_news

def generate_with_openai(client, model_name: str, summary: str) -> str:
    prompt = create_generation_prompt(summary)
    
    model = "o1-mini" if model_name == "chatgpt-o1-mini" else "gpt-4o-mini"
    
    try:
        if model_name == "chatgpt-o1-mini":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": GENERATION_SYSTEM_MESSAGE + " " + prompt}
                ]
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
            )
            
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API ({model}): {e}")
        return "[Generation Error]"

def generate_with_deepseek(client, model_name: str, summary: str) -> str:
    prompt = create_generation_prompt(summary)
    
    model = "deepseek-reasoner" if model_name == "deepseek-reasoner" else "deepseek-chat"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying DeepSeek API ({model}): {e}")
        return "[Generation Error]"

def generate_with_gemini(summary: str) -> str:
    prompt = GENERATION_SYSTEM_MESSAGE + " " + create_generation_prompt(summary)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "[Generation Error]"

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
    
    news_with_summaries = []
    for news_item in tqdm(news_items, desc="Summarizing news", unit="item"):
        news_id = news_item["id"]
        
        if any(result.get("original_id") == news_id for result in all_results):
            print(f"Skipping news with ID {news_id} - already processed")
            continue
        
        news_text = news_item["text"]
        summary = summarize_with_openai(news_text)
        
        news_with_summaries.append({
            "id": news_id,
            "original_text": news_text,
            "summary": summary,
            "dataset": "lenta_ru"
        })
        
        time.sleep(1)
    
    model_news = distribute_news(news_with_summaries)
    
    for model, news_list in model_news.items():
        print(f"\nGenerating content using model {model}...")
        
        for news_item in tqdm(news_list, desc=model, unit="article"):
            news_id = news_item["id"]
            summary = news_item["summary"]
            dataset = news_item["dataset"]
            
            try:
                if model == "chatgpt-o1-mini":
                    generated_text = generate_with_openai(openai_client, model, summary)
                    delay = 1
                elif model == "chatgpt-4o-mini":
                    generated_text = generate_with_openai(openai_client, model, summary)
                    delay = 1
                elif model == "deepseek-reasoner":
                    generated_text = generate_with_deepseek(deepseek_r1_client, model, summary)
                    delay = 1
                elif model == "deepseek-chat":
                    generated_text = generate_with_deepseek(deepseek_chat_client, model, summary)
                    delay = 1
                elif model == "gemini-2.0-flash":
                    generated_text = generate_with_gemini(summary)
                    delay = 1
                
                result = {
                    "original_id": news_id,
                    "summary": summary,
                    "generated_content": generated_text,
                    "model": model,
                    "dataset": dataset + "_" + model
                }
                
                all_results.append(result)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error generating content for news with ID {news_id}: {e}")
    
    print(f"Completed! Total processed items: {len(all_results)}.")

def main():
    input_file = "datasets/lenta_ru/lenta_480.json"
    output_file = "datasets/lenta_ru/generated_news_content.json"
    
    process_news(input_file, output_file)

if __name__ == "__main__":
    main()