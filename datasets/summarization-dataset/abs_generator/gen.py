import json
import os
import time
import random
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

SYSTEM_MESSAGE = "Ты — профессиональный научный писатель, работающий на русском языке."

def create_prompt(title: str, content: str, category: str, word_count: int) -> str:
    return f"""
    На основе следующего заголовка, содержания и категории, напиши научную статью, сохраняя стиль и тематику оригинального содержания.
    Старайся писать доступным человеческим языком без лишних перечеслений, символов или выделений текста, не используя научные термины, если они не являются обязательными для понимания.
    В ответе не должны быть ничего лишнего, только полный текст статьи без источников.
    Сгенерированная статья должна содержать примерно {word_count} слов.

    Заголовок: {title}
    Содержание: {content}
    Категория: {category}
    Ожидаемый объем: около {word_count} слов"""

def load_abstracts(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def distribute_abstracts(abstracts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    model_abstracts = {model: [] for model in MODELS}
    
    for i, abstract in enumerate(abstracts):
        model_index = i % len(MODELS)
        model = MODELS[model_index]
        model_abstracts[model].append(abstract)
    
    return model_abstracts

def generate_with_openai(client, model_name: str, title: str, content: str, category: str, word_count: int) -> str:
    prompt = create_prompt(title, content, category, word_count)
    
    model = "o1-mini" if model_name == "chatgpt-o1-mini" else "gpt-4o-mini"
    
    try:
        if model_name == "chatgpt-o1-mini":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": SYSTEM_MESSAGE + " " + prompt}
                ]
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
            )
            
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API ({model}): {e}")
        return "[Generation Error]"

def generate_with_deepseek(client, model_name: str, title: str, content: str, category: str, word_count: int) -> str:
    prompt = create_prompt(title, content, category, word_count)
    
    model = "deepseek-reasoner" if model_name == "deepseek-reasoner" else "deepseek-chat"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying DeepSeek API ({model}): {e}")
        return "[Generation Error]"

def generate_with_gemini(title: str, content: str, category: str, word_count: int) -> str:
    prompt = SYSTEM_MESSAGE + " " + create_prompt(title, content, category, word_count)
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "[Generation Error]"

def generate_articles(model_abstracts: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    for model, abstracts in model_abstracts.items():
        print(f"\nGenerating articles using model {model}...")
        
        for abstract in tqdm(abstracts, desc=model, unit="article"):
            title = abstract["title"]
            content = abstract["content"]
            category = abstract["category"]
            abstract_id = abstract["id"]
            word_count = abstract.get("word_count", 2000)
            
            if any(result.get("id") == abstract_id for result in all_results):
                print(f"Skipping article with ID {abstract_id} - already exists")
                continue
            
            try:
                if model == "chatgpt-o1-mini":
                    generated_text = generate_with_openai(openai_client, model, title, content, category, word_count)
                    delay = 1
                elif model == "chatgpt-4o-mini":
                    generated_text = generate_with_openai(openai_client, model, title, content, category, word_count)
                    delay = 1
                elif model == "deepseek-reasoner":
                    generated_text = generate_with_deepseek(deepseek_r1_client, model, title, content, category, word_count)
                    delay = 1
                elif model == "deepseek-chat":
                    generated_text = generate_with_deepseek(deepseek_chat_client, model, title, content, category, word_count)
                    delay = 1
                elif model == "gemini-2.0-flash":
                    generated_text = generate_with_gemini(title, content, category, word_count)
                    delay = 1
                
                result = {
                    "id": abstract_id,
                    "title": title,
                    "content": generated_text,
                    "source": "ai",
                    "dataset": f"SM abs {model}"
                }
                
                all_results.append(result)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error generating article with ID {abstract_id}: {e}")
    
    print(f"Completed! Total generated articles: {len(all_results)}.")

def main():
    input_file = "datasets/summarization-dataset/abs_generator/abstracts.json"
    output_file = "datasets/summarization-dataset/abs_generator/generated_scientific_articles.json"
    
    abstracts = load_abstracts(input_file)
    print(f"Loaded {len(abstracts)} abstracts")
    
    model_abstracts = distribute_abstracts(abstracts)
    
    for model, model_abs in model_abstracts.items():
        print(f"Model {model}: {len(model_abs)} abstracts")
    
    generate_articles(model_abstracts, output_file)

if __name__ == "__main__":
    main()
