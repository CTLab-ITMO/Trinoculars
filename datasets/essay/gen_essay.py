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

SUMMARY_SYSTEM_MESSAGE = "Ты — профессиональный историк, который создает краткие и информативные аннотации исторических эссе на русском языке."
GENERATION_SYSTEM_MESSAGE = "Ты — профессиональный историк, который пишет подробные исторические эссе на русском языке в академическом стиле."

def create_summary_prompt(essay_text: str) -> str:
    return f"""
    Создай краткую аннотацию следующего исторического эссе. Аннотация должна содержать ключевую информацию, основные исторические факты и выводы.
    Напиши только текст аннотации, без лишних символов или выделений.
    Эссе: {essay_text}
    """

def create_generation_prompt(summary: str) -> str:
    return f"""
    На основе следующей аннотации исторического эссе, напиши полное историческое эссе.
    Используй факты из аннотации, но добавь больше исторического контекста, деталей, аргументов и оценок.
    Эссе должно быть написано академическим историческим стилем, с анализом исторических событий, процессов и личностей.
    В ответе должен быть только полный текст эссе.

    Аннотация: {summary}
    """

def load_essays(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_with_openai(essay_text: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_MESSAGE},
                {"role": "user", "content": create_summary_prompt(essay_text)}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing with OpenAI API: {e}")
        return "[Summarization Error]"

def distribute_essays(essay_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    model_essays = {model: [] for model in MODELS}
    
    for i, essay_item in enumerate(essay_items):
        model_index = i % len(MODELS)
        model = MODELS[model_index]
        model_essays[model].append(essay_item)
    
    return model_essays

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

def process_essays(input_file: str, output_file: str) -> None:
    essay_items = load_essays(input_file)
    print(f"Loaded {len(essay_items)} essays")
    
    all_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found existing file with {len(all_results)} records")
    except FileNotFoundError:
        print(f"Creating new file: {output_file}")
    
    essays_with_summaries = []
    for essay_item in tqdm(essay_items, desc="Creating summaries", unit="essay"):
        essay_id = essay_item["id"]
        
        if any(result.get("original_id") == essay_id for result in all_results):
            print(f"Skipping essay with ID {essay_id} - already processed")
            continue
        
        essay_text = essay_item["text"]
        summary = summarize_with_openai(essay_text)
        
        essays_with_summaries.append({
            "id": essay_id,
            "original_text": essay_text,
            "summary": summary,
            "dataset": "essay"
        })
        
        time.sleep(1)
    
    model_essays = distribute_essays(essays_with_summaries)
    
    for model, essay_list in model_essays.items():
        print(f"\nGenerating content using model {model}...")
        
        for essay_item in tqdm(essay_list, desc=model, unit="essay"):
            essay_id = essay_item["id"]
            summary = essay_item["summary"]
            dataset = essay_item["dataset"]
            
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
                    "original_id": essay_id,
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
                print(f"Error generating content for essay with ID {essay_id}: {e}")
    
    print(f"Completed! Total items processed: {len(all_results)}.")

def main():
    input_file = "datasets/essay/essay_480.json"
    output_file = "datasets/essay/generated_essays.json"
    
    process_essays(input_file, output_file)

if __name__ == "__main__":
    main()