import time
import random
from openai import OpenAI
import os
import json
from tqdm import tqdm

var_name = os.getenv("OPENAI_API_KEY")

client = OpenAI()
OpenAI.api_key = var_name


dp_rewrite = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Перепиши следующий текст, значительно изменив структуру, стиль и формулировки. Сохрани основную мысль и фактическое содержание, но переформулируй каждое предложение так, чтобы конечная версия была достаточно отлична от исходной."
lp_rewrite = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Замени некоторые слова синонимами и перефразируй предложения, но не сокращай и не добавляй новую информацию. Количество слов в изменённом тексте должно остаться таким же."


def generate_paragraph(prompt, rewrite_mode="lp", model="o1-mini"):

    instruction = dp_rewrite if rewrite_mode == "dp" else lp_rewrite
    full_prompt = f"{instruction}\n\nТекст:\n{prompt}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
    )
    answer = response.choices[0].message.content.strip()
    return answer

def create_dataset(start_paragraph=0, end_paragraph=10, output_file="dataset.json"):
    with open("orig_texts.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    dataset = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Found existing dataset with {len(dataset)} entries")
    except FileNotFoundError:
        print("Creating new dataset")
    
    start_id = len(dataset) + 1
    
    end_paragraph = min(end_paragraph, len(data))
    if start_paragraph >= end_paragraph:
        raise ValueError("Start index must be less than end index")
    
    for i in tqdm(range(start_paragraph, end_paragraph), desc="Generating paragraphs", unit="paragraph"):
        prompt = data[i].get("text", "[Нет текста]")
        text = generate_paragraph(prompt)
        
        entry = {
            "id": start_id + (i - start_paragraph),
            "text": text,
            "source": "ai",
            "dataset": "SM LP chatGPT o1-mini"
        }
        dataset.append(entry)
        
        time.sleep(0.01)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Added {end_paragraph - start_paragraph} new entries to file '{output_file}'.")

if __name__ == "__main__":
    create_dataset(start_paragraph=0, end_paragraph=50, output_file="rew_from_o1-mini_two_part.json")
