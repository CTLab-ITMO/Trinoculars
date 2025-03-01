import time
import os
import json
from tqdm import tqdm
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

dp_rewrite = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Перепиши следующий текст, значительно изменив структуру, стиль и формулировки. Сохрани основную мысль и фактическое содержание, но переформулируй каждое предложение так, чтобы конечная версия была достаточно отлична от исходной."
lp_rewrite = "Ты — помощник, который переписывает текст, сохраняя его структуру и смысл. Замени некоторые слова синонимами и перефразируй предложения, но не сокращай и не добавляй новую информацию. Количество слов в изменённом тексте должно остаться таким же."

def generate_paragraph(prompt, rewrite_mode="lp", model="gemini-2.0-flash"):
    try:
        instruction = dp_rewrite if rewrite_mode == "dp" else lp_rewrite
        full_prompt = f"{instruction}\n\nТекст:\n{prompt}"
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )
        answer = response.text.strip()
        return answer
    except Exception as e:
        print(f"Error during API Gemini request: {e}")
        return "[Generation Error]"

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
        prompt = data[i].get("text", "[No text]")
        text = generate_paragraph(prompt)
        
        entry = {
            "id": start_id + (i - start_paragraph),
            "text": text,
            "source": "ai",
            "dataset": "SM LP gemini-2.0-flash"
        }
        dataset.append(entry)
        
        time.sleep(1)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Added {end_paragraph - start_paragraph} new entries to file '{output_file}'.")

if __name__ == "__main__":
    create_dataset(start_paragraph=300, end_paragraph=400, output_file="rew_from_gemini_chat_lp.json")
