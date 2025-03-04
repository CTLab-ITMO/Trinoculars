import time
import random
import os
import json
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_paragraph(prompt, temperature=0.8):
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты — помощник, который даёт развернутые ответы на русском языке. Пиши одним абзацем без лишних символов и перечислений."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating paragraph: {e}")
        return ""

def append_entry_to_json_array(entry, output_file):
    indent = " " * 4
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            entry_str = json.dumps(entry, ensure_ascii=False, indent=4)
            entry_lines = entry_str.splitlines()
            indented_entry = "\n".join(indent + line for line in entry_lines)
            f.write("[\n" + indented_entry + "\n]\n")
        return

    with open(output_file, "r+", encoding="utf-8") as f:
        content = f.read().rstrip()
        pos = content.rfind(']')
        if pos == -1:
            raise ValueError("Invalid file format: missing closing bracket")
        inner_content = content[content.find('[') + 1:pos].strip()
        comma = "," if inner_content else ""
        entry_str = json.dumps(entry, ensure_ascii=False, indent=4)
        entry_lines = entry_str.splitlines()
        indented_entry = "\n".join(indent + line for line in entry_lines)
        new_content = content[:pos].rstrip() + comma + "\n" + indented_entry + "\n" + content[pos:]
        f.seek(0)
        f.truncate()
        f.write(new_content)

def create_dataset(num_paragraphs=10, output_file="dataset.json"):
    with open('prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)['prompts']
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                start_id = len(data) + 1
            except json.JSONDecodeError:
                start_id = 1
        print(f"Found existing dataset with {start_id - 1} entries")
    else:
        print("Creating new dataset")
        start_id = 1

    for i in tqdm(range(num_paragraphs), desc="Generating paragraphs", unit="paragraph"):
        prompt = random.choice(prompts)
        text = generate_paragraph(prompt)
        
        if not text.strip():
            print("Empty response received, skipping entry.")
            continue
        
        entry = {
            "id": start_id + i,
            "text": text,
            "source": "ai",
            "dataset": "deepseek-chat"
        }
        append_entry_to_json_array(entry, output_file)
    
    print(f"Done! Added {num_paragraphs} new entries to '{output_file}'.")

if __name__ == "__main__":
    create_dataset(num_paragraphs=200, output_file="small_dataset_4.json")
