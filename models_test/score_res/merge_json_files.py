import json
import os
import sys

def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        elif isinstance(data, dict):
            return list(data.values())
        
        print(f"Error: Unknown file structure {file_path}")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def merge_json_files(file1_path, file2_path, output_path):
    print(f"Loading first file: {file1_path}")
    data1 = load_json_file(file1_path)
    
    print(f"Loading second file: {file2_path}")
    data2 = load_json_file(file2_path)
    
    text_to_data1 = {item["text"]: item for item in data1}
    
    merged_data = []
    text_to_merged_item = {}
    
    for item1 in data1:
        text = item1["text"]
        merged_item = {
            "text": text,
            "source": item1.get("source", ""),
            "dataset": item1.get("dataset", ""),
            "score1": item1.get("score", None)
        }
        text_to_merged_item[text] = merged_item
    
    matched_count = 0
    unmatched_count = 0
    
    for item2 in data2:
        text = item2["text"]
        if text in text_to_merged_item:
            merged_item = text_to_merged_item[text]
            merged_item["score2"] = item2.get("score", None)
            matched_count += 1
        else:
            merged_item = {
                "text": text,
                "source": item2.get("source", ""),
                "dataset": item2.get("dataset", ""),
                "score2": item2.get("score", None),
                "score1": None
            }
            text_to_merged_item[text] = merged_item
            unmatched_count += 1
    
    merged_data = list(text_to_merged_item.values())
    for i, item in enumerate(merged_data, 1):
        item["id"] = i
    
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump({"data": merged_data}, file, ensure_ascii=False, indent=2)
        
        print(f"Merged data saved to {output_path}")
        print(f"Total items: {len(merged_data)}")
        print(f"Matched items: {matched_count}")
        print(f"Unmatched items: {unmatched_count}")
        
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    file1 = "results_ru_ru_detection_dataset_Pair_1___deepseek_llm_7b_base_and_deepseek_coder_7b_instruct_v1.5_20250228_183100.json"
    file2 = "results_ru_ru_detection_dataset_Pair_2___deepseek_llm_7b_base_and_deepseek_llm_7b_chat_20250228_190028.json"
    output_file = "merged_results.json"
    
    merge_json_files(file1, file2, output_file) 