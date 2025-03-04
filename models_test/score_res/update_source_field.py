import json
import os
import glob
import sys

def update_source_field(json_file_path):
    print(f"Processing file: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found.")
        return False
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        total_records = 0
        updated_records = 0
        
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                total_records += 1
                if "dataset" in item and "source" in item:
                    if isinstance(item["dataset"], str) and item["dataset"].startswith("SM") and item["source"] == "ai":
                        item["source"] = "ai+rew"
                        updated_records += 1
        elif isinstance(data, list):
            for item in data:
                total_records += 1
                if "dataset" in item and "source" in item:
                    if isinstance(item["dataset"], str) and item["dataset"].startswith("SM") and item["source"] == "ai":
                        item["source"] = "ai+rew"
                        updated_records += 1
        elif isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, dict):
                    total_records += 1
                    if "dataset" in item and "source" in item:
                        if isinstance(item["dataset"], str) and item["dataset"].startswith("SM") and item["source"] == "ai":
                            item["source"] = "ai+rew"
                            updated_records += 1
        
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        
        print(f"Processing completed. Total records: {total_records}, updated: {updated_records}")
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def process_files(file_patterns):
    processed_files = 0
    
    for pattern in file_patterns:
        if not pattern.endswith('.json'):
            pattern += '.json'
            
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No files found matching pattern '{pattern}'")
            continue
        
        for file_path in files:
            if update_source_field(file_path):
                processed_files += 1
    
    return processed_files

def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
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
        return None
    
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def merge_json_files(file1_path, file2_path, output_path):
    print(f"Loading first file: {file1_path}")
    data1 = load_json_file(file1_path)
    if data1 is None:
        return False
    
    print(f"Loading second file: {file2_path}")
    data2 = load_json_file(file2_path)
    if data2 is None:
        return False
    
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
        return True
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

if __name__ == "__main__":
    files_to_process = [
        "results_ru_ru_detection_dataset_three_Pair_1___deepseek_llm_7b_base_and_deepseek_coder_7b_instruct_v1.5_20250302_113229",
        "results_ru_ru_detection_dataset_three_Pair_2___deepseek_llm_7b_base_and_deepseek_llm_7b_chat_20250302_114019"
    ]
    
    num_processed = process_files(files_to_process)
    print(f"Total files processed: {num_processed}")
    
    file1 = files_to_process[0] + ".json"
    file2 = files_to_process[1] + ".json"
    output_file = "merged_results_three.json"
    
    print("\nMerging files...")
    if merge_json_files(file1, file2, output_file):
        print("File merge completed successfully!")
    else:
        print("Error merging files.")
        sys.exit(1)