from experiments.calc_two_scores_with_analyze.func_ru import run_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse
import pandas as pd
import subprocess
import glob

def read_json_dataset(sample_limit=None):
    data_path = "./datasets/long_sc_valid/"
    
    if not os.path.exists(data_path):
        print(f"Directory {data_path} not found.")
        return None
    
    json_files = glob.glob(os.path.join(data_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_path}")
        return None
    
    print(f"Found {len(json_files)} JSON files in {data_path}")
    
    data_list = []
    for file_path in json_files:
        try:
            print(f"Reading file {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, list):
                print(f"Warning: Data in {file_path} is not a list. Skipping...")
                continue
            
            for item in json_data:
                if 'text' in item and item['text']:
                    source = item.get('source', 'human')
                    
                    data_list.append({
                        "text": item["text"],
                        "source": source
                    })
            
            print(f"Loaded {len(json_data)} records from {file_path}")
            
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
    
    print(f"Total loaded records: {len(data_list)}")
    
    if sample_limit and sample_limit < len(data_list):
        data_list = data_list[:sample_limit]
        print(f"Taking first {sample_limit} samples")
    
    return data_list

def main():
    chat_model_pairs = [
        {
            "observer": "deepseek-ai/deepseek-llm-7b-base",
            "performer": "deepseek-ai/deepseek-llm-7b-chat",
            "name": "Pair 1 - deepseek-llm-7b-base and deepseek-llm-7b-chat"
        }
    ]

    coder_model_pairs = [
        {
            "observer": "deepseek-ai/deepseek-llm-7b-base",
            "performer": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            "name": "Pair 2 - deepseek-llm-7b-base and deepseek-coder-7b-instruct-v1.5"
        }
    ]

    print(f"\nTesting pairs")
    print("-" * 50)
    
    bino_chat = Binoculars(
        mode="accuracy", 
        observer_name_or_path=chat_model_pairs[0]["observer"],
        performer_name_or_path=chat_model_pairs[0]["performer"],
        max_token_observed=2048
    )

    bino_coder = Binoculars(
        mode="accuracy", 
        observer_name_or_path=coder_model_pairs[0]["observer"],
        performer_name_or_path=coder_model_pairs[0]["performer"],
        max_token_observed=2048
    )

    sample_limit = None
    data_to_process = read_json_dataset(sample_limit)
    
    if data_to_process is None:
        print("Failed to load data from JSON files. Exiting program.")
        return
    
    total_samples = len(data_to_process)
    print(f"Loaded {total_samples} samples for processing")
    
    output_dir = "./results_long_sc_valid"
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_dataset(bino_chat, bino_coder, data=data_to_process)
    
    results["total_dataset_size"] = total_samples
    results["sampled_size"] = total_samples
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"long_sc_valid_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

    bino_chat.free_memory()
    bino_coder.free_memory()

if __name__ == "__main__":
    main()
