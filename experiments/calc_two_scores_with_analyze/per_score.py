from experiments.calc_two_scores_with_analyze.func_ru import run_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse
import pandas as pd
import subprocess

def read_coat_data(sample_limit=None):
    csv_path = "./CoAT/datasets/binary/val.csv"
    
    if not os.path.exists(csv_path):
        if not os.path.exists("./CoAT"):
            print("CoAT repository not found. Cloning...")
            try:
                subprocess.run(["git", "clone", "https://github.com/RussianNLP/CoAT.git"], check=True)
                subprocess.run(["git", "lfs", "pull"], cwd="./CoAT", check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error cloning repository: {str(e)}")
                return None
            except FileNotFoundError:
                print("Git is not installed or not in PATH. Please install Git and Git LFS.")
                return None
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found even after cloning the repository.")
        return None
    
    try:
        print(f"Reading file {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Total number of rows: {len(df)}")
        print(f"Columns in dataset: {', '.join(df.columns)}")
        
        if sample_limit and sample_limit < len(df):
            df = df.head(sample_limit)
            print(f"Taking first {sample_limit} samples")
        
        data_list = []
        for _, row in df.iterrows():
            data_list.append({
                "text": row["text"],
                "is_artificial": row["label"] == 1
            })
        
        return data_list
    
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

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
    output_dir = "./results_two_scores"
    os.makedirs(output_dir, exist_ok=True)

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

    sample_limit = 10000
    data_to_process = read_coat_data(sample_limit)
    
    if data_to_process is None:
        print("Failed to load CoAT data. Exiting program.")
        return
    
    total_samples = len(data_to_process)
    print(f"Loaded {total_samples} samples for processing")
    
    output_dir = "./results_coat"
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_dataset(bino_chat, bino_coder, data=data_to_process)
    
    results["total_dataset_size"] = total_samples
    results["sampled_size"] = total_samples
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"coat_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

    bino_chat.free_memory()
    bino_coder.free_memory()

if __name__ == "__main__":
    main()
