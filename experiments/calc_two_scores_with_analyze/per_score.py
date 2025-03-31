from experiments.calc_two_scores_with_analyze.func_ru import run_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse
from datasets import load_dataset

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

    try:
        dataset = load_dataset("RussianNLP/CoAT")
        print("Successfully loaded CoAT dataset from Hugging Face")
    except:
        print("Could not load from Hugging Face, trying local files...")
        import glob
        
        dataset_files = glob.glob("./CoAT/datasets/**/*.json", recursive=True)
        if not dataset_files:
            import subprocess
            print("Cloning CoAT repository...")
            subprocess.run(["git", "clone", "https://github.com/RussianNLP/CoAT.git"])
            dataset_files = glob.glob("./CoAT/datasets/**/*.json", recursive=True)
        
        if not dataset_files:
            raise Exception("Failed to find CoAT dataset files")
            
        print(f"Found {len(dataset_files)} dataset files")
        dataset = {"data": []}
        
        for file_path in dataset_files:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                dataset["data"].extend(file_data)
    
    output_dir = "./results_coat"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing CoAT dataset with {len(dataset['data'] if isinstance(dataset, dict) else dataset)} samples")
    
    processed_data = []
    if isinstance(dataset, dict) and "data" in dataset:
        data_to_process = dataset["data"]
    else:
        data_to_process = []
        for split in dataset:
            for item in dataset[split]:
                if "text" in item:
                    data_to_process.append({
                        "text": item["text"],
                        "source": item.get("source", "unknown"),
                        "dataset": "CoAT"
                    })
    
    results = run_dataset(bino_chat, bino_coder, data=data_to_process)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"coat_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

    bino_chat.free_memory()
    bino_coder.free_memory()

if __name__ == "__main__":
    main()
