from meta_class.func_ru import run_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse

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
        max_token_observed=512
    )

    bino_coder = Binoculars(
        mode="accuracy", 
        observer_name_or_path=coder_model_pairs[0]["observer"],
        performer_name_or_path=coder_model_pairs[0]["performer"],
        max_token_observed=512
    )

    data_dir = "./datasets/per_dataset"
    json_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    for json_file in json_files:
        print(f"\nProcessing file: {json_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        dataset_name = os.path.splitext(os.path.basename(json_file))[0]
        
        results = run_dataset(bino_chat, bino_coder, data=dataset)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"per_score_{dataset_name}_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")

    bino_chat.free_memory()
    bino_coder.free_memory()

if __name__ == "__main__":
    main()
