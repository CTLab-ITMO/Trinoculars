from meta_class.func_ru import run_ru_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse

def main():

    model_pairs = [
        {
            "observer": "deepseek-ai/deepseek-llm-7b-base",
            "performer": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            "name": "Pair 1 - deepseek-llm-7b-base and deepseek-coder-7b-instruct-v1.5"
        },
        {
            "observer": "deepseek-ai/deepseek-llm-7b-base",
            "performer": "deepseek-ai/deepseek-llm-7b-chat",
            "name": "Pair 2 - deepseek-llm-7b-base and deepseek-llm-7b-chat"
        }
    ]
    output_dir = "./results_two_scores"
    os.makedirs(output_dir, exist_ok=True)

    for pair in model_pairs:
        print(f"\nTesting {pair['name']}")
        print("-" * 50)
        
        bino = Binoculars(
            mode="accuracy", 
            observer_name_or_path=pair["observer"],
            performer_name_or_path=pair["performer"]
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
            
            results_ru = run_ru_dataset(bino, data=dataset)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_ru = os.path.join(output_dir, f"results_ru_{dataset_name}_{timestamp}.json")
            with open(output_file_ru, 'w', encoding='utf-8') as f:
                json.dump(results_ru, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file_ru}")

        bino.free_memory()

if __name__ == "__main__":
    main()
