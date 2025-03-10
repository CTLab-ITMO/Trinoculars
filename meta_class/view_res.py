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
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    for pair in model_pairs:
        print(f"\nTesting {pair['name']}")
        print("-" * 50)
        
        bino = Binoculars(
            mode="accuracy", 
            observer_name_or_path=pair["observer"],
            performer_name_or_path=pair["performer"]
        )

        data_dir = "./data"
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            print(f"\nProcessing file: {json_file}")
            file_path = os.path.join(data_dir, json_file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            dataset_name = os.path.splitext(json_file)[0]
            
            results_ru = run_ru_dataset(bino, data=dataset)
            results_ru['model_pair'] = {
                'observer': pair['observer'],
                'performer': pair['performer'],
                'pair_name': pair['name']
            }
            results_ru['dataset_name'] = dataset_name
            
            print(f"\nResults for dataset: {dataset_name}")
            print("\nStatistics:")
            print(f"Total examples: {results_ru['overall_stats']['total_examples']}")
            print(f"AI examples: {results_ru['overall_stats']['ai_examples']}")
            print(f"Human examples: {results_ru['overall_stats']['human_examples']}")
            print(f"Average score: {results_ru['overall_stats']['avg_score']:.4f}")
            print(f"Average AI score: {results_ru['overall_stats']['avg_ai_score']:.4f}")
            print(f"Average Human score: {results_ru['overall_stats']['avg_human_score']:.4f}")
            
            print(f"\nProcessed: {results_ru['check_counter']} examples")
            print(f"Errors: {results_ru['error_count']}")
            model_name = pair['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_ru = os.path.join(output_dir, f"results_ru_{dataset_name}_{model_name}_{timestamp}.json")
            with open(output_file_ru, 'w', encoding='utf-8') as f:
                json.dump(results_ru, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file_ru}")

        bino.free_memory()

if __name__ == "__main__":
    main()
