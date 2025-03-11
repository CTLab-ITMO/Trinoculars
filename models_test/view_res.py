from func import run_eng_dataset, run_ru_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run model testing on selected dataset')
    parser.add_argument('--dataset', type=str, choices=['ru', 'eng', 'all'], 
                       default='all', help='Choose dataset for testing (ru/eng/all)')
    args = parser.parse_args()

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

        if args.dataset in ['eng', 'all']:
            results_eng = run_eng_dataset(bino, sample_rate=0.003, max_samples=5000)
            results_eng['model_pair'] = {
                'observer': pair['observer'],
                'performer': pair['performer'],
                'pair_name': pair['name']
            }
            
            print("\nEnglish dataset results:")
            print("\nMetrics:")
            print(f"F1 Score: {results_eng['metrics']['f1_score']:.4f}")
            print(f"ROC AUC: {results_eng['metrics']['roc_auc']:.4f}")
            print(f"TPR at 0.01% FPR: {results_eng['metrics']['tpr_at_fpr_0_01']:.4f}")

            print("\nCounts:")
            print(f"True Positives: {len(results_eng['data']['true_positives'])}")
            print(f"False Positives: {len(results_eng['data']['false_positives'])}")
            print(f"True Negatives: {len(results_eng['data']['true_negatives'])}")
            print(f"False Negatives: {len(results_eng['data']['false_negatives'])}")
            print(f"Errors: {results_eng['data']['error_count']}")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = pair['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f"results_eng_{model_name}_{timestamp}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_eng, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")

        if args.dataset in ['ru', 'all']:
            data_dir = "./data"
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            
            for json_file in json_files:
                print(f"\nProcessing file: {json_file}")
                file_path = os.path.join(data_dir, json_file)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
                
                dataset_name = os.path.splitext(json_file)[0]
                
                results_ru = run_ru_dataset(bino, sample_rate=1, data=dataset, max_samples=10000)
                results_ru['model_pair'] = {
                    'observer': pair['observer'],
                    'performer': pair['performer'],
                    'pair_name': pair['name']
                }
                results_ru['dataset_name'] = dataset_name
                
                print(f"\nResults for dataset: {dataset_name}")
                print("\nOverall Metrics:")
                print(f"F1 Score: {results_ru['overall_metrics']['f1_score']:.4f}")
                print(f"ROC AUC: {results_ru['overall_metrics']['roc_auc']:.4f}")
                print(f"TPR at 0.01% FPR: {results_ru['overall_metrics']['tpr_at_fpr_0_01']:.4f}")

                print("\nCounts:")
                print(f"True Positives: {len(results_ru['data']['true_positives'])}")
                print(f"False Positives: {len(results_ru['data']['false_positives'])}")
                print(f"True Negatives: {len(results_ru['data']['true_negatives'])}")
                print(f"False Negatives: {len(results_ru['data']['false_negatives'])}")
                print(f"Errors: {results_ru['data']['error_count']}")

                model_name = pair['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_ru = os.path.join(output_dir, f"results_ru_{dataset_name}_{model_name}_{timestamp}.json")
                with open(output_file_ru, 'w', encoding='utf-8') as f:
                    json.dump(results_ru, f, ensure_ascii=False, indent=2)
                print(f"\nResults saved to: {output_file_ru}")

        bino.free_memory()

if __name__ == "__main__":
    main()