import json
import matplotlib.pyplot as plt
import numpy as np

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def summarize_datasets_and_sources(data):
    dataset_counts = {}
    ai_human_ratio = {"AI": 0, "Human": 0}
    
    for record in data:
        dataset = record.get("dataset", "Unknown")
        source = record.get("source", "Unknown")
        
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        if "ai" in source.lower():
            ai_human_ratio["AI"] += 1
        else:
            ai_human_ratio["Human"] += 1
    
    return dataset_counts, ai_human_ratio

def print_statistics(dataset_counts, ai_human_ratio):
    print("\nDataset statistics:")
    print("-" * 40)
    for dataset, count in sorted(dataset_counts.items()):
        print(f"{dataset}: {count} records")
    
    print("\nAI/Human content ratio:")
    print("-" * 40)
    total = sum(ai_human_ratio.values())
    for source, count in ai_human_ratio.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{source}: {count} records ({percentage:.1f}%)")

def main():
    filename = r'E:\diplom\Trinoculars\datasets\ru_detection_dataset_two.json'
    data = load_json(filename)
    dataset_counts, ai_human_ratio = summarize_datasets_and_sources(data)
    print_statistics(dataset_counts, ai_human_ratio)

if __name__ == '__main__':
    main()
