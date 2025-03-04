from binoculars import Binoculars
import os
import requests
import pyarrow.parquet as pq
import random
import sys
import json
import pandas as pd
from sklearn import metrics
import numpy as np


def run_eng_dataset(bino, sample_rate, max_samples=2000):
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    error_count = 0
    check_counter = 0

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    file_urls = [
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00000-of-00007-bc5952582e004d67.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00001-of-00007-71c80017bc45f30d.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00002-of-00007-ee2d43f396e78fbc.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00003-of-00007-529931154b42b51d.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00004-of-00007-b269dc49374a2c0b.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00005-of-00007-3dce5e05ddbad789.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00006-of-00007-3d8a471ba0cf1c8d.parquet",
    ]

    filenames = []

    for url in file_urls:
        file_name = url.split("/")[-1]
        local_path = os.path.join(data_dir, file_name)
        filenames.append(local_path)

        if not os.path.exists(local_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{file_name} download complete!")
        else:
            print(f"{file_name} already exists. Skipping download.")

    for filename in filenames:
        parquet_file = pq.ParquetFile(filename)
        print()
        print(f"Loaded {filename}")

        for batch in parquet_file.iter_batches(batch_size=1000):
            if check_counter >= max_samples:
                break
                
            batch_pandas = batch.to_pandas()

            for index, row in batch_pandas.iterrows():
                if random.random() < sample_rate:
                    try:
                        prediction = bino.predict(row.text)
                        if isinstance(prediction, (int, float)) and prediction > 1:
                            error_count += 1
                            continue
                        predicted_ai = (prediction == "Most likely AI-generated")
                    except Exception as e:
                        print(f"\nError predicting for text: {row.text}, Error: {e}")
                        error_count += 1
                        continue

                    actually_ai = (row.source == "ai")

                    if predicted_ai and actually_ai:
                        true_positives.append(row.text)
                    elif predicted_ai and not actually_ai:
                        false_positives.append(row.text)
                    elif not predicted_ai and actually_ai:
                        false_negatives.append(row.text)
                    elif not predicted_ai and not actually_ai:
                        true_negatives.append(row.text)

                    check_counter += 1
                    if check_counter % 10 == 0:
                        sys.stdout.write(f"\rProcessed: {check_counter} items")
                        sys.stdout.flush()

    results_data = {
        'text': true_positives + false_positives + true_negatives + false_negatives,
        'pred': [1] * len(true_positives) + [1] * len(false_positives) + 
                [0] * len(true_negatives) + [0] * len(false_negatives),
        'class': [1] * len(true_positives) + [0] * len(false_positives) + 
                 [0] * len(true_negatives) + [1] * len(false_negatives)
    }
    score_df = pd.DataFrame(results_data)

    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score_df["pred"], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    
    tn = len(true_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tp = len(true_positives)
    
    tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr_value = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'metrics': {
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'tpr_at_fpr_0_01': tpr_at_fpr_0_01,
            'tpr': tpr_value,
            'fpr': fpr_value,
            'tnr': tnr_value,
            'fnr': fnr_value
        },
        'data': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'error_count': error_count,
            'check_counter': check_counter
        }
    }



def run_ru_dataset(bino, data):
    results = []
    error_count = 0
    check_counter = 0
    
    dataset_results = {}

    for row in data:
        try:
            score = bino.compute_score(row["text"])
                    
        except Exception as e:
            print(f"\nError computing score for text: {row['text']}, Error: {e}")
            error_count += 1
            continue

        dataset_name = row.get("dataset", "unknown")
        
        example_data = {
            "text": row["text"],
            "source": row["source"],
            "dataset": dataset_name,
            "score": score
        }
        
        results.append(example_data)
        
        if dataset_name not in dataset_results:
            dataset_results[dataset_name] = []
            
        dataset_results[dataset_name].append(example_data)
        check_counter += 1
        if check_counter % 10 == 0:
            sys.stdout.write(f"\rProcessed: {check_counter} items")
            sys.stdout.flush()
    
    dataset_stats = {}
    for dataset_name, examples in dataset_results.items():
        ai_examples = [ex for ex in examples if ex["source"] == "ai"]
        human_examples = [ex for ex in examples if ex["source"] != "ai"]
        
        dataset_stats[dataset_name] = {
            "total_examples": len(examples),
            "ai_examples": len(ai_examples),
            "human_examples": len(human_examples),
            "avg_score": sum(ex["score"] for ex in examples) / len(examples) if examples else 0,
            "avg_ai_score": sum(ex["score"] for ex in ai_examples) / len(ai_examples) if ai_examples else 0,
            "avg_human_score": sum(ex["score"] for ex in human_examples) / len(human_examples) if human_examples else 0
        }

    all_ai_examples = [ex for ex in results if ex["source"] == "ai"]
    all_human_examples = [ex for ex in results if ex["source"] != "ai"]
    
    overall_stats = {
        "total_examples": len(results),
        "ai_examples": len(all_ai_examples),
        "human_examples": len(all_human_examples),
        "avg_score": sum(ex["score"] for ex in results) / len(results) if results else 0,
        "avg_ai_score": sum(ex["score"] for ex in all_ai_examples) / len(all_ai_examples) if all_ai_examples else 0,
        "avg_human_score": sum(ex["score"] for ex in all_human_examples) / len(all_human_examples) if all_human_examples else 0
    }

    return {
        'overall_stats': overall_stats,
        'dataset_stats': dataset_stats,
        'data': results,
        'error_count': error_count,
        'check_counter': check_counter
    }