import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

THRESHOLD = 0.9015310749276843

def calculate_metrics(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    ground_truth = []
    chat_predictions = []
    coder_predictions = []
    
    human_indices = []
    ai_indices = []
    
    for i, item in enumerate(data):
        is_human = 1 if item['source'] == 'human' else 0
        ground_truth.append(is_human)
        
        if is_human:
            human_indices.append(i)
        else:
            ai_indices.append(i)
        
        chat_pred = 1 if item['score_chat'] > THRESHOLD else 0
        coder_pred = 1 if item['score_coder'] > THRESHOLD else 0
        
        chat_predictions.append(chat_pred)
        coder_predictions.append(coder_pred)
    
    ground_truth = np.array(ground_truth)
    chat_predictions = np.array(chat_predictions)
    coder_predictions = np.array(coder_predictions)
    
    print("=== OVERALL METRICS ===")
    
    chat_f1 = f1_score(ground_truth, chat_predictions)
    chat_precision = precision_score(ground_truth, chat_predictions)
    chat_recall = recall_score(ground_truth, chat_predictions)
    chat_cm = confusion_matrix(ground_truth, chat_predictions)
    
    coder_f1 = f1_score(ground_truth, coder_predictions)
    coder_precision = precision_score(ground_truth, coder_predictions)
    coder_recall = recall_score(ground_truth, coder_predictions)
    coder_cm = confusion_matrix(ground_truth, coder_predictions)
    
    print("Chat Model Metrics:")
    print(f"F1 Score: {chat_f1:.4f}")
    print(f"Precision: {chat_precision:.4f}")
    print(f"Recall: {chat_recall:.4f}")
    print("Confusion Matrix:")
    print(chat_cm)
    print("\n")
    
    print("Coder Model Metrics:")
    print(f"F1 Score: {coder_f1:.4f}")
    print(f"Precision: {coder_precision:.4f}")
    print(f"Recall: {coder_recall:.4f}")
    print("Confusion Matrix:")
    print(coder_cm)
    print("\n")
    
    print("=== HUMAN TEXTS ONLY ===")
    if human_indices:
        chat_human_accuracy = np.sum(chat_predictions[human_indices] == 1) / len(human_indices)
        coder_human_accuracy = np.sum(coder_predictions[human_indices] == 1) / len(human_indices)
        
        print(f"Total human texts: {len(human_indices)}")
        print(f"Chat model correctly identified: {chat_human_accuracy:.4f} ({int(chat_human_accuracy * len(human_indices))}/{len(human_indices)})")
        print(f"Coder model correctly identified: {coder_human_accuracy:.4f} ({int(coder_human_accuracy * len(human_indices))}/{len(human_indices)})")
    else:
        print("No human texts found in the dataset")
    print("\n")
    
    print("=== AI TEXTS ONLY ===")
    if ai_indices:
        chat_ai_accuracy = np.sum(chat_predictions[ai_indices] == 0) / len(ai_indices)
        coder_ai_accuracy = np.sum(coder_predictions[ai_indices] == 0) / len(ai_indices)
        
        print(f"Total AI texts: {len(ai_indices)}")
        print(f"Chat model correctly identified: {chat_ai_accuracy:.4f} ({int(chat_ai_accuracy * len(ai_indices))}/{len(ai_indices)})")
        print(f"Coder model correctly identified: {coder_ai_accuracy:.4f} ({int(coder_ai_accuracy * len(ai_indices))}/{len(ai_indices)})")
    else:
        print("No AI texts found in the dataset")
    
    return {
        "overall": {
            "chat": {"f1": chat_f1, "precision": chat_precision, "recall": chat_recall},
            "coder": {"f1": coder_f1, "precision": coder_precision, "recall": coder_recall}
        },
        "human_texts": {
            "chat": chat_human_accuracy if human_indices else None,
            "coder": coder_human_accuracy if human_indices else None
        },
        "ai_texts": {
            "chat": chat_ai_accuracy if ai_indices else None,
            "coder": coder_ai_accuracy if ai_indices else None
        }
    }

if __name__ == "__main__":
    json_file_path = "experiments/results/coat/coat_results_20250331_135912.json"
    calculate_metrics(json_file_path)