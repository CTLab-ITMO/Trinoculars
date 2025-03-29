import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os   
import argparse
from model_utils import load_model, classify_text
from binoculars_utils import initialize_binoculars, compute_scores

def map_source_to_label(source):
    label_mapping = {
        'ai': 'Raw AI',
        'human': 'Human',
        'ai+par': 'Rephrased AI'
    }
    return label_mapping.get(source, source)

def validate_on_dataset(dataset_path, limit=None, compute_binoculars=False, random_seed=42):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} texts in dataset")
    
    if limit and limit < len(dataset):
        np.random.seed(random_seed)
        
        class_examples = {}
        for i, item in enumerate(dataset):
            source = item['source']
            if source not in class_examples:
                class_examples[source] = []
            class_examples[source].append(i)
        
        total_samples = len(dataset)
        samples_per_class = {}
        for cls, examples in class_examples.items():
            class_proportion = len(examples) / total_samples
            samples_per_class[cls] = max(1, int(limit * class_proportion))
        
        total_selected = sum(samples_per_class.values())
        if total_selected < limit:
            remaining = limit - total_selected
            for cls in sorted(class_examples.keys(), 
                              key=lambda c: len(class_examples[c]) / total_samples,
                              reverse=True):
                if remaining <= 0:
                    break
                samples_per_class[cls] += 1
                remaining -= 1
        elif total_selected > limit:
            excess = total_selected - limit
            for cls in sorted(class_examples.keys(), 
                              key=lambda c: len(class_examples[c]),
                              reverse=True):
                if excess <= 0:
                    break
                reduction = min(excess, samples_per_class[cls] - 1)
                samples_per_class[cls] -= reduction
                excess -= reduction
        
        selected_indices = []
        for cls, count in samples_per_class.items():
            if len(class_examples[cls]) <= count:
                selected_indices.extend(class_examples[cls])
            else:
                selected = np.random.choice(class_examples[cls], size=count, replace=False)
                selected_indices.extend(selected)
        
        dataset = [dataset[i] for i in selected_indices]
        print(f"Selected {len(dataset)} examples with stratified sampling")
        
        class_distribution = {}
        for item in dataset:
            cls = item['source']
            class_distribution[cls] = class_distribution.get(cls, 0) + 1
        
        print("Class distribution in sample:")
        for cls, count in class_distribution.items():
            print(f"  - {cls}: {count} examples ({count/len(dataset)*100:.1f}%)")
    
    bino_chat = None
    bino_coder = None
    if compute_binoculars:
        print("Initializing Binoculars models...")
        bino_chat, bino_coder = initialize_binoculars()
    
    print("Loading model...")
    model, scaler, label_encoder, imputer = load_model()
    
    results = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    chat_scores = []
    coder_scores = []
    
    print("Processing texts...")
    for item in tqdm(dataset):
        text = item['text']
        true_source = item['source']
        true_label = map_source_to_label(true_source)
        
        try:
            scores = None
            if compute_binoculars:
                scores = compute_scores(text, bino_chat, bino_coder)
            
            classification = classify_text(text, model, scaler, label_encoder, imputer=imputer, scores=scores)
            
            predicted_class = classification['predicted_class']
            probabilities = classification['probabilities']
            confidence = probabilities[predicted_class]
            
            result_item = {
                'id': item.get('id', ''),
                'text_preview': text[:100] + '...',
                'true_source': true_source,
                'true_label': true_label,
                'predicted_label': predicted_class,
                'confidence': confidence,
                'correct': predicted_class == true_label
            }
            
            if scores:
                if 'score_chat' in scores:
                    result_item['score_chat'] = scores['score_chat']
                    chat_scores.append(scores['score_chat'])
                
                if 'score_coder' in scores:
                    result_item['score_coder'] = scores['score_coder']
                    coder_scores.append(scores['score_coder'])
            
            results.append(result_item)
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_class)
            confidence_scores.append(confidence)
            
        except Exception as e:
            print(f"Error processing text {item.get('id', '')}: {str(e)}")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_texts': len(results),
        'correct_predictions': sum(1 for r in results if r['correct']),
        'avg_confidence': np.mean(confidence_scores)
    }

    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/validation_confusion_matrix.png')
    plt.close()
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('validation_results.csv', index=False)
    
    with open('validation_metrics.json', 'w') as f:
        json.dump({
            'overall': metrics,
            'class_report': report
        }, f, indent=4)
    
    if compute_binoculars:
        if bino_chat:
            bino_chat.free_memory()
        if bino_coder:
            bino_coder.free_memory()
    
    return metrics, df_results

def display_results(metrics, df_results):
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    print(f"\nTotal texts: {metrics['total_texts']}")
    print(f"Correctly classified: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average confidence: {metrics['avg_confidence']:.4f}")
    
    class_accuracy = df_results.groupby('true_label')['correct'].mean()
    class_counts = df_results.groupby('true_label').size()
    
    print("\nAccuracy by class:")
    for label, acc in class_accuracy.items():
        print(f"  - {label}: {acc:.4f} ({class_counts[label]} samples)")
    
    wrong_predictions = df_results[~df_results['correct']]
    if not wrong_predictions.empty:
        high_conf_errors = wrong_predictions.sort_values(by='confidence', ascending=False).head(5)
        
        print("\nTop 5 most confident incorrect predictions:")
        for _, row in high_conf_errors.iterrows():
            print(f"  ID: {row['id']}, True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['confidence']:.4f}")
            
            if 'score_chat' in row:
                print(f"  score_chat: {row['score_chat']:.4f}")
            if 'score_coder' in row:
                print(f"  score_coder: {row['score_coder']:.4f}")
                
            print(f"  Text preview: {row['text_preview']}")
            print()
    
    print("\nResults saved to validation_results.csv")
    print("Metrics saved to validation_metrics.json")
    print("Confusion matrix saved to plots/validation_confusion_matrix.png")

def main():
    parser = argparse.ArgumentParser(description='Validate model on a dataset')
    parser.add_argument('--dataset', type=str, default='datasets/ru_detection_dataset.json',
                        help='Path to the dataset JSON file')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit the number of texts to process (for testing)')
    parser.add_argument('--compute-binoculars', action='store_true',
                        help='Compute Binoculars metrics (score_chat and score_coder)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling examples')
    args = parser.parse_args()
    
    metrics, df_results = validate_on_dataset(
        args.dataset, 
        args.limit,
        compute_binoculars=args.compute_binoculars,
        random_seed=args.seed
    )
    display_results(metrics, df_results)

if __name__ == "__main__":
    main() 