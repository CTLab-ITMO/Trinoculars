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

def map_source_to_label(source):
    label_mapping = {
        'ai': 'Raw AI',
        'human': 'Human',
        'ai+rew': 'Rephrased AI'
    }
    return label_mapping.get(source, source)

def validate_on_dataset(dataset_path, limit=None):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} texts in dataset")
    
    if limit:
        dataset = dataset[:limit]
        print(f"Limited to {limit} texts for testing")
    
    print("Loading model...")
    model, scaler, label_encoder, imputer = load_model()
    
    results = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    print("Processing texts...")
    for item in tqdm(dataset):
        text = item['text']
        true_source = item['source']
        true_label = map_source_to_label(true_source)
        
        try:
            classification = classify_text(text, model, scaler, label_encoder, imputer=imputer)
            
            predicted_class = classification['predicted_class']
            probabilities = classification['probabilities']
            confidence = probabilities[predicted_class]
            
            results.append({
                'id': item.get('id', ''),
                'text_preview': text[:100] + '...',
                'true_source': true_source,
                'true_label': true_label,
                'predicted_label': predicted_class,
                'confidence': confidence,
                'correct': predicted_class == true_label
            })
            
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
    args = parser.parse_args()
    
    metrics, df_results = validate_on_dataset(args.dataset, args.limit)
    display_results(metrics, df_results)

if __name__ == "__main__":
    main() 