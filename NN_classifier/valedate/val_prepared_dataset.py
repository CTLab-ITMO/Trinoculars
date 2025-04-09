import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import argparse
import torch
import joblib
from NN_classifier.simple_binary_classifier import Medium_Binary_Network
from NN_classifier.simple_binary_classifier import select_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def map_source_to_label(source):
    label_mapping = {
        'ai': 'AI',
        'human': 'Human',
        'ai+par': 'AI',
        'ai+rew': 'AI'
    }
    return label_mapping.get(source, source)

def load_binary_model(model_dir='models/medium_binary_classifier'):
    model_path = os.path.join(model_dir, 'nn_model.pt')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
    imputer_path = os.path.join(model_dir, 'imputer.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    imputer = None
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
    else:
        print("Warning: Imputer not found, will create a new one during classification")
    
    input_size = scaler.n_features_in_
    
    model = Medium_Binary_Network(input_size, hidden_sizes=[256, 192, 128, 64], dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model, scaler, label_encoder, imputer

def validate_coat_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    dataset = data_dict.get('data', [])
    if not dataset:
        dataset = data_dict
    
    print(f"Found {len(dataset)} texts in dataset")
    
    class_distribution = {}
    for item in dataset:
        cls = item['source']
        class_distribution[cls] = class_distribution.get(cls, 0) + 1
    
    print("Class distribution in dataset:")
    for cls, count in class_distribution.items():
        print(f"  - {cls}: {count} examples ({count/len(dataset)*100:.1f}%)")
    
    print("Loading model...")
    model, scaler, label_encoder, imputer = load_binary_model('models/medium_binary_classifier')
    
    df = pd.DataFrame(dataset)
    
    feature_config = {
        'basic_scores': True, 
        'basic_text_stats': ['total_tokens', 'total_words', 'unique_words', 'stop_words', 'avg_word_length'],
        'morphological': ['pos_distribution', 'unique_lemmas', 'lemma_word_ratio'],
        'syntactic': ['dependencies', 'noun_chunks'],
        'entities': ['total_entities', 'entity_types'],
        'diversity': ['ttr', 'mtld'],
        'structure': ['sentence_count', 'avg_sentence_length', 'question_sentences', 'exclamation_sentences'],
        'readability': ['words_per_sentence', 'syllables_per_word', 'flesh_kincaid_score', 'long_words_percent'],
        'semantic': True
    }
    
    features_df = select_features(df, feature_config)
    
    results = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    if features_df.isnull().values.any():
        print("Warning: Found NaN values in features. Using imputer to fill missing values.")
        if imputer:
            features = imputer.transform(features_df)
        else:
            raise ValueError("Imputer not available to handle missing features")
    else:
        features = features_df.values
    
    features_scaled = scaler.transform(features)
    
    features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
    
    print("Processing texts...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        
        for i, item in enumerate(tqdm(dataset)):
            true_source = item['source']
            true_label = map_source_to_label(true_source)
            
            predicted_idx = predicted[i].item()
            predicted_class = label_encoder.classes_[predicted_idx]
            confidence = probabilities[i][predicted_idx].item()
            
            result_item = {
                'id': item.get('id', i),
                'true_source': true_source,
                'true_label': true_label,
                'predicted_label': predicted_class,
                'confidence': confidence,
                'correct': predicted_class == true_label
            }
            
            if 'score_chat' in item:
                result_item['score_chat'] = item['score_chat']
            if 'score_coder' in item:
                result_item['score_coder'] = item['score_coder']
            
            results.append(result_item)
            true_labels.append(true_label)
            predicted_labels.append(predicted_class)
            confidence_scores.append(confidence)
    
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
    
    print("\n" + "="*50)
    print("COAT VALIDATION RESULTS")
    print("="*50)
    
    print(f"\nTotal texts: {metrics['total_texts']}")
    print(f"Correctly classified: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average confidence: {metrics['avg_confidence']:.4f}")
    
    df_results = pd.DataFrame(results)
    class_accuracy = df_results.groupby('true_label')['correct'].mean()
    class_counts = df_results.groupby('true_label').size()
    
    print("\nAccuracy by class:")
    for label, acc in class_accuracy.items():
        print(f"  - {label}: {acc:.4f} ({class_counts[label]} samples)")
    
    os.makedirs('results', exist_ok=True)
    with open('results/coat_validation_metrics.json', 'w') as f:
        json.dump({
            'overall': metrics,
            'class_report': report
        }, f, indent=4)
    
    print(f"\nResults saved to results/coat_validation_metrics.json")
    
    return metrics, df_results

def main():
    parser = argparse.ArgumentParser(description='Validate binary classifier on COAT dataset')
    parser.add_argument('--dataset', type=str, default='experiments/results/coat/coat_results_20250331_135912_analyzed.json',
                        help='Path to the COAT dataset JSON file')
    args = parser.parse_args()
    
    try:
        validate_coat_dataset(args.dataset)
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()