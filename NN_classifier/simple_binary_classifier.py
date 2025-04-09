import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import json
import joblib
import os
import seaborn as sns
from scipy import stats
import time
import argparse

def setup_gpu():
    if torch.cuda.is_available():
        return True
    else:
        print("No GPUs found. Using CPU.")
        return False

GPU_AVAILABLE = setup_gpu()
DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')

def load_data_from_json(directory_path):
    if os.path.isfile(directory_path):
        directory = os.path.dirname(directory_path)
    else:
        directory = directory_path
        
    print(f"Loading JSON files from directory: {directory}")
    
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.endswith('.json') and os.path.isfile(os.path.join(directory, f))]
    
    if not json_files:
        raise ValueError(f"No JSON files found in directory {directory}")
    
    print(f"Found {len(json_files)} JSON files")
    
    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            if 'data' in data_dict:
                all_data.extend(data_dict['data'])
            else:
                print(f"Warning: 'data' key not found in {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
    
    if not all_data:
        raise ValueError("Failed to load data from JSON files")
        
    df = pd.DataFrame(all_data)
    
    label_mapping = {
        'ai': 'AI',
        'human': 'Human',
        'ai+rew': 'AI',
    }
    
    if 'source' in df.columns:
        df['label'] = df['source'].map(lambda x: label_mapping.get(x, x))
    else:
        print("Warning: 'source' column not found, using default label")
        df['label'] = 'Unknown'
    
    valid_labels = ['AI', 'Human']
    df = df[df['label'].isin(valid_labels)]
    
    print(f"Filtered to {len(df)} examples with labels: {valid_labels}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

class Medium_Binary_Network(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout=0.3):
        super(Medium_Binary_Network, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 2))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def cross_validate_simple_classifier(directory_path="experiments/results/two_scores_with_long_text_analyze_2048T", 
                                   feature_config=None,
                                   n_splits=5,
                                   random_state=42,
                                   epochs=100,
                                   hidden_sizes=[256, 128, 64, 32],
                                   dropout=0.3,
                                   early_stopping_patience=10):
    print("\n" + "="*50)
    print("MEDIUM BINARY CLASSIFIER CROSS-VALIDATION")
    print("="*50)
    
    if feature_config is None:
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
    
    df = load_data_from_json(directory_path)
    
    features_df = select_features(df, feature_config)
    print(f"Selected {len(features_df.columns)} features")
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(features_df)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'].values)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    fold_models = []
    
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    
    all_y_true = []
    all_y_pred = []
    
    best_fold_score = -1
    best_fold_index = -1
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = Medium_Binary_Network(X_train_scaled.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(DEVICE)
        print(f"Model created with {len(hidden_sizes)} hidden layers: {hidden_sizes}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                val_losses.append(val_loss.item())
                
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = torch.sum(val_preds == y_test_tensor).item() / len(y_test_tensor)
                val_accs.append(val_acc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            print("Loaded best model weights")
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            test_acc = torch.sum(predicted == y_test_tensor).item() / len(y_test_tensor)
            
            y_test_np = y_test
            predicted_np = predicted.cpu().numpy()
            
            all_y_true.extend(y_test_np)
            all_y_pred.extend(predicted_np)
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, predicted_np, average='weighted')
            
            fold_metric = {
                'fold': fold + 1,
                'accuracy': float(test_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'val_loss': float(best_val_loss)
            }
            
            fold_metrics.append(fold_metric)
            
            fold_models.append({
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'imputer': imputer,
                'score': test_acc
            })
            
            if test_acc > best_fold_score:
                best_fold_score = test_acc
                best_fold_index = fold
            
            all_train_losses.extend(train_losses)
            all_val_losses.extend(val_losses)
            all_train_accs.extend(train_accs)
            all_val_accs.extend(val_accs)
            
            print(f"Fold {fold+1} Results:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
    
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted'
    )
    
    fold_accuracies = [metrics['accuracy'] for metrics in fold_metrics]
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(n_splits)
    ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(n_splits)
    
    plot_learning_curve(all_train_losses, all_val_losses)
    plot_accuracy_curve(all_train_accs, all_val_accs)
    
    class_names = label_encoder.classes_
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Binary Classification Confusion Matrix (Cross-Validation)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')    
    os.makedirs('plots/binary', exist_ok=True)
    plt.savefig('plots/binary/confusion_matrix_medium.png')
    plt.close()
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    
    print(f"\nBest Fold: {best_fold_index + 1} (Accuracy: {fold_metrics[best_fold_index]['accuracy']:.4f})")
    
    best_model_data = fold_models[best_fold_index]
    
    results = {
        'fold_metrics': fold_metrics,
        'overall': {
            'accuracy': float(overall_accuracy),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1)
        },
        'cross_validation': {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'confidence_interval_95': [float(ci_lower), float(ci_upper)]
        },
        'best_fold': {
            'fold': best_fold_index + 1,
            'accuracy': float(fold_metrics[best_fold_index]['accuracy'])
        },
        'model_config': {
            'hidden_sizes': hidden_sizes,
            'dropout': dropout
        }
    }
    
    output_dir = 'models/medium_binary_classifier'
    save_paths = save_binary_model(best_model_data, results, output_dir=output_dir)
    
    return best_model_data, results, save_paths

def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('plots/binary', exist_ok=True)
    plt.savefig('plots/binary/learning_curve.png')
    plt.close()
    print("Learning curve saved to plots/binary/learning_curve.png")

def plot_accuracy_curve(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'm-', label='Validation Accuracy')
    
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.ylim(0, 1.0)
    
    os.makedirs('plots/binary', exist_ok=True)
    plt.savefig('plots/binary/accuracy_curve.png')
    plt.close()
    print("Accuracy curve saved to plots/binary/accuracy_curve.png")

def select_features(df, feature_config):
    features_df = pd.DataFrame()
    
    if feature_config.get('basic_scores', True):
        if 'score_chat' in df.columns:
            features_df['score_chat'] = df['score_chat']
        if 'score_coder' in df.columns:
            features_df['score_coder'] = df['score_coder']
    
    if 'text_analysis' in df.columns:
        if feature_config.get('basic_text_stats'):
            for feature in feature_config['basic_text_stats']:
                features_df[f'basic_{feature}'] = df['text_analysis'].apply(
                    lambda x: x.get('basic_stats', {}).get(feature, 0) if isinstance(x, dict) else 0
                )
        
        if feature_config.get('morphological'):
            for feature in feature_config['morphological']:
                if feature == 'pos_distribution':
                    pos_types = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'DET', 'ADP', 'PRON', 'CCONJ', 'SCONJ']
                    for pos in pos_types:
                        features_df[f'pos_{pos}'] = df['text_analysis'].apply(
                            lambda x: x.get('morphological_analysis', {}).get('pos_distribution', {}).get(pos, 0) 
                            if isinstance(x, dict) else 0
                        )
                else:
                    features_df[f'morph_{feature}'] = df['text_analysis'].apply(
                        lambda x: x.get('morphological_analysis', {}).get(feature, 0) if isinstance(x, dict) else 0
                    )
        
        if feature_config.get('syntactic'):
            for feature in feature_config['syntactic']:
                if feature == 'dependencies':
                    dep_types = ['nsubj', 'obj', 'amod', 'nmod', 'ROOT', 'punct', 'case']
                    for dep in dep_types:
                        features_df[f'dep_{dep}'] = df['text_analysis'].apply(
                            lambda x: x.get('syntactic_analysis', {}).get('dependencies', {}).get(dep, 0) 
                            if isinstance(x, dict) else 0
                        )
                else:
                    features_df[f'synt_{feature}'] = df['text_analysis'].apply(
                        lambda x: x.get('syntactic_analysis', {}).get(feature, 0) if isinstance(x, dict) else 0
                    )
        
        if feature_config.get('entities'):
            for feature in feature_config['entities']:
                if feature == 'entity_types':
                    entity_types = ['PER', 'LOC', 'ORG']
                    for ent in entity_types:
                        features_df[f'ent_{ent}'] = df['text_analysis'].apply(
                            lambda x: x.get('named_entities', {}).get('entity_types', {}).get(ent, 0) 
                            if isinstance(x, dict) else 0
                        )
                else:
                    features_df[f'ent_{feature}'] = df['text_analysis'].apply(
                        lambda x: x.get('named_entities', {}).get(feature, 0) if isinstance(x, dict) else 0
                    )
        
        if feature_config.get('diversity'):
            for feature in feature_config['diversity']:
                features_df[f'div_{feature}'] = df['text_analysis'].apply(
                    lambda x: x.get('lexical_diversity', {}).get(feature, 0) if isinstance(x, dict) else 0
                )
        
        if feature_config.get('structure'):
            for feature in feature_config['structure']:
                features_df[f'struct_{feature}'] = df['text_analysis'].apply(
                    lambda x: x.get('text_structure', {}).get(feature, 0) if isinstance(x, dict) else 0
                )
        
        if feature_config.get('readability'):
            for feature in feature_config['readability']:
                features_df[f'read_{feature}'] = df['text_analysis'].apply(
                    lambda x: x.get('readability', {}).get(feature, 0) if isinstance(x, dict) else 0
                )
        
        if feature_config.get('semantic'):
            features_df['semantic_coherence'] = df['text_analysis'].apply(
                lambda x: x.get('semantic_coherence', {}).get('avg_coherence_score', 0) if isinstance(x, dict) else 0
            )
    
    print(f"Generated {len(features_df.columns)} features")
    return features_df

def augment_text_features(features_df, num_augmentations=5, noise_factor=0.05):
    augmented_dfs = [features_df]
    
    for i in range(num_augmentations):
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        augmented_df = features_df.copy()
        for col in numeric_cols:
            augmented_df[col] = augmented_df[col].astype(float)
        
        noise = augmented_df[numeric_cols] * np.random.normal(0, noise_factor, size=augmented_df[numeric_cols].shape)
        augmented_df[numeric_cols] += noise
        augmented_dfs.append(augmented_df)
    
    return pd.concat(augmented_dfs, ignore_index=True)

def cross_validate_binary_classifier(directory_path="experiments/results/two_scores_with_long_text_analyze_2048T", 
                                    model_config=None, 
                                    feature_config=None,
                                    n_splits=5,
                                    random_state=42,
                                    epochs=100,
                                    early_stopping_patience=10,
                                    use_augmentation=True,
                                    num_augmentations=2,
                                    noise_factor=0.05):
    if model_config is None:
        model_config = {
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.3
        }
    
    if feature_config is None:
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
    
    print("\n" + "="*50)
    print("BINARY CLASSIFIER CROSS-VALIDATION")
    print("="*50)
    
    df = load_data_from_json(directory_path)
    
    features_df = select_features(df, feature_config)
    print(f"Selected features: {features_df.columns.tolist()}")
    
    imputer = SimpleImputer(strategy='mean')
    
    if use_augmentation:
        print(f"Augmenting data with {num_augmentations} copies (noise factor: {noise_factor})...")
        original_size = len(features_df)
        features_df_augmented = augment_text_features(features_df, 
                                                    num_augmentations=num_augmentations, 
                                                    noise_factor=noise_factor)
        y_augmented = np.tile(df['label'].values, num_augmentations + 1)
        print(f"Data size increased from {original_size} to {len(features_df_augmented)}")
        
        X = imputer.fit_transform(features_df_augmented)
        y = y_augmented
    else:
        X = imputer.fit_transform(features_df)
        y = df['label'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Data size: {X.shape}")
    print(f"Labels distribution: {pd.Series(y).value_counts().to_dict()}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    fold_models = []
    all_y_true = []
    all_y_pred = []
    all_y_scores = []
    
    best_fold_score = -1
    best_fold_index = -1
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    num_avg_epochs = 5
    saved_weights = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        num_classes = len(label_encoder.classes_)
        model = build_neural_network(X_train_scaled.shape[1], num_classes, 
                                     hidden_layers=model_config['hidden_layers'])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        saved_weights = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                val_losses.append(val_loss.item())
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if epoch >= epochs - num_avg_epochs:
                saved_weights.append(model.state_dict().copy())
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if len(saved_weights) > 0:
            print(f"Averaging weights from last {len(saved_weights)} epochs...")
            avg_state_dict = saved_weights[0].copy()
            for key in avg_state_dict.keys():
                if epoch >= epochs - num_avg_epochs:
                    for i in range(1, len(saved_weights)):
                        avg_state_dict[key] += saved_weights[i][key]
                    avg_state_dict[key] /= len(saved_weights)
            
            model.load_state_dict(avg_state_dict)
            print("Model loaded with averaged weights")
        elif best_model_state:
            model.load_state_dict(best_model_state)
            print("Model loaded with best validation weights")
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            predicted_np = predicted.cpu().numpy()
            
            probabilities = torch.softmax(test_outputs, dim=1)
            pos_scores = probabilities[:, 1].cpu().numpy()
            
            all_y_true.extend(y_test)
            all_y_pred.extend(predicted_np)
            all_y_scores.extend(pos_scores)
        
        fold_acc = accuracy_score(y_test, predicted_np)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted_np, average='weighted')
        
        try:
            fold_auc = roc_auc_score(y_test, pos_scores)
        except:
            fold_auc = 0.0
            print("Warning: Could not compute AUC")
        
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': float(fold_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(fold_auc),
            'best_val_loss': float(best_val_loss)
        })
        
        fold_models.append({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'imputer': imputer,
            'score': fold_acc
        })
        
        if fold_acc > best_fold_score:
            best_fold_score = fold_acc
            best_fold_index = fold
            
        print(f"Fold {fold+1} Results:")
        print(f"  Accuracy: {fold_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if fold_auc > 0:
            print(f"  AUC: {fold_auc:.4f}")
    
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted'
    )
    
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_scores)
    except:
        overall_auc = 0.0
        print("Warning: Could not compute overall AUC")
    
    fold_accuracies = [metrics['accuracy'] for metrics in fold_metrics]
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(n_splits)
    ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(n_splits)
    
    class_counts = np.bincount(y_encoded)
    baseline_accuracy = np.max(class_counts) / len(y_encoded)
    most_frequent_class = np.argmax(class_counts)
    
    t_stat, p_value = stats.ttest_1samp(fold_accuracies, baseline_accuracy)
    
    best_model_data = fold_models[best_fold_index]
    
    class_names = label_encoder.classes_
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Binary Classification Confusion Matrix (Cross-Validation)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs('plots/binary', exist_ok=True)
    plt.savefig('plots/binary/confusion_matrix_cv.png')
    plt.close()
    
    if overall_auc > 0:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {overall_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('plots/binary/roc_curve.png')
        plt.close()
    
    results = {
        'fold_metrics': fold_metrics,
        'overall': {
            'accuracy': float(overall_accuracy),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1),
            'auc': float(overall_auc) if overall_auc > 0 else None
        },
        'cross_validation': {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'confidence_interval_95': [float(ci_lower), float(ci_upper)],
            'baseline_accuracy': float(baseline_accuracy),
            'most_frequent_class': str(label_encoder.inverse_transform([most_frequent_class])[0]),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'statistically_significant': "yes" if p_value < 0.05 else "no"
        },
        'best_fold': {
            'fold': best_fold_index + 1,
            'accuracy': float(fold_metrics[best_fold_index]['accuracy'])
        }
    }
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f} (most frequent class: {label_encoder.inverse_transform([most_frequent_class])[0]})")
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("The model is significantly better than the baseline (p < 0.05)")
    else:
        print("The model is NOT significantly better than the baseline (p >= 0.05)")
    
    print(f"\nBest Fold: {best_fold_index + 1} (Accuracy: {fold_metrics[best_fold_index]['accuracy']:.4f})")
    
    return best_model_data, results

def save_binary_model(model_data, results, output_dir='models/binary_classifier'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, 'nn_model.pt')
    torch.save(model_data['model'].state_dict(), model_path)
    
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(model_data['scaler'], scaler_path)
    
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(model_data['label_encoder'], encoder_path)
    
    imputer_path = os.path.join(output_dir, 'imputer.joblib')
    joblib.dump(model_data['imputer'], imputer_path)
    
    results_path = os.path.join(output_dir, 'cv_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Binary model saved to {model_path}")
    print(f"CV results saved to {results_path}")
    
    return {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'encoder_path': encoder_path,
        'imputer_path': imputer_path,
        'results_path': results_path
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Neural Network Classifier (Human vs AI) with Cross-Validation')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs per fold')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    return parser.parse_args()

def main():
    print("\n" + "="*50)
    print("MEDIUM BINARY CLASSIFIER")
    print("="*50 + "\n")
    
    args = parse_args()
    
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if GPU_AVAILABLE:
        torch.cuda.manual_seed_all(seed)
    
    plt.switch_backend('agg')
    
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
    
    model_data, results, save_paths = cross_validate_simple_classifier(
        directory_path="experiments/results/two_scores_with_long_text_analyze_2048T",
        feature_config=feature_config,
        n_splits=5,
        random_state=seed,
        epochs=150,
        hidden_sizes=[256, 192, 128, 64],
        dropout=0.3,
        early_stopping_patience=15
    )
    
    print("\nTraining completed.")
    print(f"Medium binary classifier saved to {save_paths['model_path']}")

if __name__ == "__main__":
    main() 
