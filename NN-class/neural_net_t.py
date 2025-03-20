import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import json
import joblib
import os

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
        'ai': 'Raw AI',
        'human': 'Human',
        'ai+rew': 'Rephrased AI'
    }
    
    if 'source' in df.columns:
        df['label'] = df['source'].map(lambda x: label_mapping.get(x, x))
    else:
        print("Warning: 'source' column not found, using default label")
        df['label'] = 'Unknown'
    
    return df

class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.2):
        super(Neural_Network, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def build_neural_network(input_shape, num_classes, hidden_layers=[64, 32]):
    model = Neural_Network(input_shape, hidden_layers, num_classes).to(DEVICE)
    print(f"Model created with hidden layers {hidden_layers} on device: {DEVICE}")
    return model

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
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/learning_curve.png')
    plt.close()
    print("Learning curve saved to plots/learning_curve.png")

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
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/accuracy_curve.png')
    plt.close()
    print("Accuracy curve saved to plots/accuracy_curve.png")

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

def train_neural_network(directory_path="experiments/results/two_scores_with_long_text_analyze_2048T", 
                          model_config=None, 
                          feature_config=None):
    if model_config is None:
        model_config = {
            'hidden_layers': [128, 96, 64, 32],
            'dropout_rate': 0.1
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
    
    df = load_data_from_json(directory_path)
    
    features_df = select_features(df, feature_config)
    
    print(f"Selected features: {features_df.columns.tolist()}")
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(features_df)
    y = df['label'].values
    
    print(f"Final data size after NaN processing: {X.shape}")
    print(f"Labels distribution: {pd.Series(y).value_counts().to_dict()}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(DEVICE)
    y_val_tensor = torch.LongTensor(y_val).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_classes = len(label_encoder.classes_)
    model = build_neural_network(X_train_scaled.shape[1], num_classes, 
                                 hidden_layers=model_config['hidden_layers'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
            
            _, predicted_val = torch.max(val_outputs.data, 1)
            val_accuracy = (predicted_val == y_val_tensor).sum().item() / len(y_val_tensor)
            val_accuracies.append(val_accuracy)
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    plot_learning_curve(train_losses, val_losses)
    plot_accuracy_curve(train_accuracies, val_accuracies)
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_prob, dim=1).cpu().numpy()
        
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.6f}")
    
    class_names = label_encoder.classes_
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model, scaler, label_encoder, accuracy

def save_model(model, scaler, label_encoder, output_dir='models/neural_network'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, 'nn_model.pt')
    torch.save(model.state_dict(), model_path)
    
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(label_encoder, encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    return model_path, scaler_path, encoder_path

def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if GPU_AVAILABLE:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    plt.switch_backend('agg')
    
    model_config = {
        'hidden_layers': [128, 96, 64, 32],
        'dropout_rate': 0.1
    }
    
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
    
    model, scaler, label_encoder, accuracy = train_neural_network(
        directory_path="experiments/results/two_scores_with_long_text_analyze_2048T",
        model_config=model_config,
        feature_config=feature_config
    )
    
    save_model(model, scaler, label_encoder)

if __name__ == "__main__":
    main()