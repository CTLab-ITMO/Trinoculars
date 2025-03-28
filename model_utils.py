import os
import torch
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from NN_classifier.neural_net_t import Neural_Network, DEVICE
from feature_extraction import extract_features
import pandas as pd

def load_model(model_dir='models/neural_network'):
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
    hidden_layers = [128, 96, 64, 32]
    num_classes = len(label_encoder.classes_)
    
    model = Neural_Network(input_size, hidden_layers, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    if imputer is not None:
        try:
            if hasattr(imputer, 'feature_names_in_'):
                print(f"Imputer has {len(imputer.feature_names_in_)} features")
                print(f"First few feature names: {imputer.feature_names_in_[:5]}")
            else:
                print("Warning: Imputer does not have feature_names_in_ attribute")
        except Exception as e:
            print(f"Error checking imputer: {str(e)}")
    
    return model, scaler, label_encoder, imputer

def classify_text(text, model, scaler, label_encoder, imputer=None, scores=None):
    print("Extracting features...")
    features_df, text_analysis = extract_features(text, scores=scores)
    
    if imputer is not None:
        expected_feature_names = imputer.feature_names_in_
    else:
        expected_feature_names = None
    
    if expected_feature_names is not None:
        aligned_features = pd.DataFrame(columns=expected_feature_names)
        
        for col in features_df.columns:
            if col in expected_feature_names:
                aligned_features[col] = features_df[col]
        
        for col in expected_feature_names:
            if col not in aligned_features.columns or aligned_features[col].isnull().all():
                aligned_features[col] = 0
                print(f"Added missing feature: {col}")
        
        features_df = aligned_features
    
    if imputer is None:
        print("Warning: No imputer provided, creating a new one")
        imputer = SimpleImputer(strategy='mean')
        features = imputer.fit_transform(features_df)
    else:
        features = imputer.transform(features_df)
    
    features_scaled = scaler.transform(features)
    
    features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_label = label_encoder.classes_[pred_class]
    
    probs_dict = {label_encoder.classes_[i]: probabilities[0][i].item() for i in range(len(label_encoder.classes_))}
    
    return {
        'predicted_class': predicted_label,
        'probabilities': probs_dict,
        'features': features_df,
        'text_analysis': text_analysis,
        'scores': scores
    } 