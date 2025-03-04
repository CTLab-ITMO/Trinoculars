import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import json
import joblib
import os

def load_data_from_json(json_path='models_test/score_res/mer.json'):
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    df = pd.DataFrame(data_dict['data'])
    
    label_mapping = {
        'ai': 'Raw AI',
        'ai+rew': 'Raw AI',
        'human': 'Human'
    }
    
    df = df.rename(columns={'score1': 's1', 'score2': 's2'})
    
    df = df[df['source'].isin(['ai', 'human', 'ai+rew'])]
    
    df['label'] = df['source'].map(lambda x: label_mapping.get(x, x))
    
    print(f"Initial data size: {df.shape}")
    print(f"Number of NaNs in s1: {df['s1'].isna().sum()}")
    print(f"Number of NaNs in s2: {df['s2'].isna().sum()}")
    
    features_df = df[['s1', 's2']]
    labels = df['label'].values
    
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features_df)
    
    print(f"Data size after NaN processing: {features.shape}")
    
    return features, labels, df

def train_logistic_regression():
    X, y, _ = load_data_from_json()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(solver='lbfgs', 
                              max_iter=1000, 
                              random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=model.classes_,
               yticklabels=model.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')
    plt.tight_layout()
    
    visualize_decision_boundary(X_test_scaled, y_test, model, scaler)
    plot_roc_curve(X_test_scaled, y_test, model)
    
    print("\nLogistic regression coefficients:")
    print(f"Coefficient for score1: {model.coef_[0][0]:.4f}")
    print(f"Coefficient for score2: {model.coef_[0][1]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    return model, scaler

def visualize_decision_boundary(X, y, model, scaler, plot_step=0.02):
    X_original = scaler.inverse_transform(X)
    
    x_min, x_max = X_original[:, 0].min() - 0.1, X_original[:, 0].max() + 0.1
    y_min, y_max = X_original[:, 1].min() - 0.1, X_original[:, 1].max() + 0.1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    
    Z = model.predict(grid_points_scaled)
    Z = np.array([list(model.classes_).index(label) for label in Z])
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    markers = ['o', 's']
    colors = ['blue', 'red']
    
    for i, label in enumerate(np.unique(y)):
        idx = (y == label)
        plt.scatter(X_original[idx, 0], X_original[idx, 1], 
                   c=colors[i], marker=markers[i], label=label, edgecolor='k')
    
    plt.xlabel('Feature s1 (score1)')
    plt.ylabel('Feature s2 (score2)')
    plt.title('Logistic regression decision boundary')
    plt.legend()
    plt.tight_layout()

def plot_roc_curve(X_test, y_test, model):
    y_test_binary = (y_test == model.classes_[1]).astype(int)
    y_score = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()

def visualize_data_distribution(X, y):
    df = pd.DataFrame(X, columns=['score1', 'score2'])
    df['label'] = y
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df)
    plt.title('Number of samples in each class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    sns.boxplot(x='label', y='score1', data=df)
    plt.title('Distribution of score1 by class')
    plt.xlabel('Class')
    plt.ylabel('score1')
    
    plt.subplot(2, 1, 2)
    sns.boxplot(x='label', y='score2', data=df)
    plt.title('Distribution of score2 by class')
    plt.xlabel('Class')
    plt.ylabel('score2')
    
    plt.tight_layout()
    
    plt.figure(figsize=(10, 8))
    
    color_map = {'Human': 'blue', 'Raw AI': 'red'}
    
    sns.scatterplot(x='score1', y='score2', hue='label', data=df, palette=color_map, s=100, alpha=0.7)
    plt.title('Data distribution by class')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Class')

def save_model(model, scaler, output_dir='models/binary'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, 'binary_logreg_model.joblib')
    joblib.dump(model, model_path)
    
    scaler_path = os.path.join(output_dir, 'binary_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model_path, scaler_path

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")
    
    return model, scaler

def test_model_manually(model, scaler):
    print("\n=== Manual model testing ===")
    print("Enter values for score1 and score2 (or 'exit' to quit):")
    
    while True:
        try:
            score1_input = input("score1 (between 0 and 1): ")
            if score1_input.lower() == 'exit':
                break
                
            score2_input = input("score2 (between 0 and 1): ")
            if score2_input.lower() == 'exit':
                break
            
            score1 = float(score1_input)
            score2 = float(score2_input)
            
            X_manual = np.array([[score1, score2]])
            
            X_manual_scaled = scaler.transform(X_manual)
            
            prediction = model.predict(X_manual_scaled)[0]
            probabilities = model.predict_proba(X_manual_scaled)[0]
            
            print(f"\nPrediction for score1={score1}, score2={score2}:")
            print(f"Class: {prediction}")
            print("Class probabilities:")
            for i, cls in enumerate(model.classes_):
                print(f"  {cls}: {probabilities[i]:.4f}")
            
            print("\nEnter new values or 'exit' to quit:")
            
        except ValueError:
            print("Error: Enter numeric values or 'exit'.")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    print("Training binary logistic regression on data from JSON...")
    
    X, y, _ = load_data_from_json()
    
    visualize_data_distribution(X, y)
    
    model, scaler = train_logistic_regression()
    
    model_path, scaler_path = save_model(model, scaler)
    
    print("\nExample of using the model for prediction:")
    
    test_samples = np.array([
        [0.65, 0.85],
        [0.75, 0.65],
        [0.55, 0.95]
    ])
    
    test_samples_scaled = scaler.transform(test_samples)
    
    predictions = model.predict(test_samples_scaled)
    probabilities = model.predict_proba(test_samples_scaled)
    
    print("\nPredictions for test samples:")
    for i, (sample, pred, probs) in enumerate(zip(test_samples, predictions, probabilities)):
        print(f"Sample {i+1}: score1={sample[0]:.2f}, score2={sample[1]:.2f}")
        print(f"  Predicted class: {pred}")
        print(f"  Class probabilities: {dict(zip(model.classes_, probs))}")
        print()
    
    test_model_manually(model, scaler)
    
    print("Binary logistic regression training and analysis completed.")
    plt.show()

def run_manual_testing():
    model_path = 'models/binary/binary_logreg_model.joblib'
    scaler_path = 'models/binary/binary_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model files not found. Train the model first.")
        return
    
    model, scaler = load_model(model_path, scaler_path)
    
    test_model_manually(model, scaler)

if __name__ == "__main__":
    main()
    
    # run_manual_testing() 