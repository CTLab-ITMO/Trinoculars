import argparse
import pandas as pd

from text_analysis import show_text_analysis
from binoculars_utils import initialize_binoculars, compute_scores
from model_utils import load_model, classify_text

def main():
    parser = argparse.ArgumentParser(description='Text classifier demonstration (Human vs AI)')
    parser.add_argument('--text', type=str, help='Text for classification')
    parser.add_argument('--file', type=str, help='Path to file with text')
    parser.add_argument('--analysis', action='store_true', help='Show detailed text analysis')
    parser.add_argument('--compute-scores', action='store_true', help='Compute score_chat and score_coder')
    parser.add_argument('--model-type', type=str, choices=['binary', 'three-class'], default='binary',
                       help='Type of classification model to use (binary or three-class)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive chat-like mode')
    args = parser.parse_args()
    
    bino_chat = None
    bino_coder = None
    if args.compute_scores:
        bino_chat, bino_coder = initialize_binoculars()
    
    model_type = args.model_type
    print(f"Loading {model_type} classifier model...")
    model, scaler, label_encoder, imputer = load_model(model_type=model_type)
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end the session.")
        print("Type your text and press Enter to classify it.\n")
        
        while True:
            text = input("Enter text (or 'exit' to quit): ")
            if text.lower() in ['exit', 'quit']:
                break
                
            if not text.strip():
                continue
                
            scores = None
            if args.compute_scores:
                scores = compute_scores(text, bino_chat, bino_coder)
            
            print(f"\nAnalyzing text...")
            result = classify_text(text, model, scaler, label_encoder, imputer=imputer, scores=scores)
            
            print("\n" + "="*50)
            print("CLASSIFICATION RESULTS")
            print("="*50)
            print(f"Predicted class: {result['predicted_class']}")
            print("Class probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"  - {cls}: {prob:.4f}")
            
            if scores:
                print("\nComputed scores:")
                if 'score_chat' in scores:
                    print(f"  - Score Chat: {scores['score_chat']:.4f}")
                if 'score_coder' in scores:
                    print(f"  - Score Coder: {scores['score_coder']:.4f}")
            
            if args.analysis:
                show_text_analysis(result['text_analysis'])
            
            print("\n" + "-"*50 + "\n")
    else:
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = input("Enter text for classification: ")
        
        scores = None
        if args.compute_scores:
            scores = compute_scores(text, bino_chat, bino_coder)
        
        print(f"\nAnalyzing text...")
        result = classify_text(text, model, scaler, label_encoder, imputer=imputer, scores=scores)
        
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Predicted class: {result['predicted_class']}")
        print("Class probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  - {cls}: {prob:.4f}")
        
        if model_type == 'three-class':
            print("\nThree-class model explanation:")
        else:
            print("\nBinary model explanation:")
        
        if scores:
            print("\nComputed scores:")
            if 'score_chat' in scores:
                print(f"  - Score Chat: {scores['score_chat']:.4f}")
            if 'score_coder' in scores:
                print(f"  - Score Coder: {scores['score_coder']:.4f}")
        
        if args.analysis:
            show_text_analysis(result['text_analysis'])
    
    if args.compute_scores:
        if bino_chat:
            bino_chat.free_memory()
        if bino_coder:
            bino_coder.free_memory()

if __name__ == "__main__":
    main()


