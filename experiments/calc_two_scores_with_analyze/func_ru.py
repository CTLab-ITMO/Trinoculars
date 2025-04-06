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

def run_dataset(bino_chat, bino_coder, data):
    results = []
    error_count = 0
    check_counter = 0

    for row in data:
        try:
            text = row.get("text", "")
            if not text and "content" in row:
                text = row["content"]
            
            if not text:
                print(f"Warning: Empty text found in data row: {row}")
                continue
                
            score_chat = bino_chat.compute_score(text)
            score_coder = bino_coder.compute_score(text)
                    
        except Exception as e:
            print(f"\nError computing score for text: {row.get('text', '[NO TEXT]')[:100]}..., Error: {e}")
            error_count += 1
            continue
        
        source = row.get("source", "human")
        
        example_data = {
            "text": text,
            "source": source,
            "score_chat": score_chat,
            "score_coder": score_coder,
        }
        
        results.append(example_data)

        check_counter += 1
        if check_counter % 10 == 0:
            sys.stdout.write(f"\rProcessed: {check_counter} items")
            sys.stdout.flush()

    return {
        'data': results,
        'error_count': error_count,
        'check_counter': check_counter
    }