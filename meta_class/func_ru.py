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
from meta_class.analyzer import analyze_text

def run_ru_dataset(bino_chat, bino_coder, data):
    results = []
    error_count = 0
    check_counter = 0

    for row in data:
        try:
            score_chat = bino_chat.compute_score(row["text"])
            score_coder = bino_coder.compute_score(row["text"])

            text_analysis = analyze_text(row["text"])
                    
        except Exception as e:
            print(f"\nError computing score for text: {row['text']}, Error: {e}")
            error_count += 1
            continue
        
        example_data = {
            "text": row["text"],
            "source": row["source"],
            "dataset": row.get("dataset", "unknown"),
            "score_chat": score_chat,
            "score_coder": score_coder,
            "text_analysis": text_analysis
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