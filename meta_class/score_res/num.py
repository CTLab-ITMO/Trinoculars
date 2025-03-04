import json
import os
from collections import Counter

file_path = "models_test/score_res/mer.json"

if not os.path.exists(file_path):
    print(f"File {file_path} not found. Please provide the correct file path.")
    exit(1)

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file format.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)

sources = [item["source"] for item in data["data"]]

source_counts = Counter(sources)

total = len(sources)

source_percentages = {source: (count / total * 100) for source, count in source_counts.items()}

print(f"Total number of records: {total}\n")
print("Number of records by source:")
for source, count in source_counts.items():
    print(f"- {source}: {count} ({source_percentages[source]:.2f}%)")