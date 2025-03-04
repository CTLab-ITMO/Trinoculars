import json
import os
import re
import pandas as pd
import glob

def extract_source(file_path):
    base = os.path.basename(file_path)
    parts = base.split("___")
    if len(parts) < 2:
        return base
    
    remainder = parts[1]
    remainder = re.sub(r'_\d{8}_\d{6}', '', remainder)
    remainder, _ = os.path.splitext(remainder)
    sources = remainder.split("_and_")
    result = ", ".join(sources)
    return result

def extract_language(file_path):
    base = os.path.basename(file_path)
    if base.startswith("results_eng"):
        return "eng"
    elif base.startswith("results_ru"):
        return "ru"
    return "unknown"

def extract_dataset(file_path):
    base = os.path.basename(file_path)
    parts = base.split("___")[0].split("_")
    if len(parts) >= 3:  # results_ru_gemini-2.0-pro-exp-02-05
        return "_".join(parts[2:])  # вернет "gemini-2.0-pro-exp-02-05"
    return "unknown"

def recalc_f1_if_needed(rows):
    for row in rows:
        try:
            f1 = row.get("f1_score", None)
            if f1 == 0:
                fpr_val = row.get("fpr", None)
                tnr_val = row.get("tnr", None)
                if fpr_val is not None and tnr_val is not None:
                    fpr_val = float(fpr_val)
                    tnr_val = float(tnr_val)
                    denom = (1 - fpr_val) + tnr_val
                    if denom != 0:
                        computed_f1 = 2 * (1 - fpr_val) * tnr_val / denom
                        row["f1_score"] = computed_f1
        except Exception as e:
            pass

files = glob.glob("models_test/res/with_score/*.json")

rows = []

metric_keys = [
    "f1_score", "avg_prediction", "roc_auc", "tpr_at_fpr_0_01", "tpr", "fpr", "tnr", "fnr"
]

for file_path in files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    source_label = extract_source(file_path)
    language_label = extract_language(file_path)
    
    if "metrics" in data:
        metrics = data["metrics"]
        row = {"Source": source_label, "Dataset": "overall", "Language": language_label}
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)
    
    if "overall_metrics" in data:
        metrics = data["overall_metrics"]
        dataset_name = extract_dataset(file_path)
        row = {"Source": source_label, "Dataset": dataset_name, "Language": language_label}
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)
    """
    if "dataset_metrics" in data:
        dataset_metrics = data["dataset_metrics"]
        for dataset_name, metrics in dataset_metrics.items():
            row = {"Source": source_label, "Dataset": dataset_name, "Language": language_label}
            for key in metric_keys:
                row[key] = metrics.get(key)
            rows.append(row)
    """
#recalc_f1_if_needed(rows)

df = pd.DataFrame(rows)

print(df.to_string(index=False))

output_filename = "metrics_new_2.xlsx"
df.to_excel(output_filename, index=False)
print(f"Data saved to Excel file: {output_filename}")