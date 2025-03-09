import pandas as pd
import json

num_samples = 2000
dataset_name = "Lenta.ru news"

df = pd.read_csv("lenta-ru-news.csv", parse_dates=["date"], dayfirst=False)

df_filtered = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2019-12-31")]

df_sampled = df_filtered.sample(n=min(num_samples, len(df_filtered)), random_state=42)

records = [
    {
        "id": i + 1,
        "text": row.text,
        "source": "human",
        "dataset": dataset_name
    }
    for i, row in enumerate(df_sampled.itertuples(index=False, name="Record"))
]

with open("filtered_data.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print(f"File saved: filtered_data.json with {num_samples} random records.")
