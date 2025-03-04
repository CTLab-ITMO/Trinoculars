import json

file_paths = [

]

merged_dataset = []
max_id = 0

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)['data']
        for item in dataset:
            item['id'] += max_id
        max_id = max(item['id'] for item in dataset) + 1
        merged_dataset.extend(dataset)

with open(r'mer.json', 'w', encoding='utf-8') as file:
    json.dump({"data": merged_dataset}, file, ensure_ascii=False, indent=4)

print("Merged dataset saved to 'mer.json'.")
