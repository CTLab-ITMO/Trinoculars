import json

# Список путей к файлам, которые нужно объединить
file_paths = [

]

merged_dataset = []
max_id = 0

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        # Обновляем идентификаторы
        for item in dataset:
            item['id'] += max_id
        max_id = max(item['id'] for item in dataset) + 1
        merged_dataset.extend(dataset)

with open(r'ex.json', 'w', encoding='utf-8') as file:
    json.dump(merged_dataset, file, ensure_ascii=False, indent=4)

print("Merged dataset saved to 'ex.json'.")
