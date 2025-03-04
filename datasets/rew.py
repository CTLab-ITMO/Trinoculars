import json

def update_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Проверяем, является ли data списком (если JSON содержит несколько объектов)
    if isinstance(data, list):
        for item in data:
            dataset = item.get("dataset", "")
            if isinstance(dataset, list):
                dataset_str = " ".join(dataset)
            else:
                dataset_str = dataset
            
            # Проверяем, содержит ли dataset "SM"
            if "SM" in dataset_str:
                item["source"] = "ai+par"
    else:
        dataset = data.get("dataset", "")
        if isinstance(dataset, list):
            dataset_str = " ".join(dataset)
        else:
            dataset_str = dataset
        
        # Проверяем, содержит ли dataset "SM"
        if "SM" in dataset_str:
            data["source"] = "ai+par"
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    print("Файл обновлён.")

# Использование
file_path = "ru_detection_dataset.json"  # Укажите путь к файлу
update_json_file(file_path)
