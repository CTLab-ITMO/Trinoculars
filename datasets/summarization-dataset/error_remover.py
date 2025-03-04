import json
import os
import glob

def process_json_files(directory_path):
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory {directory_path}")
        return
    
    for file_path in json_files:
        print(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        if isinstance(data, dict):
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            if not list_keys:
                print(f"No data lists found in file {file_path}")
                continue
            
            for key in list_keys:
                original_length = len(data[key])
                data[key] = [item for item in data[key] if not (isinstance(item, dict) and 
                                                              item.get('text') == "[Ошибка генерации]")]
                removed_count = original_length - len(data[key])
                
                for i, item in enumerate(data[key], 1):
                    if isinstance(item, dict) and 'id' in item:
                        item['id'] = i
                
                print(f"  - In key '{key}' removed records: {removed_count}, renumbered: {len(data[key])}")
        
        elif isinstance(data, list):
            original_length = len(data)
            data = [item for item in data if not (isinstance(item, dict) and 
                                              item.get('text') == "[Ошибка генерации]")]
            removed_count = original_length - len(data)
            
            for i, item in enumerate(data, 1):
                if isinstance(item, dict) and 'id' in item:
                    item['id'] = i
            
            print(f"  - Removed records: {removed_count}, renumbered: {len(data)}")
        
        else:
            print(f"Unsupported data format in file {file_path}")
            continue
        
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            print(f"  - File successfully updated")
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")

if __name__ == "__main__":
    directory = input("Enter the path to the directory with JSON files: ")
    process_json_files(directory)
