import json
import random
import os

input_file = "meta_class/score_res/mer.json"
output_file = "meta_class/score_res/mer_balanced.json"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data_container = json.load(f)
    
    print(f"File {input_file} successfully loaded")
    
    if isinstance(data_container, dict) and "data" in data_container:
        data = data_container["data"]
        print(f"Array found in 'data' key with {len(data)} elements")
    else:
        print("Error: 'data' key not found in JSON")
        exit(1)
    
    source_groups = {}
    for item in data:
        if isinstance(item, dict) and "source" in item:
            source_type = item["source"]
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(item)
    
    print("Number of elements by class:")
    for source_type, items in source_groups.items():
        print(f"{source_type}: {len(items)}")
    
    min_count = min(len(group) for group in source_groups.values())
    print(f"{min_count} elements will be selected for each class")
    
    balanced_data = []
    for source_type, items in source_groups.items():
        if len(items) > min_count:
            selected_items = random.sample(items, min_count)
        else:
            selected_items = items
        balanced_data.extend(selected_items)
    
    result = {"data": balanced_data}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Balanced data saved to {output_file}")
    print(f"Total elements in result: {len(balanced_data)}")
    print(f"Classes in result: {len(source_groups)}")
    print(f"{min_count} elements for each class")

except Exception as e:
    print(f"An error occurred: {e}")
    
    if not os.path.exists(input_file):
        print(f"File {input_file} not found!")
