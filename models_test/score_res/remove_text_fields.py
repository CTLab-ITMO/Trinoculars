import json
import os
import sys

def remove_text_fields(input_file_path, output_file_path=None):
    print(f"Processing file: {input_file_path}")
    
    if output_file_path is None:
        output_file_path = input_file_path
    
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} not found.")
        return False
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        removed_fields = 0
        
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if "text" in item:
                    del item["text"]
                    removed_fields += 1
        elif isinstance(data, list):
            for item in data:
                if "text" in item:
                    del item["text"]
                    removed_fields += 1
        elif isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, dict) and "text" in item:
                    del item["text"]
                    removed_fields += 1
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        
        print(f"Processing completed. Removed 'text' fields: {removed_fields}")
        print(f"Result saved to file: {output_file_path}")
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    input_file = "merged_results_three.json"
    output_file = "merged_results_three_no_text.json"
    
    print("Removing 'text' fields from the merged file...")
    if remove_text_fields(input_file, output_file):
        print("Processing completed successfully!")
    else:
        print("An error occurred while processing the file.") 