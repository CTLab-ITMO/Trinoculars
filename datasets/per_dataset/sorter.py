import json
import sys

def sort_json_by_id(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '_sorted.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            sorted_data = sorted(data, key=lambda x: x.get('id', 0))
        elif isinstance(data, dict):
            sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
        else:
            print("Unsupported JSON structure. Please provide a list or dictionary.")
            return False
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sorted JSON saved to {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: {input_file} contains invalid JSON.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

input_file = "datasets/summarization-dataset/abs_generator/generated_scientific_articles.json"
output_file = "datasets/summarization-dataset/abs_generator/generated_scientific_articles_sorted.json"

sort_json_by_id(input_file, output_file)