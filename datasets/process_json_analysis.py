import json
import os
from text_analysis import analyze_text
import time

def process_json_file(input_file, output_file=None):
    print(f"Loading JSON file: {input_file}")
    
    if output_file is None:
        base_name, ext = os.path.splitext(input_file)
        output_file = f"{base_name}_analyzed{ext}"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' not in data:
        print("Error: JSON file does not have a 'data' field")
        return None
    
    total_entries = len(data['data'])
    print(f"Found {total_entries} entries to process")
    
    for i, entry in enumerate(data['data']):
        if i % 10 == 0:
            print(f"Processing entry {i+1}/{total_entries} ({(i+1)/total_entries*100:.1f}%)")
        
        if 'text' in entry:
            analysis_results = analyze_text(entry['text'])
            entry['text_analysis'] = analysis_results
    
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Processing complete!")
    return output_file

if __name__ == "__main__":
    input_file = "results/long_sc_valid_results_20250407_000413.json"
    
    start_time = time.time()
    output_file = process_json_file(input_file)
    end_time = time.time()
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}") 