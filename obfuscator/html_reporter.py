from datetime import datetime
import os
import re
import glob

def ensure_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def generate_html_report(text_versions=None, analysis_result=None, timestamp=None, file_list=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = ensure_directory(f"output_{timestamp}")
    filename = os.path.join(output_dir, f"obfuscation_report_{timestamp}.html")
    
    sections = ""
    
    if file_list:
        sorted_files = sort_files_by_type(file_list)
        
        for file_path in sorted_files:
            base_name = os.path.basename(file_path)
            title = get_title_from_filename(base_name)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            safe_content = content.replace("<", "&lt;").replace(">", "&gt;")
            sections += f"""
            <div class="text-section">
                <h3>{title} ({base_name})</h3>
                <pre class="text-content">{safe_content}</pre>
            </div>
            """
    elif text_versions:
        stages = [
            ("original", "Original Text"),
            ("cleaned", "Text After Formatting Cleanup"),
            ("with_scores", "Text with Binocular Scores"),
        ]
        
        for i in range(1, 4):
            if f"tagged_{i}" in text_versions:
                stages.append((f"tagged_{i}", f"Iteration {i}: Text with <EDIT> Tags"))
            if f"edited_{i}" in text_versions:
                stages.append((f"edited_{i}", f"Iteration {i}: Edited Text"))
            if f"with_scores_{i}" in text_versions:
                stages.append((f"with_scores_{i}", f"Iteration {i}: Text with Scores"))
        
        stages.extend([
            ("final_cleaned", "Final Cleaned Text"),
            ("final", "Final Obfuscated Text")
        ])
        
        for key, title in stages:
            if key in text_versions and text_versions[key]:
                content = text_versions[key]
                
                if key.startswith("with_scores") and analysis_result and "tokens" in analysis_result and "binocular_scores" in analysis_result:
                    highlighted_content = ""
                    tokens = analysis_result["tokens"]
                    scores = analysis_result["binocular_scores"].squeeze().tolist()
                    
                    for token, score in zip(tokens, scores):
                        color_value = int(255 * score)
                        highlighted_content += f"<span style='background-color: rgb(255, {255-color_value}, {255-color_value}); color: black;'>{token}</span>"
                    
                    sections += f"""
                    <div class="text-section">
                        <h3>{title}</h3>
                        <div class="text-content highlighted-content">{highlighted_content}</div>
                    </div>
                    """
                else:
                    safe_content = content.replace("<", "&lt;").replace(">", "&gt;")
                    sections += f"""
                    <div class="text-section">
                        <h3>{title}</h3>
                        <pre class="text-content">{safe_content}</pre>
                    </div>
                    """
    
    if analysis_result and "html_edits" in analysis_result:
        sections += f"""
        <div class="text-section">
            <h3>Text with Highlighted Edits</h3>
            <div class="text-content">{analysis_result["html_edits"]}</div>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Text Obfuscation Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
            h2 {{ color: #333; background-color: #e7f5fe; padding: 10px; border-radius: 5px; }}
            h3 {{ color: #444; margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .text-section {{ background-color: white; margin: 15px 0; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .text-content {{ background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            pre.text-content {{ white-space: pre-wrap; }}
            .highlighted-content {{ line-height: 1.5; }}
            .highlight {{ background-color: #e0f7fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .meta-info {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Text Obfuscation Report</h2>
            <p class="meta-info">Report generated on: {timestamp}</p>
            
            {sections}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {filename}")
    return filename

def sort_files_by_type(file_list):
    file_order = {
        "original": 1,
        "cleaned": 2,
        "word_scores_1": 3,
        "token_scores_1": 4,
        "scored": 5,
        "tagged_1": 6,
        "edited_1": 7,
        "word_scores_2": 8,
        "token_scores_2": 9,
        "scored_1": 10,
        "tagged_2": 11,
        "edited_2": 12,
        "word_scores_3": 13,
        "token_scores_3": 14,
        "scored_2": 15,
        "tagged_3": 16,
        "edited_3": 17,
        "scored_3": 18,
        "final_cleaned": 19,
        "final": 20
    }
    
    def get_file_order(file_path):
        basename = os.path.basename(file_path)
        for prefix, order in file_order.items():
            if basename.startswith(prefix):
                return order
        return 999
    
    return sorted(file_list, key=get_file_order)

def get_title_from_filename(filename):
    name = re.sub(r'_\d{8}_\d{6}\.txt$', '', filename)
    
    if name.startswith("original"):
        return "Original Text"
    elif name.startswith("cleaned"):
        return "Text After Formatting Cleanup"
    elif name.startswith("word_scores_"):
        iteration = name.split("_")[2]
        return f"Iteration {iteration}: Word-Based Binocular Scores"
    elif name.startswith("token_scores_"):
        iteration = name.split("_")[2]
        return f"Iteration {iteration}: Token-Based Binocular Scores"
    elif name.startswith("scored") and not name[6:].isdigit():
        return "Text with Binocular Scores"
    elif name.startswith("scored_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Text with Scores"
    elif name.startswith("tagged_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Text with <EDIT> Tags"
    elif name.startswith("edited_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Edited Text"
    elif name.startswith("final_cleaned"):
        return "Final Cleaned Text"
    elif name.startswith("final"):
        return "Final Obfuscated Text"
    else:
        return " ".join(word.capitalize() for word in name.split("_"))

def generate_report_from_files(timestamp):
    output_dir = f"output_{timestamp}"
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found")
        return None
    
    text_files = glob.glob(os.path.join(output_dir, "*.txt"))
    if not text_files:
        print(f"No text files found in {output_dir}")
        return None
    
    html_file = generate_html_report(timestamp=timestamp, file_list=text_files)
    return html_file

def save_text_to_file(text, prefix="text", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    first_word = text.strip().split()[0] if text.strip() else "empty"
    first_word = ''.join(c for c in first_word if c.isalnum())
    first_word = first_word[:20]
    output_dir = ensure_directory(f"output_{timestamp}")
    filename = os.path.join(output_dir, f"{prefix}_{first_word}_{timestamp}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filename 