from datetime import datetime
import os
import re
import glob

def ensure_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_text_folder_name(text):
    words = text.strip().split()[:5]
    folder_name = '_'.join(words)
    folder_name = ''.join(c if c.isalnum() else '_' for c in folder_name)
    folder_name = folder_name[:50]
    return folder_name or "empty_text"

def generate_html_report(text_versions=None, analysis_result=None, file_list=None):
    folder_name = os.path.basename(os.path.dirname(file_list[0])) if file_list else "report"
    folder_name = folder_name.replace("output_", "")
    
    output_dir = os.path.dirname(file_list[0]) if file_list else ensure_directory(f"output_{folder_name}")
    filename = os.path.join(output_dir, f"obfuscation_report.html")
    
    sections = ""
    
    if file_list:
        sorted_files = sort_files_by_type(file_list)
        
        for file_path in sorted_files:
            base_name = os.path.basename(file_path)
            title = get_title_from_filename(base_name)
            
            if base_name.startswith("token_scores_") or base_name.startswith("scored_"):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "verdict_" in base_name:
                if base_name.startswith("verdict_summary"):
                    content = convert_markdown_to_html(content)
                    sections += f"""
                    <div class="text-section verdict-summary">
                        <h3>{title}</h3>
                        <div class="verdict-content">{content}</div>
                    </div>
                    """
                else:
                    verdict_html = format_verdict_as_html(content)
                    sections += f"""
                    <div class="text-section verdict">
                        <h3>{title}</h3>
                        <div class="verdict-content">{verdict_html}</div>
                    </div>
                    """
            elif base_name.startswith("word_scores_"):
                sections += f"""
                <div class="text-section">
                    <h3>{title}</h3>
                    <div class="text-content highlighted-content">{content}</div>
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
    elif text_versions:
        stages = [
            ("original", "Original Text"),
            ("cleaned", "Text After Formatting Cleanup"),
        ]
        
        max_iteration = 0
        for key in text_versions.keys():
            if key.startswith(("tagged_", "edited_")):
                try:
                    iteration = int(key.split("_")[1])
                    max_iteration = max(max_iteration, iteration)
                except (ValueError, IndexError):
                    pass
        
        for i in range(1, max_iteration + 1):
            if f"tagged_{i}" in text_versions:
                stages.append((f"tagged_{i}", f"Iteration {i}: Text with <EDIT> Tags"))
            if f"edited_{i}" in text_versions:
                stages.append((f"edited_{i}", f"Iteration {i}: Edited Text"))
        
        stages.extend([
            ("final_cleaned", "Final Cleaned Text"),
            ("final", "Final Obfuscated Text")
        ])
        
        for key, title in stages:
            if key in text_versions and text_versions[key]:
                content = text_versions[key]
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
        <title>Text Obfuscation Report</title>
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
            .verdict {{ background-color: #f8f9ff; }}
            .verdict-summary {{ background-color: #f0f4ff; }}
            .verdict-content {{ padding: 10px; border-radius: 5px; }}
            .verdict-row {{ margin: 5px 0; padding: 5px; }}
            .verdict-label {{ font-weight: bold; display: inline-block; width: 200px; }}
            .verdict-value {{ display: inline-block; }}
            .human-generated {{ color: #2e7d32; background-color: #e8f5e9; padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
            .ai-generated {{ color: #c62828; background-color: #ffebee; padding: 5px 10px; border-radius: 3px; font-weight: bold; }}
            .score-value {{ font-family: monospace; font-size: 1.1em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Text Obfuscation Report</h2>
            
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
    base_order = {
        "original": 10,
        "cleaned": 20,
        "verdict_summary": 9000,
        "final_cleaned": 9010,
        "final": 9020
    }
    
    def get_file_order(file_path):
        basename = os.path.basename(file_path)
        name = basename.replace('.txt', '')
        
        if name in base_order:
            return base_order[name]
        
        if any(name.startswith(prefix) for prefix in ["verdict_", "word_scores_", "tagged_", "edited_"]):
            parts = name.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                iteration = int(parts[1])

                if name.startswith("verdict_"):
                    return (iteration * 1000) + 100
                elif name.startswith("word_scores_"):
                    return (iteration * 1000) + 200
                elif name.startswith("tagged_"):
                    return (iteration * 1000) + 300
                elif name.startswith("edited_"):
                    return (iteration * 1000) + 400
        
        return 9999
    
    return sorted(file_list, key=get_file_order)

def get_title_from_filename(filename):
    name = os.path.basename(filename).replace('.txt', '')
    
    if name == "original":
        return "Original Text"
    elif name == "cleaned":
        return "Text After Formatting Cleanup"
    elif name == "verdict_summary":
        return "History of Obfuscator Verdicts"
    elif name.startswith("verdict_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Analysis Result"
    elif name.startswith("word_scores_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Word-Based Binocular Scores"
    elif name.startswith("token_scores_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Token-Based Binocular Scores"
    elif name.startswith("scored_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Text with Scores"
    elif name.startswith("tagged_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Text with <EDIT> Tags"
    elif name.startswith("edited_"):
        iteration = name.split("_")[1]
        return f"Iteration {iteration}: Edited Text"
    elif name == "final_cleaned":
        return "Final Cleaned Text"
    elif name == "final":
        return "Final Obfuscated Text"
    else:
        return " ".join(word.capitalize() for word in name.split("_"))

def generate_report_from_files(folder_name):
    output_dir = f"{folder_name}"
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found")
        return None
    
    text_files = glob.glob(os.path.join(output_dir, "*.txt"))
    if not text_files:
        print(f"No text files found in {output_dir}")
        return None
    
    html_file = generate_html_report(file_list=text_files)
    return html_file

def save_text_to_file(text, prefix="text", folder_name=None):
    if folder_name is None:
        folder_name = get_text_folder_name(text)
    
    output_dir = ensure_directory(f"{folder_name}")
    filename = os.path.join(output_dir, f"{prefix}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filename

def format_verdict_as_html(verdict_text):
    lines = verdict_text.strip().split('\n')
    result = '<div class="verdict-details">'
    
    for line in lines:
        if not line.strip():
            continue
            
        if ':' in line:
            label, value = line.split(':', 1)
            
            if "Verdict" in label:
                if "human-generated" in value:
                    value_class = "human-generated"
                    value_text = value.strip()
                else:
                    value_class = "ai-generated"
                    value_text = value.strip()
                    
                result += f'<div class="verdict-row"><span class="verdict-label">{label}:</span><span class="verdict-value {value_class}">{value_text}</span></div>'
            
            elif "Average score" in label:
                value_text = value.strip()
                result += f'<div class="verdict-row"><span class="verdict-label">{label}:</span><span class="verdict-value score-value">{value_text}</span></div>'
            
            else:
                result += f'<div class="verdict-row"><span class="verdict-label">{label}:</span><span class="verdict-value">{value.strip()}</span></div>'
        else:
            result += f'<div class="verdict-row">{line}</div>'
    
    result += '</div>'
    return result

def convert_markdown_to_html(markdown_text):
    html = ""
    in_list = False
    
    for line in markdown_text.split('\n'):
        if line.startswith('# '):
            html += f'<h2>{line[2:]}</h2>'
        elif line.startswith('## '):
            html += f'<h3>{line[3:]}</h3>'
        elif line.startswith('### '):
            html += f'<h4>{line[4:]}</h4>'
        elif line.startswith('- '):
            if not in_list:
                html += '<ul>'
                in_list = True
            html += f'<li>{line[2:]}</li>'
        elif line.strip() == '':
            if in_list:
                html += '</ul>'
                in_list = False
            html += '<br>'
        else:
            if in_list:
                html += '</ul>'
                in_list = False
            html += f'<p>{line}</p>'
    
    if in_list:
        html += '</ul>'
    
    return html 