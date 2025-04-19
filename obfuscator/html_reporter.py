from datetime import datetime
import os

def generate_html_report(text_versions, analysis_file=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"obfuscation_report_{timestamp}.html"
    
    sections = ""
    
    stages = [
        ("original", "Original Text"),
        ("cleaned", "Text After Formatting Cleanup"),
        ("with_scores", "Text with Binocular Scores"),
        ("tagged", "Text with <EDIT> Tags"),
        ("final", "Final Obfuscated Text")
    ]
    
    for key, title in stages:
        if key in text_versions and text_versions[key]:
            content = text_versions[key].replace("<", "&lt;").replace(">", "&gt;")
            sections += f"""
            <div class="text-section">
                <h3>{title}</h3>
                <pre class="text-content">{content}</pre>
            </div>
            """
    
    analysis_link = ""
    if analysis_file:
        analysis_link = f'<p>For detailed analysis, see: <a href="{analysis_file}" target="_blank">{analysis_file}</a></p>'
    
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
            .text-content {{ white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .highlight {{ background-color: #e0f7fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .meta-info {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Text Obfuscation Report</h2>
            <p class="meta-info">Report generated on: {timestamp}</p>
            
            {sections}
            
            {analysis_link}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {filename}")
    return filename

def save_text_to_file(text, prefix="text", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{prefix}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filename 