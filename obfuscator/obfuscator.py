import os
import argparse
from datetime import datetime
from bino_analyzer import analyze_text
from character_editor import CharacterEditor
from edit_writer import EditWriter
from html_reporter import generate_html_report, save_text_to_file, ensure_directory

class TextObfuscator:
    def __init__(self, api_key=None, api_type="deepseek"):
        self.api_type = api_type
        
        if api_type == "deepseek":
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API key is not specified. Provide it when creating an instance or through the DEEPSEEK_API_KEY environment variable")
        elif api_type == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key is not specified. Provide it when creating an instance or through the GEMINI_API_KEY environment variable")
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Supported types are 'deepseek' and 'gemini'")
        
        self.character_editor = CharacterEditor(api_key=self.api_key, api_type=self.api_type)
        self.edit_writer = EditWriter(api_key=self.api_key, api_type=self.api_type)
    
    def obfuscate_text(self, text, edit_threshold=0.7, cleanup_formatting=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        text_versions = {
            "original": text
        }
        saved_files = []
        
        original_file = save_text_to_file(text, "original", timestamp)
        saved_files.append(original_file)
        
        if cleanup_formatting:
            print("Step 1: Cleaning up formatting...")
            cleaned_text = self.character_editor.remove_extra_characters(text)
            text_versions["cleaned"] = cleaned_text
            
            cleaned_file = save_text_to_file(cleaned_text, "cleaned", timestamp)
            saved_files.append(cleaned_file)
            
            current_text = cleaned_text
        else:
            current_text = text
        
        print("Step 2: Analyzing text to identify suspicious parts...")
        analysis_result = analyze_text(current_text, add_edit_tags=True, edit_threshold=edit_threshold)
        
        text_versions["with_scores"] = analysis_result["text_with_scores"]
        scored_file = save_text_to_file(analysis_result["text_with_scores"], "scored", timestamp)
        saved_files.append(scored_file)
        
        if "edited_text" not in analysis_result:
            print("No sections requiring edits were identified.")
            
            html_file = generate_html_report(text_versions, analysis_result, timestamp)
            saved_files.append(html_file)
            
            return {
                "original_text": text,
                "processed_text": current_text,
                "text_versions": text_versions,
                "files": saved_files
            }
        
        tagged_text = analysis_result["edited_text"]
        text_versions["tagged"] = tagged_text
        
        tagged_file = save_text_to_file(tagged_text, "tagged", timestamp)
        saved_files.append(tagged_file)
        
        print("Step 3: Rewriting identified sections...")
        final_text = self.edit_writer.process_text(tagged_text)
        text_versions["final"] = final_text
        
        final_file = save_text_to_file(final_text, "final", timestamp)
        saved_files.append(final_file)
        
        html_file = generate_html_report(
            text_versions,
            analysis_result,
            timestamp
        )
        saved_files.append(html_file)
        
        return {
            "original_text": text,
            "processed_text": final_text,
            "text_versions": text_versions,
            "files": saved_files
        }
    
    def obfuscate_file(self, input_file, edit_threshold=0.7, cleanup_formatting=True):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file {input_file}: {str(e)}")
            return None
        
        result = self.obfuscate_text(text, edit_threshold, cleanup_formatting)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text obfuscation pipeline")
    parser.add_argument("--input", "-i", help="Input file path", required=True)
    parser.add_argument("--threshold", "-t", help="Edit threshold (0.0-1.0)", type=float, default=0.7)
    parser.add_argument("--no-cleanup", help="Skip the formatting cleanup step", action="store_true")
    parser.add_argument("--api-key", help="API key for chosen model")
    parser.add_argument("--api-type", help="API type to use (deepseek or gemini)", choices=["deepseek", "gemini"], default="deepseek")
    
    args = parser.parse_args()
    
    obfuscator = TextObfuscator(api_key=args.api_key, api_type=args.api_type)
    result = obfuscator.obfuscate_file(
        args.input,
        edit_threshold=args.threshold,
        cleanup_formatting=not args.no_cleanup
    )
    
    if result:
        print("\nObfuscation completed successfully!")
        print(f"Output directory: {os.path.dirname(result['files'][0])}")
        print("\nGenerated files:")
        for i, file in enumerate(result["files"]):
            print(f"  {i+1}. {os.path.basename(file)}")