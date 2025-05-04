import os
import argparse
from datetime import datetime
from bino_analyzer import analyze_text
from character_editor import CharacterEditor
from edit_writer import EditWriter
from html_reporter import generate_report_from_files, save_text_to_file, get_text_folder_name

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
        elif api_type == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is not specified. Provide it when creating an instance or through the OPENAI_API_KEY environment variable")
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Supported types are 'deepseek', 'gemini', and 'openai'")
        
        self.character_editor = CharacterEditor(api_key=self.api_key, api_type=self.api_type)
        self.edit_writer = EditWriter(api_key=self.api_key, api_type=self.api_type)
    
    def obfuscate_text(self, text, num_regions=3, cleanup_formatting=True):
        text_versions = {
            "original": text
        }
        saved_files = []
        
        folder_name = get_text_folder_name(text)
        
        original_file = save_text_to_file(text, "original", folder_name)
        saved_files.append(original_file)
        
        if cleanup_formatting:
            print("Step 1: Cleaning up formatting...")
            cleaned_text = self.character_editor.remove_extra_characters(text)
            text_versions["cleaned"] = cleaned_text
            
            cleaned_file = save_text_to_file(cleaned_text, "cleaned", folder_name)
            saved_files.append(cleaned_file)
            
            current_text = cleaned_text
        else:
            current_text = text
        
        iteration = 0
        max_iterations = 3
        current_verdict = None
        verdict_history = []
        
        while iteration < max_iterations:
            iteration += 1
            
            print(f"Iteration {iteration}/{max_iterations}: Analyzing text...")
            analysis_result = analyze_text(current_text, add_edit_tags=True, num_regions=num_regions)
            current_verdict = analysis_result["verdict"]
            binoculars_score = analysis_result["binoculars_score"]
            
            verdict_info = f"Iteration {iteration}\n"
            verdict_info += f"Verdict: {current_verdict}\n"
            verdict_info += f"Binoculars score: {binoculars_score:.6f}\n"
            
            verdict_file = save_text_to_file(verdict_info, f"verdict_{iteration}", folder_name)
            saved_files.append(verdict_file)
            verdict_history.append({
                "iteration": iteration, 
                "verdict": current_verdict, 
                "binoculars_score": binoculars_score
            })
            
            if "word_bino_html" in analysis_result:
                word_bino_file = save_text_to_file(analysis_result["word_bino_html"], f"word_scores_{iteration}", folder_name)
                saved_files.append(word_bino_file)
            
            if current_verdict == "Most likely human-generated":
                print(f"Text detected as human-generated. No further obfuscation needed.")
                break
            
            if "edited_text" not in analysis_result:
                print(f"No sections requiring edits were identified.")
                break
            
            print(f"Iteration {iteration}/{max_iterations}: Text detected as AI-generated. Obfuscating...")
            
            tagged_text = analysis_result["edited_text"]
            text_versions[f"tagged_{iteration}"] = tagged_text
            
            tagged_file = save_text_to_file(tagged_text, f"tagged_{iteration}", folder_name)
            saved_files.append(tagged_file)
            
            print(f"Iteration {iteration}/{max_iterations}: Rewriting identified sections...")
            edited_text = self.edit_writer.process_text(tagged_text)
            text_versions[f"edited_{iteration}"] = edited_text
            
            edited_file = save_text_to_file(edited_text, f"edited_{iteration}", folder_name)
            saved_files.append(edited_file)
            
            current_text = edited_text
            
            if iteration == max_iterations:
                break
        
        if verdict_history:
            summary = "# History of Obfuscator Verdicts\n\n"
            
            for entry in verdict_history:
                summary += f"## Iteration {entry['iteration']}\n"
                summary += f"- Verdict: {entry['verdict']}\n"
                summary += f"- Binoculars score: {entry['binoculars_score']:.6f}\n"
            
            verdict_summary_file = save_text_to_file(summary, "verdict_summary", folder_name)
            saved_files.append(verdict_summary_file)
        
        if iteration > 0 and current_verdict == "Most likely AI-generated":
            final_text = text_versions.get(f"edited_{iteration}", current_text)
        else:
            final_text = current_text
            
        if cleanup_formatting:
            print("Final step: Cleaning up formatting of the processed text...")
            final_text = self.character_editor.remove_extra_characters(final_text)
            text_versions["final_cleaned"] = final_text
            
            final_cleaned_file = save_text_to_file(final_text, "final_cleaned", folder_name)
            saved_files.append(final_cleaned_file)
            
        text_versions["final"] = final_text
        
        final_file = save_text_to_file(final_text, "final", folder_name)
        saved_files.append(final_file)
        
        print("Generating HTML report from all saved files...")
        html_file = generate_report_from_files(folder_name)
        saved_files.append(html_file)
        
        return {
            "original_text": text,
            "processed_text": final_text,
            "text_versions": text_versions,
            "files": saved_files,
            "verdict": current_verdict,
            "iterations": iteration,
            "verdict_history": verdict_history
        }
    
    def obfuscate_file(self, input_file, num_regions=3, cleanup_formatting=True):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file {input_file}: {str(e)}")
            return None
        
        result = self.obfuscate_text(text, num_regions, cleanup_formatting)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text obfuscation pipeline")
    parser.add_argument("--input", "-i", help="Input file path", required=True)
    parser.add_argument("--regions", "-r", help="Number of regions to edit (2-5)", type=int, default=3)
    parser.add_argument("--no-cleanup", help="Skip the formatting cleanup step", action="store_true")
    parser.add_argument("--api-key", help="API key for chosen model")
    parser.add_argument("--api-type", help="API type to use", choices=["deepseek", "gemini", "openai"], default="deepseek")
    
    args = parser.parse_args()
    
    obfuscator = TextObfuscator(api_key=args.api_key, api_type=args.api_type)
    result = obfuscator.obfuscate_file(
        args.input,
        num_regions=args.regions,
        cleanup_formatting=not args.no_cleanup
    )
    
    if result:
        print("\nObfuscation completed successfully!")
        print(f"Output directory: {os.path.dirname(result['files'][0])}")
        print(f"Final verdict: {result.get('verdict', 'Unknown')}")
        print(f"Obfuscation iterations: {result['iterations']}")
        print("\nGenerated files:")
        for i, file in enumerate(result["files"]):
            print(f"  {i+1}. {os.path.basename(file)}")