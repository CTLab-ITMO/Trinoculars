import requests
import json
import os
from typing import Optional

class CharacterEditor:
    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not specified. Provide it when creating an instance or through the DEEPSEEK_API_KEY environment variable")
        
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def remove_extra_characters(self, text: str) -> str:
        prompt = f"""
        Please process the following text by removing all unnecessary characters, 
        such as repeated spaces, extra punctuation marks, invisible characters, 
        and other unwanted formatting elements.
        Keep only standard characters and correct punctuation.
        Preserve the text structure, paragraphs, and meaning.
        Return only the processed text without additional comments.
        
        Text to process:
        ```
        {text}
        ```
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 1,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            cleaned_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return cleaned_text.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return text
        except (KeyError, IndexError) as e:
            print(f"Error processing API response: {e}")
            return text

if __name__ == "__main__":
    editor = CharacterEditor()
    
    sample_text = "Это   текст  с  лишними     пробелами...... и другими!!!!   проблемами   форматирования."
    
    cleaned_text = editor.remove_extra_characters(sample_text)
    print(f"Text after API processing: {cleaned_text}")