import requests
import json
import os
import re
from typing import Optional, Dict, List, Tuple
from google import genai
from openai import OpenAI

class EditWriter:
    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.deepseek.com/v1/chat/completions", api_type: str = "deepseek"):
        self.api_type = api_type
        
        if api_type == "deepseek":
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API key is not specified. Provide it when creating an instance or through the DEEPSEEK_API_KEY environment variable")
            
            self.api_url = api_url
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif api_type == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key is not specified. Provide it when creating an instance or through the GEMINI_API_KEY environment variable")
            
            self.gemini_client = genai.Client(api_key=self.api_key)
        elif api_type == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is not specified. Provide it when creating an instance or through the OPENAI_API_KEY environment variable")
            
            self.openai_client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Supported types are 'deepseek', 'gemini', and 'openai'")
    
    def rewrite_text(self, text_to_edit: str) -> str:
        prompt = f"""
        Ниже приведён текст с фрагментами, требующими существенной переработки. Каждый такой фрагмент обрамлён тегами <EDIT> и </EDIT>.
        Твоя задача - полностью перефразировать эти фрагменты так, чтобы:
        1. Текст выглядел как написанный автором
        2. Значение и смысл сохранились, но выражены другими словами
        3. Структура предложений была значительно изменена (длина, порядок слов, активный/пассивный залог)
        4. Использовались разнообразные синтаксические конструкции
        
        При этом:
        1. Всё вне тегов <EDIT> должно остаться абсолютно неизменным (включая пробелы, списки, заголовки, кавычки и т. д.)
        2. Теги <EDIT> и </EDIT> не включай в итог, вместо них вставляй только перефразированный текст
        3. Общий тон и стиль документа должен быть сохранен
        
        Текст для переработки:
        ```
        {text_to_edit}
        ```
        """
        
        if self.api_type == "deepseek":
            return self._rewrite_with_deepseek(prompt)
        elif self.api_type == "openai":
            return self._rewrite_with_openai(prompt)
        else:
            return self._rewrite_with_gemini(prompt)
    
    def _rewrite_with_deepseek(self, prompt: str) -> str:
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
            rewritten_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not rewritten_text:
                raise ValueError("DeepSeek API returned empty response")
                
            return rewritten_text.strip()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API request error: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Error processing DeepSeek API response: {e}")
    
    def _rewrite_with_gemini(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            if not response or not response.text:
                raise ValueError("Gemini API returned empty response")
                
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Error with Gemini API: {e}")
    
    def _rewrite_with_openai(self, prompt: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0
            )
            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError("OpenAI API returned empty response")
                
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error with OpenAI API: {e}")
    
    def process_text(self, text: str) -> str:
        try:
            rewritten = self.rewrite_text(text)
            return rewritten
        except Exception as e:
            print(f"\nError during text rewriting: {str(e)}")
            raise