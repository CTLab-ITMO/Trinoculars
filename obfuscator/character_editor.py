import requests
import json
import os
from typing import Optional
from google import genai
from openai import OpenAI

class CharacterEditor:
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
    
    def remove_extra_characters(self, text: str) -> str:
        try:
            prompt = f"""
            Внимательно прочитай предоставленный текст и удали из него все лишние элементы форматирования и служебные метки, не изменяя смысл. Выполни следующую очистку:
            1. Форматирование: убери разметку Markdown (например, символы `**`, `_`, `~~` для оформления текста) и все HTML-теги, если они присутствуют. Текст должен остаться без **жирного**, *курсивного* или ~зачёркнутого~ оформления – только обычный текст.
            2. Структурные метки: удали заголовки или префиксы вроде «Тема:», «Вопрос:», «Ответ:» – оставь вместо них просто текст вопроса или ответа без слов «вопрос/ответ». Также удали маркеры списков: дефисы, точки, звездочки, нумерацию перед элементами списка. Содержимое бывших списков оставь как отдельные предложения или объедини в абзацы, но без спецсимволов в начале.
            3. Технические комментарии: убери из текста любые части вроде «Пример:», «Примечание:», «Замечание:» и похожие служебные комментарии. Также удали возможные пояснения от лица модели (например, фразы про то, какое это задание или инструкция), если они есть. Оставь только сам текст без дополнительных объяснений.
            4. Сохранение смысла: не добавляй и не убирай смысловую информацию. Перефразируй минимально, только если нужно убрать лишние метки или форматирование. Структура и смысл предложений исходного текста должны сохраниться, просто без форматирования и служебных элементов.
            На выходе выдай только очищенный текст на русском языке, без каких-либо дополнительных комментариев.
            
            Текст для очистки:
            ```
            {text}
            ```
            """
            
            if self.api_type == "deepseek":
                return self._process_with_deepseek(prompt)
            elif self.api_type == "openai":
                return self._process_with_openai(prompt)
            else:
                return self._process_with_gemini(prompt)
        except Exception as e:
            print(f"\nError during text cleaning: {str(e)}")
            raise
    
    def _process_with_deepseek(self, prompt: str) -> str:
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
            
            if not cleaned_text:
                raise ValueError("DeepSeek API returned empty response")
                
            return cleaned_text.strip()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API request error: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Error processing DeepSeek API response: {e}")
    
    def _process_with_gemini(self, prompt: str) -> str:
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
            
    def _process_with_openai(self, prompt: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError("OpenAI API returned empty response")
                
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error with OpenAI API: {e}")