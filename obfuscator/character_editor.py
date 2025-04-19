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