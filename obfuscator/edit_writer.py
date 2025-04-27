import requests
import json
import os
import re
from typing import Optional, Dict, List, Tuple
from google import genai

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
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Supported types are 'deepseek' and 'gemini'")
    
    def rewrite_text(self, text_to_edit: str) -> str:
        prompt = f"""
        Ниже приведён текст с фрагментами, требующими доработки. Каждый такой фрагмент обрамлён тегами <EDIT> и </EDIT>.  
        Пожалуйста, верни полный текст, в котором:
        1. Перефразированы и расчищены от ошибок только части между <EDIT> и </EDIT>.
        2. Всё вне этих тегов сохранено дословно (включая пробелы, списки, заголовки, кавычки и т. д.).
        3. Теги <EDIT> и </EDIT> не включай в итог вместо них вставляй отредактированный текст.
        4. Сохрани общий тон и стиль документа.
        
        Текст для переработки:
        ```
        {text_to_edit}
        ```
        """
        
        if self.api_type == "deepseek":
            return self._rewrite_with_deepseek(prompt)
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
            
            return rewritten_text.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API request error: {e}")
            return re.sub(r'<\/?EDIT>', '', prompt.split("```")[1].strip())
        except (KeyError, IndexError) as e:
            print(f"Error processing DeepSeek API response: {e}")
            return re.sub(r'<\/?EDIT>', '', prompt.split("```")[1].strip())
    
    def _rewrite_with_gemini(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return re.sub(r'<\/?EDIT>', '', prompt.split("```")[1].strip())
    
    def process_text(self, text: str) -> str:
        rewritten = self.rewrite_text(text)
        return rewritten

if __name__ == "__main__":
    
    text = '''### Искусственный интеллект в современном мире
    Искусственный интеллект (ИИ) стал <EDIT>один из наиболее важных технологических достижений последних десятилетий. Он трансформирует множество областей, включая медицина, финансы, производство и транспорт.</EDIT>
    #### Основные применения ИИ:
    1. Медицина - <EDIT>ИИ помогает врачам диагностировать заболевания, анализировать медицинские снимки и разрабатывать планы лечения. Системы ИИ способны обрабатывать большие объёмы медицинских данных для выявления закономерностей, которые человек может пропустить.</EDIT>
    2. Финансы - ИИ используется для выявления мошеннических транзакций, оценки кредитных рисков и автоматизации торговли.
    3. Транспорт - <EDIT>Автономные транспортные средства, использующие алгоритмы ИИ, обещают революционизировать транспортную отрасль и сделать дороги более безопасными. Компании как Tesla, Waymo и другие инвестируют миллиарды в разработку самоуправляемых автомобилей.</EDIT>
    #### Этические вопросы
    Развитие ИИ поднимает важные этические вопросы, включая проблемы <EDIT>приватность данных, алгоритмическая предвзятость, автоматизация рабочих мест и потенциальная автономность систем вооружений.</EDIT>
    В заключение, ИИ представляет собой мощный инструмент с огромным потенциалом для решения сложных проблем, но требует ответственного подхода к его развитию и применению.'''
    
    writer = EditWriter(api_type="deepseek")
    output_text = writer.process_text(text)
    print("\nИсходный текст:")
    print(text)
    print("\nОбработанный текст (DeepSeek):")
    print(output_text)
    
    if os.environ.get("GEMINI_API_KEY"):
        writer_gemini = EditWriter(api_type="gemini")
        output_text_gemini = writer_gemini.process_text(text)
        print("\nОбработанный текст (Gemini):")
        print(output_text_gemini)