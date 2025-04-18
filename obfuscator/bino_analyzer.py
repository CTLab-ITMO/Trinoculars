from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import os

DEVICE_1 = "cuda:0"

torch.set_grad_enabled(False)

observer_name = "deepseek-ai/deepseek-llm-7b-chat"
performer_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5" 

try:
    print("Loading tokenizers...")
    identical_tokens = (AutoTokenizer.from_pretrained(observer_name).vocab ==
                        AutoTokenizer.from_pretrained(performer_name).vocab)
    
    print("Loading observer model...")
    observer_model = AutoModelForCausalLM.from_pretrained(observer_name,
                                                       device_map={"": DEVICE_1},
                                                       trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16)

    print("Loading performer model...")
    performer_model = AutoModelForCausalLM.from_pretrained(performer_name,
                                                         device_map={"": DEVICE_1},
                                                         trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16)

    observer_model.eval()
    performer_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(observer_name)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def tokenize(batch):
    encodings = tokenizer(batch, return_tensors="pt", 
    padding="longest" if len(batch) > 1 else False, truncation=True,
    max_length=10000, return_token_type_ids=False).to(DEVICE_1)
    return encodings

@torch.inference_mode()
def get_logits(encodings):
    observer_logits = observer_model(**encodings.to(DEVICE_1)).logits
    performer_logits = performer_model(**encodings.to(DEVICE_1)).logits
    torch.cuda.synchronize()

    return observer_logits, performer_logits

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity(encoding, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    shifted_logits = shifted_logits.to("cpu")
    shifted_labels = shifted_labels.to("cpu")
    shifted_attention_mask = shifted_attention_mask.to("cpu")

    ppl = loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
    ppl = ppl.sum(1) / shifted_attention_mask.sum(1)
    
    return ppl.float().numpy()

def cross_perplexity(observer_logits, performer_logits, encoding):
    V = observer_logits.shape[-1]
    S = observer_logits.shape[-2]

    performer_probs = softmax_fn(performer_logits).view(-1, V).to("cpu")
    observer_scores = observer_logits.view(-1, V).to("cpu")
    
    xppl = loss_fn(observer_scores, performer_probs).view(-1, S)
    padding_mask = (encoding.input_ids != tokenizer.pad_token_id).type(torch.uint8)
    
    xppl = (xppl * padding_mask).sum(1) / padding_mask.sum(1)
    
    return xppl.to("cpu").float().numpy()

def two_level_normalize(scores, threshold_percentile=95):
    threshold = torch.quantile(scores, threshold_percentile/100)
    mask_high = scores >= threshold
    mask_low = scores < threshold
    
    normal_scores = torch.zeros_like(scores)
    if torch.any(mask_low):
        min_normal = torch.min(scores[mask_low])
        max_normal = torch.max(scores[mask_low])
        normal_scores[mask_low] = 0.7 * (scores[mask_low] - min_normal) / (max_normal - min_normal + 1e-10)
    
    if torch.any(mask_high):
        min_high = torch.min(scores[mask_high])
        max_high = torch.max(scores[mask_high])
        normal_scores[mask_high] = 0.7 + 0.3 * (scores[mask_high] - min_high) / (max_high - min_high + 1e-10)
    
    return normal_scores

def generate_html_output(tokens, scores, title):
    html = f"<h3>{title}</h3>\n<p>"
    for token, score in zip(tokens, scores.squeeze().tolist()):
        color_value = int(255 * score)
        html += f"<span style='background-color: rgb(255, {255-color_value}, {255-color_value}); color: black;'>{token}</span>"
    html += "</p>\n"
    return html

def save_results_to_file(text, ppl_scores, xppl_scores, bino_scores, tokens):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Text Analysis Results - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #333; }}
            h3 {{ color: #666; margin-top: 20px; }}
            p {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <h2>Text Analysis Results</h2>
        <p>Analysis performed on: {timestamp}</p>
        <h3>Original Text:</h3>
        <p>{text}</p>
    """
    
    html_content += generate_html_output(tokens, ppl_scores, "Perplexity Scores")
    html_content += generate_html_output(tokens, xppl_scores, "Cross-Perplexity Scores")
    html_content += generate_html_output(tokens, bino_scores, "Binocular Scores")
    
    html_content += """
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Results saved to {filename}")
    return filename

def analyze_text(text):
    encoding = tokenize([text])
    
    observer_logits, performer_logits = get_logits(encoding)
    
    S = observer_logits.shape[-2]
    V = observer_logits.shape[-1]
    
    shifted_logits = observer_logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    
    shifted_logits = shifted_logits.to("cpu")
    shifted_labels = shifted_labels.to("cpu")
    
    ppl = loss_fn(shifted_logits.transpose(1, 2), shifted_labels).float()
    
    tokens = [tokenizer.decode([tok], clean_up_tokenization_spaces=False) 
              for tok in encoding.input_ids.squeeze().tolist()]
    
    performer_probs = softmax_fn(performer_logits).view(-1, V).to("cpu")
    observer_scores = observer_logits.view(-1, V).to("cpu")
    
    xppl = loss_fn(observer_scores[:-1], performer_probs[:-1]).view(-1, S - 1).to("cpu").float()
    
    binocular_score = ppl / xppl
    normalized_binocular_score = two_level_normalize(binocular_score, threshold_percentile=95)
    
    filename = save_results_to_file(
        text,
        ppl,
        xppl,
        normalized_binocular_score,
        tokens
    )
    
    return filename

if __name__ == "__main__":
    # Example usage
    sample_text = '''### Советский Союз 1922–1939: Формирование, Реформы и Их Последствия\n\n#### Введение\n\nПериод с 1922 по 1939 год стал одним из наиболее трансформационных этапов в истории Советского Союза. Основание СССР, приход к власти Иосифа Виссарионовича Сталина, реализация масштабных экономических и социальных реформ, а также активная внешняя политика определили дальнейшее развитие страны. Этот период характеризуется как выдающимися достижениями в области индустриализации и науки, так и трагическими последствиями для миллионов граждан. Данное эссе рассматривает ключевые события и процессы, происходившие в СССР в указанный период, анализируя их влияние на внутреннюю структуру государства и его международное положение.\n\n#### Образование Советского Союза и Международное Признание\n\n14 декабря 1922 года состоялось официальное образование Союза Советских Социалистических Республик (СССР), объединившего Российскую, Украинскую, Белорусскую и Закавказскую Советские Социалистические Республики. Этот шаг был кульминацией революционных преобразований, начавшихся с Октябрьской революции 1917 года. Создание СССР символизировало объединение различных национальных республик под единым центральным правлением, что было направлено на укрепление внутренней стабильности и повышение международного статуса государства.'''
    
    result_file = analyze_text(sample_text)
    print(f"\nResults have been saved to {result_file}")