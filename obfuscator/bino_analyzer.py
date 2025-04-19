from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import os
import re

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

def generate_edit_html(text, tokens, scores, threshold=0.7):
    scores_list = scores.squeeze().tolist()
    regions = []
    current_region = None
    
    for i, (token, score) in enumerate(zip(tokens, scores_list)):
        if score >= threshold:
            if current_region is None:
                current_region = {"start": i, "end": i}
            else:
                current_region["end"] = i
        else:
            if current_region is not None:
                regions.append(current_region)
                current_region = None
                
    if current_region is not None:
        regions.append(current_region)
    
    extended_regions = []
    for region in regions:
        start = region["start"]
        end = region["end"]
        
        if end - start + 1 < 2:
            continue
        
        while start > 0:
            token = tokens[start]
            prev_token = tokens[start-1]
            if re.search(r'[.!?;:,]$', prev_token) or re.search(r'^\n', token):
                break
            if '####' in prev_token:
                break
            if end - (start-1) + 1 > 30:
                break
            start -= 1
        
        while end < len(tokens) - 1:
            token = tokens[end]
            next_token = tokens[end+1]
            if re.search(r'[.!?;:,]$', token) or re.search(r'^\n', next_token):
                end += 1
                break
            if '####' in next_token:
                break
            if (end+1) - start + 1 > 30:
                break
            end += 1
        
        token_count = end - start + 1
        if token_count < 2:
            continue
            
        if token_count > 30:
            for i in range(start, end + 1, 30):
                chunk_end = min(i + 30 - 1, end)
                if chunk_end - i + 1 >= 2:
                    extended_regions.append({"start": i, "end": chunk_end})
        else:
            extended_regions.append({"start": start, "end": end})
    
    if extended_regions:
        extended_regions.sort(key=lambda r: r["start"])
        merged_regions = [extended_regions[0]]
        
        for region in extended_regions[1:]:
            prev_region = merged_regions[-1]
            if region["start"] <= prev_region["end"] + 1:
                prev_region["end"] = max(prev_region["end"], region["end"])
            else:
                merged_regions.append(region)
    else:
        merged_regions = []
    
    html = "<h3>Text with Highlighted Edits</h3>\n<p>"
    last_end = 0
    
    for region in merged_regions:
        start, end = region["start"], region["end"]
        
        normal_text = ''.join(tokens[last_end:start])
        edit_text = ''.join(tokens[start:end+1])
        
        html += f"{normal_text}<span style='background-color: #ffcccc; color: black;'>{edit_text}</span>"
        last_end = end + 1
    
    if last_end < len(tokens):
        html += ''.join(tokens[last_end:])
    
    html += "</p>\n"
    return html

def place_edit_tags(text, tokens, scores, threshold=0.7, min_tokens=2, max_tokens=30):
    scores_list = scores.squeeze().tolist()
    regions = []
    current_region = None
    
    for i, (token, score) in enumerate(zip(tokens, scores_list)):
        if score >= threshold:
            if current_region is None:
                current_region = {"start": i, "end": i}
            else:
                current_region["end"] = i
        else:
            if current_region is not None:
                regions.append(current_region)
                current_region = None
    
    if current_region is not None:
        regions.append(current_region)
    
    extended_regions = []
    for region in regions:
        start = region["start"]
        end = region["end"]
        
        if end - start + 1 < min_tokens:
            continue
        
        while start > 0:
            token = tokens[start]
            prev_token = tokens[start-1]
            if re.search(r'[.!?;:,]$', prev_token) or re.search(r'^\n', token):
                break
            if '####' in prev_token:
                break
            if end - (start-1) + 1 > max_tokens:
                break
            start -= 1
        
        while end < len(tokens) - 1:
            token = tokens[end]
            next_token = tokens[end+1]
            if re.search(r'[.!?;:,]$', token) or re.search(r'^\n', next_token):
                end += 1
                break
            if '####' in next_token:
                break
            if (end+1) - start + 1 > max_tokens:
                break
            end += 1
        
        token_count = end - start + 1
        if token_count < min_tokens:
            continue
            
        if token_count > max_tokens:
            for i in range(start, end + 1, max_tokens):
                chunk_end = min(i + max_tokens - 1, end)
                if chunk_end - i + 1 >= min_tokens:
                    extended_regions.append({"start": i, "end": chunk_end})
        else:
            extended_regions.append({"start": start, "end": end})
    
    if extended_regions:
        extended_regions.sort(key=lambda r: r["start"])
        merged_regions = [extended_regions[0]]
        
        for region in extended_regions[1:]:
            prev_region = merged_regions[-1]
            if region["start"] <= prev_region["end"] + 1:
                prev_region["end"] = max(prev_region["end"], region["end"])
            else:
                merged_regions.append(region)
    else:
        merged_regions = []
    
    token_text = ''.join(tokens)
    result_text = token_text
    
    char_positions = []
    pos = 0
    for token in tokens:
        char_positions.append(pos)
        pos += len(token)
    
    for region in reversed(merged_regions):
        start_pos = char_positions[region["start"]]
        end_pos = char_positions[region["end"]] + len(tokens[region["end"]])
        
        result_text = (
            result_text[:end_pos] + 
            "</EDIT>" + 
            result_text[end_pos:])
        
        result_text = (
            result_text[:start_pos] + 
            "<EDIT>" + 
            result_text[start_pos:])
    
    return result_text

def analyze_text(text, add_edit_tags=False, edit_threshold=0.7):
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
    
    ppl_html = generate_html_output(tokens, ppl, "Perplexity Scores")
    xppl_html = generate_html_output(tokens, xppl, "Cross-Perplexity Scores") 
    bino_html = generate_html_output(tokens, normalized_binocular_score, "Binocular Scores")
    
    edited_text = None
    html_edits = None
    
    if add_edit_tags:
        edited_text = place_edit_tags(text, tokens, normalized_binocular_score, threshold=edit_threshold)
        html_edits = generate_edit_html(text, tokens, normalized_binocular_score, threshold=edit_threshold)
    
    text_with_scores = ''.join([f"{token}" for token, score in zip(tokens, normalized_binocular_score.squeeze().tolist())])
    
    result = {
        "tokens": tokens,
        "ppl_scores": ppl,
        "xppl_scores": xppl,
        "binocular_scores": normalized_binocular_score,
        "ppl_html": ppl_html,
        "xppl_html": xppl_html,
        "bino_html": bino_html,
        "text_with_scores": text_with_scores
    }
    
    if html_edits:
        result["html_edits"] = html_edits
    
    if edited_text:
        result["edited_text"] = edited_text
    
    return result

if __name__ == "__main__":
    sample_text = '''### Советский Союз 1922–1939: Формирование, Реформы и Их Последствия #### Введение Период с 1922 по 1939 год стал одним из наиболее трансформационных этапов в истории Советского Союза. Основание СССР, приход к власти Иосифа Виссарионовича Сталина, реализация масштабных экономических и социальных реформ, а также активная внешняя политика определили дальнейшее развитие страны. Этот период характеризуется как выдающимися достижениями в области индустриализации и науки, так и трагическими последствиями для миллионов граждан.'''
    result = analyze_text(sample_text, add_edit_tags=True, edit_threshold=0.7)
    print("Analysis completed.")