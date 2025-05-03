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

observer_name = "deepseek-ai/deepseek-llm-7b-base"
performer_name = "deepseek-ai/deepseek-llm-7b-chat"

THRESHOLD_RU = 0.962617

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

def binoculars_perplexity(encoding, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous() if encoding.get("attention_mask") is not None else None
    
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    
    if shifted_attention_mask is None:
        shifted_attention_mask = torch.ones_like(shifted_labels)
    
    gathered_log_probs = torch.gather(log_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    
    masked_log_probs = gathered_log_probs * shifted_attention_mask.float()
    
    sum_log_probs = masked_log_probs.sum(dim=1)
    token_count = shifted_attention_mask.sum(dim=1).float()
    
    avg_neg_log_probs = -sum_log_probs / token_count
    
    return avg_neg_log_probs.cpu().numpy()

def binoculars_entropy(observer_logits, performer_logits, encoding, pad_token_id):

    B = observer_logits.shape[0]
    S = observer_logits.shape[1]
    V = observer_logits.shape[2]
    
    attention_mask = (encoding.input_ids != pad_token_id).float()
    
    performer_probs = torch.softmax(performer_logits, dim=-1)
    observer_logits = observer_logits.view(B * S, V)
    performer_probs = performer_probs.view(B * S, V)
    
    ce_loss = loss_fn(observer_logits, performer_probs).view(B, S)
    
    ce_loss = ce_loss * attention_mask
    
    ce_loss = ce_loss.sum(dim=1) / attention_mask.sum(dim=1)
    
    return ce_loss.cpu().numpy()

def compute_binoculars_score(input_text):
    batch = [input_text] if isinstance(input_text, str) else input_text
    encodings = tokenize(batch)
    observer_logits, performer_logits = get_logits(encodings)
    
    ppl = binoculars_perplexity(encodings, performer_logits)
    x_ppl = binoculars_entropy(observer_logits, performer_logits, encodings, tokenizer.pad_token_id)
    
    binoculars_scores = ppl / x_ppl
    
    return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

def predict_binoculars(input_text, threshold=THRESHOLD_RU):
    binoculars_scores = np.array(compute_binoculars_score(input_text))
    
    pred = np.where(binoculars_scores < threshold,
                   "Most likely AI-generated",
                   "Most likely human-generated"
                   ).tolist()

    return pred[0] if isinstance(input_text, str) else pred

def adaptive_context_normalize(scores, window_size=5, sensitivity=2.0, min_threshold=0.0, max_threshold=1.0):
    scores_np = scores.cpu().numpy().squeeze()
    result = torch.zeros_like(scores)
    
    for i in range(len(scores_np)):
        start = max(0, i - window_size)
        end = min(len(scores_np), i + window_size + 1)
        window = scores_np[start:end]
        
        local_mean = window.mean()
        local_std = window.std() + 1e-10
        
        z_score = (scores_np[i] - local_mean) / local_std
        
        normalized = 1 / (1 + np.exp(-sensitivity * z_score))
        
        normalized = min_threshold + normalized * (max_threshold - min_threshold)
        
        result[0, i] = float(normalized)
    
    return result

def split_into_words(text):
    words = []
    word_indices = []
    
    pattern = r'(\w+|\s+|[^\w\s])'
    
    char_index = 0
    for match in re.finditer(pattern, text):
        word = match.group(0)
        start_index = match.start()
        end_index = match.end()
        
        words.append(word)
        word_indices.append((start_index, end_index))
        
    return words, word_indices

def map_tokens_to_words(tokens, words, word_indices, text):
    word_to_token_map = {}
    token_index = 0
    char_pos = 0
    
    for i, (word, (word_start, word_end)) in enumerate(zip(words, word_indices)):
        word_tokens = []
        word_token_indices = []
        
        while token_index < len(tokens) and char_pos < word_end:
            token = tokens[token_index]
            token_chars = len(token)
            
            word_tokens.append(token)
            word_token_indices.append(token_index)
            
            char_pos += token_chars
            token_index += 1
        
        word_to_token_map[i] = {
            "word": word,
            "token_indices": word_token_indices,
            "tokens": word_tokens
        }
    
    return word_to_token_map

def aggregate_token_scores_to_words(token_scores, word_to_token_map):
    word_scores = []
    
    for i in range(len(word_to_token_map)):
        if i not in word_to_token_map:
            word_scores.append(0.0)
            continue
            
        token_indices = word_to_token_map[i]["token_indices"]
        if not token_indices:
            word_scores.append(0.0)
            continue
        
        word_token_scores = [token_scores[j].item() for j in token_indices if j < len(token_scores)]
        if word_token_scores:
            word_score = sum(word_token_scores) / len(word_token_scores)
        else:
            word_score = 0.0
        
        word_scores.append(word_score)
    
    return word_scores

def generate_html_output(tokens, scores, title):
    html = f"<h3>{title}</h3>\n<p>"
    for token, score in zip(tokens, scores.squeeze().tolist()):
        color_value = int(255 * score)
        html += f"<span style='background-color: rgb(255, {255-color_value}, {255-color_value}); color: black;'>{token}</span>"
    html += "</p>\n"
    return html

def generate_word_based_html_output(words, word_scores, title):
    html = f"<h3>{title}</h3>\n<p>"
    for word, score in zip(words, word_scores):
        if word.isspace():
            html += word
        else:
            color_value = int(255 * score)
            html += f"<span style='background-color: rgb(255, {255-color_value}, {255-color_value}); color: black;'>{word}</span>"
    html += "</p>\n"
    return html

def generate_edit_html(text, words, word_scores, num_regions=3):
    word_with_scores = [(i, word, score) for i, (word, score) in enumerate(zip(words, word_scores)) if not word.isspace()]
    word_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    high_scoring_indices = [item[0] for item in word_with_scores[:min(len(word_with_scores), num_regions*5)]]
    high_scoring_indices.sort()
    
    regions = []
    current_region = None
    
    for i in range(len(words)):
        if i in high_scoring_indices:
            if current_region is None:
                current_region = {"start": i, "end": i}
            else:
                current_region["end"] = i
        else:
            if current_region is not None and i - current_region["end"] > 3:
                regions.append(current_region)
                current_region = None
    
    if current_region is not None:
        regions.append(current_region)
    
    region_scores = []
    for region in regions:
        start, end = region["start"], region["end"]
        region_words = [w for w in words[start:end+1] if not w.isspace()]
        if region_words:
            region_score = sum(word_scores[start:end+1]) / len(region_words)
            region_scores.append((region, region_score))
    
    region_scores.sort(key=lambda x: x[1], reverse=True)
    selected_regions = [r[0] for r in region_scores[:min(len(region_scores), num_regions)]]
    
    extended_regions = []
    for region in selected_regions:
        start = region["start"]
        end = region["end"]
        
        while start > 0:
            if words[start-1].isspace():
                start -= 1
                continue
                
            if words[start-1].strip().endswith(('.', '!', '?', ';', ':', ',')):
                break
                
            if end - (start-1) + 1 > 30:
                break
                
            start -= 1
        
        while end < len(words) - 1:
            if words[end+1].isspace():
                end += 1
                continue
                
            if words[end].strip().endswith(('.', '!', '?', ';', ':', ',')):
                end += 1
                break
                
            if (end+1) - start + 1 > 30:
                break
                
            end += 1
        
        word_count = sum(1 for w in words[start:end+1] if not w.isspace())
        if word_count < 2:
            continue
            
        if word_count > 30:
            curr_start = start
            word_count = 0
            for i in range(start, end + 1):
                if not words[i].isspace():
                    word_count += 1
                if word_count >= 30:
                    extended_regions.append({"start": curr_start, "end": i})
                    curr_start = i + 1
                    word_count = 0
            if curr_start <= end and word_count >= 2:
                extended_regions.append({"start": curr_start, "end": end})
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
        
        normal_text = ''.join(words[last_end:start])
        edit_text = ''.join(words[start:end+1])
        
        html += f"{normal_text}<span style='background-color: #ffcccc; color: black;'>{edit_text}</span>"
        last_end = end + 1
    
    if last_end < len(words):
        html += ''.join(words[last_end:])
    
    html += "</p>\n"
    return html

def place_edit_tags(text, words, word_scores, min_words=2, max_words=30, num_regions=3):
    word_with_scores = [(i, word, score) for i, (word, score) in enumerate(zip(words, word_scores)) if not word.isspace()]
    word_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    high_scoring_indices = [item[0] for item in word_with_scores[:min(len(word_with_scores), num_regions*5)]]
    high_scoring_indices.sort()
    
    regions = []
    current_region = None
    
    for i in range(len(words)):
        if i in high_scoring_indices:
            if current_region is None:
                current_region = {"start": i, "end": i}
            else:
                current_region["end"] = i
        else:
            if current_region is not None and i - current_region["end"] > 3:
                regions.append(current_region)
                current_region = None
    
    if current_region is not None:
        regions.append(current_region)
    
    region_scores = []
    for region in regions:
        start, end = region["start"], region["end"]
        region_words = [w for w in words[start:end+1] if not w.isspace()]
        if region_words:
            region_score = sum(word_scores[start:end+1]) / len(region_words)
            region_scores.append((region, region_score))
    
    region_scores.sort(key=lambda x: x[1], reverse=True)
    selected_regions = [r[0] for r in region_scores[:min(len(region_scores), num_regions)]]
    
    print(f"Top suspicious regions selected: {len(selected_regions)}")
    
    extended_regions = []
    for region in selected_regions:
        start = region["start"]
        end = region["end"]
        
        while start > 0:
            if words[start-1].isspace():
                start -= 1
                continue
                
            if words[start-1].strip().endswith(('.', '!', '?', ';', ':', ',')):
                break
                
            non_space_count = sum(1 for w in words[start-1:end+1] if not w.isspace())
            if non_space_count > max_words:
                break
                
            start -= 1
        
        while end < len(words) - 1:
            if words[end+1].isspace():
                end += 1
                continue
                
            if words[end].strip().endswith(('.', '!', '?', ';', ':', ',')):
                end += 1
                break
                
            non_space_count = sum(1 for w in words[start:end+2] if not w.isspace())
            if non_space_count > max_words:
                break
                
            end += 1
        
        non_space_count = sum(1 for w in words[start:end+1] if not w.isspace())
        if non_space_count < min_words:
            continue
            
        if non_space_count > max_words:
            curr_start = start
            word_count = 0
            for i in range(start, end + 1):
                if not words[i].isspace():
                    word_count += 1
                if word_count >= max_words:
                    if word_count >= min_words:
                        extended_regions.append({"start": curr_start, "end": i})
                    curr_start = i + 1
                    word_count = 0
            if curr_start <= end and word_count >= min_words:
                extended_regions.append({"start": curr_start, "end": end})
        else:
            extended_regions.append({"start": start, "end": end})
    
    print(f"Extended regions: {len(extended_regions)}")
    
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
    
    print(f"Final merged regions: {len(merged_regions)}")
    
    _, word_indices = split_into_words(text)
    
    result_text = text
    
    for region in reversed(merged_regions):
        start_word_idx = region["start"]
        end_word_idx = region["end"]
        
        if start_word_idx < len(word_indices) and end_word_idx < len(word_indices):
            start_pos = word_indices[start_word_idx][0]
            end_pos = word_indices[end_word_idx][1]
            
            result_text = (
                result_text[:end_pos] + 
                "</EDIT>" + 
                result_text[end_pos:])
            
            result_text = (
                result_text[:start_pos] + 
                "<EDIT>" + 
                result_text[start_pos:])
    
    return result_text

def analyze_text(text, add_edit_tags=False, num_regions=3):
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
    normalized_binocular_score = adaptive_context_normalize(binocular_score)
    
    words, word_indices = split_into_words(text)
    word_to_token_map = map_tokens_to_words(tokens, words, word_indices, text)
    word_scores = aggregate_token_scores_to_words(normalized_binocular_score.squeeze(), word_to_token_map)

    binoculars_verdict = predict_binoculars(text, THRESHOLD_RU)
    
    token_bino_html = generate_html_output(tokens, normalized_binocular_score, "Token-Based Binocular Scores")
    word_bino_html = generate_word_based_html_output(words, word_scores, "Word-Based Binocular Scores")
    
    
    edited_text = place_edit_tags(text, words, word_scores, num_regions=num_regions)
    html_edits = generate_edit_html(text, words, word_scores, num_regions=num_regions)
    
    result = {
        "tokens": tokens,
        "words": words,
        "ppl_scores": ppl,
        "xppl_scores": xppl,
        "binocular_scores": normalized_binocular_score,
        "word_scores": word_scores,
        "token_bino_html": token_bino_html,
        "word_bino_html": word_bino_html,
        "html_edits": html_edits,
        "edited_text": edited_text,
        "verdict": binoculars_verdict,
        "binoculars_score": compute_binoculars_score(text)
    }

    return result