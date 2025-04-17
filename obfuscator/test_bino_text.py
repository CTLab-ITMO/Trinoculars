from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE_1 = "cuda:0"

torch.set_grad_enabled(False)

observer_name = "deepseek-ai/deepseek-llm-7b-base"
performer_name = "deepseek-ai/deepseek-llm-7b-chat"

try:
    logger.info("Loading tokenizers...")
    identical_tokens = (AutoTokenizer.from_pretrained(observer_name).vocab ==
                        AutoTokenizer.from_pretrained(performer_name).vocab)
    
    logger.info("Loading observer model...")
    observer_model = AutoModelForCausalLM.from_pretrained(observer_name,
                                                       device_map={"": DEVICE_1},
                                                       trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16)

    logger.info("Loading performer model...")
    performer_model = AutoModelForCausalLM.from_pretrained(performer_name,
                                                         device_map={"": DEVICE_1},
                                                         trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16)

    observer_model.eval()
    performer_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(observer_name)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def generate_console_output(tokens, scores):
    output = tokens[0]
    for token, score in zip(tokens[1:], scores.squeeze().tolist()):
        # Convert score to color intensity (0-255)
        color_value = int(255 * score)
        # ANSI escape code for background color
        color_code = f"\033[48;2;255;{255-color_value};{255-color_value}m"
        # Reset color code
        reset_code = "\033[0m"
        output += f"{color_code}{token}{reset_code}"
    return output

# redefine to handle batch of strings
def tokenize(batch):
    encodings = tokenizer(batch, return_tensors="pt", 
    padding="longest" if len(batch) > 1 else False, truncation=True,
    max_length=10000, return_token_type_ids=False).to(DEVICE_1)
    return encodings

# redefinition with cuda sync
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

    ppl = loss_fn(shifted_logits.transpose(1, 2).to("cpu"), shifted_labels) * shifted_attention_mask
    ppl = ppl.sum(1) / shifted_attention_mask.sum(1)
    
    return ppl.to("cpu").float().numpy()

def cross_perplexity(observer_logits, performer_logits, encoding):
    V = observer_logits.shape[-1]
    S = observer_logits.shape[-2]

    performer_probs = softmax_fn(performer_logits).view(-1, V).to("cpu")
    observer_scores = observer_logits.view(-1, V).to("cpu")
    
    xppl = loss_fn(observer_scores, performer_probs).view(-1, S)
    padding_mask = (encoding.input_ids != tokenizer.pad_token_id).type(torch.uint8)
    
    xppl = (xppl * padding_mask).sum(1) / padding_mask.sum(1)
    
    return xppl.to("cpu").float().numpy()

def binocular_score(text):
    if not text:
        logger.warning("Empty text provided")
        return []
        
    batch = [text] if isinstance(text, str) else text
    try:
        encodings = tokenize(batch)
        observer_logits, performer_logits = get_logits(encodings)
        ppl = perplexity(encodings, observer_logits)
        xppl = cross_perplexity(observer_logits, performer_logits, encodings)
        return (ppl / xppl).tolist()
    except Exception as e:
        logger.error(f"Error calculating binocular score: {str(e)}")
        raise

human = '''The healthcare industry typically draws sufficient attention to patients' education, especially when it comes to representatives of minority groups. That is why the article by McCurley et al. (2017) offers valuable information. The researchers demonstrate that Hispanic individuals deal with improved diabetes prevention when they participate in individual and group face-to-face sessions (McCurley et al., 2017). I believe that there is an apparent reason why such positive outcomes are achieved. It seems that face-to-face interventions are effective because patients have an opportunity to ask questions if they require explanations. Simultaneously, such educational sessions demonstrate that a patient is not unique with such a health issue. As a result, such interventions can improve people's morale, which, in turn, will lead to increased motivation to take preventive measures and protect health.'''

encoding = tokenize([human])

observer_logits, performer_logits = get_logits(encoding)

S = observer_logits.shape[-2]
V = observer_logits.shape[-1]

shifted_logits = observer_logits[..., :-1, :].contiguous()
shifted_labels = encoding.input_ids[..., 1:].contiguous()

ppl = loss_fn(shifted_logits.transpose(1, 2).to("cpu"), shifted_labels).float()

normalized_ppl = ppl / torch.max(ppl)

tokens = [tokenizer.decode([tok], clean_up_tokenization_spaces=False) for tok in encoding.input_ids.squeeze().tolist()]
console_output = generate_console_output(tokens, normalized_ppl)
print("\nPerplexity scores:")
print(console_output)

performer_probs = softmax_fn(performer_logits).view(-1, V).to("cpu")
observer_scores = observer_logits.view(-1, V).to("cpu")

xppl = loss_fn(observer_scores[:-1], performer_probs[:-1]).view(-1, S - 1).to("cpu").float()
normalized_xppl = xppl / torch.max(xppl)

console_output = generate_console_output(tokens, normalized_xppl)
print("\nCross-perplexity scores:")
print(console_output)

binocular_score = normalized_ppl / normalized_xppl
normalized_binocular_score = binocular_score / torch.max(binocular_score)

console_output = generate_console_output(tokens, normalized_binocular_score)
print("\nBinocular scores:")
print(console_output)