from binoculars import Binoculars

def initialize_binoculars():
    chat_model_pair = {
        "observer": "deepseek-ai/deepseek-llm-7b-base",
        "performer": "deepseek-ai/deepseek-llm-7b-chat"
    }

    coder_model_pair = {
        "observer": "deepseek-ai/deepseek-llm-7b-base",
        "performer": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    }
    
    print("Initializing Binoculars models...")
    
    bino_chat = Binoculars(
        mode="accuracy", 
        observer_name_or_path=chat_model_pair["observer"],
        performer_name_or_path=chat_model_pair["performer"],
        max_token_observed=2048
    )

    bino_coder = Binoculars(
        mode="accuracy", 
        observer_name_or_path=coder_model_pair["observer"],
        performer_name_or_path=coder_model_pair["performer"],
        max_token_observed=2048
    )
    
    return bino_chat, bino_coder

def compute_scores(text, bino_chat=None, bino_coder=None):
    scores = {}
    
    if bino_chat:
        #print("Computing score_chat...")
        scores['score_chat'] = bino_chat.compute_score(text)
    
    if bino_coder:
        #print("Computing score_coder...")
        scores['score_coder'] = bino_coder.compute_score(text)
    
    return scores 