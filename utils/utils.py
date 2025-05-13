import numpy as np
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
# from neural_che.neural_cherche import  models, utils, retrieve


def make_conversation(example,SYSTEM_PROMPT,offcial_prompt):
    if not offcial_prompt:
        return {'prompt':SYSTEM_PROMPT+example["prompt"]+" [NEW QUERY]:"}
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
    }


from tqdm import tqdm

def generate_reformulations(
    queries: dict,
    model,
    tokenizer,
    device: str,
    make_conversation,
    system_prompt: str,
    offcial_prompt=False,
    max_new_tokens: int = 50,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[str]:
    """
    For each query in `queries`, formats the prompt, runs generation, and decodes the output.
    
    Args:
      queries:       dict of {query_id: query_text}.
      model:         a Transformers model with a .generate() method.
      tokenizer:     matching tokenizer, with `.apply_chat_template()` and `.decode()`.
      device:        device string, e.g. "mps" or "cuda".
      make_conversation: function(example, system_prompt, non_stra) → {"prompt": ...}
      system_prompt: the system prompt string to use in conversation_fn.
      offcial_prompt: 
      max_new_tokens: max tokens to generate.
      do_sample:     whether to sample.
      temperature:   sampling temperature.
      top_k:         sampling top_k.
    
    Returns:
      A list of generated reformulation strings, in the same order as queries.items().
    """
    outputs = []
    for _, query_text in tqdm(queries.items(), desc="Generating"):
        # 1) Build the conversation
        example      = {"prompt": query_text}
        conv_payload = make_conversation(example, system_prompt,offcial_prompt=offcial_prompt)
        
        # 2) Render full prompt
        full_prompt = tokenizer.apply_chat_template(
            conv_payload["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3) Tokenize & move to device
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        # 4) Generate
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        
        # 5) Decode the new tokens only
        prompt_len     = inputs["input_ids"].shape[1]
        new_tokens     = gen[0][prompt_len:]
        reformulation  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(reformulation)
    
    return outputs


def eval_reformulations(
    completions: list[str],
    query_ids: list[str],
    qrels,
    queries: dict = None,
    searcher=None,
    k: int = 100
):
    """
    Evaluate reformulated queries using NDCG@{1,10,100}.
    Optionally compares against the original queries.

    Args:
        completions: List of reformulated query strings.
        query_ids: List of corresponding query IDs.
        queries: Original queries dictionary (optional).
        searcher: Pyserini LuceneSearcher instance.
        k: Number of retrieved documents.

    Returns:
        Tuple of:
            - ndcg, recall, precision for reformulated
            - ndcg, recall, precision for original (if `queries` is provided)
    """
    from collections import defaultdict

    scores_ref = {}
    scores_orig = {} if queries else None

    for query, qid in zip(completions, query_ids):
        # Reformulated query retrieval
        hits = searcher.search(query, k=k, fields={"contents": 1.0})
        scores_ref[qid] = {
            d.docid: d.score for d in hits if d.docid != str(qid)
        }

        # Original query retrieval (optional)
        if queries:
            hits_orig = searcher.search(queries[qid], k=k, fields={"contents": 1.0})
            scores_orig[qid] = {
                d.docid: d.score for d in hits_orig if d.docid != str(qid)
            }

    # Evaluate reformulated queries
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, scores_ref, [1, 10, 100])

    # Evaluate original queries if provided
    if queries:
        ndcg_orig, map_orig, recall_orig, p_orig = EvaluateRetrieval.evaluate(
            qrels, scores_orig, [1, 10, 100]
        )
        return ndcg, recall, p, ndcg_orig, recall_orig, p_orig

    return ndcg, recall, p



