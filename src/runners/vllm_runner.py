"""vLLM generation wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.profiling.gpu_stats import snapshot_vram_bytes, track_gpu


def build_llm(model_path: str, model_cfg: Dict[str, Any]):
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=model_cfg.get("max_model_len", 32768),
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        enforce_eager=model_cfg.get("enforce_eager", True),
    )


def generate_one(
    llm,
    prompt: str,
    decoding: Dict[str, Any],
    seed: int,
    model_path: str,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = prompt

    params = SamplingParams(
        temperature=decoding.get("temperature", 0.6),
        top_p=decoding.get("top_p", 0.95),
        max_tokens=decoding.get("max_tokens", 32768),
        seed=seed,
    )

    vram_before = snapshot_vram_bytes()
    with track_gpu() as stats:
        outputs = llm.generate([rendered], params)
    vram_after = snapshot_vram_bytes()

    completion = outputs[0].outputs[0].text
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)

    return {
        "prompt": rendered,
        "completion": completion,
        "latency_sec": stats.latency_sec,
        "peak_vram_gb": max(vram_before, vram_after, stats.peak_vram_bytes) / (1024**3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def generate_batch(
    llm,
    prompts: List[str],
    decoding: Dict[str, Any],
    seed: int,
    model_path: str,
    use_chat_template: bool = True,
) -> List[Dict[str, Any]]:
    results = []
    for prompt in prompts:
        results.append(
            generate_one(
                llm,
                prompt,
                decoding=decoding,
                seed=seed,
                model_path=model_path,
                use_chat_template=use_chat_template,
            )
        )
    return results
