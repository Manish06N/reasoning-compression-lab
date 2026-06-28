"""vLLM generation wrapper."""



from __future__ import annotations



from typing import Any, Dict, List



from src.profiling.gpu_stats import snapshot_vram_bytes, track_gpu





def _ensure_tokenizer_compatibility() -> None:

    """Bridge vLLM 0.8.x with Transformers versions that dropped this property."""

    from transformers import PreTrainedTokenizerBase



    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

        return



    @property

    def all_special_tokens_extended(self: PreTrainedTokenizerBase) -> list[str]:

        return self.all_special_tokens



    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended





def render_prompt(prompt: str, model_path: str, use_chat_template: bool = True) -> str:

    from transformers import AutoTokenizer



    _ensure_tokenizer_compatibility()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_chat_template and getattr(tokenizer, "chat_template", None):

        messages = [{"role": "user", "content": prompt}]

        return tokenizer.apply_chat_template(

            messages, tokenize=False, add_generation_prompt=True

        )

    return prompt





def build_llm(model_path: str, model_cfg: Dict[str, Any]):

    _ensure_tokenizer_compatibility()



    from vllm import LLM



    llm_kwargs: Dict[str, Any] = {

        "model": model_path,

        "dtype": model_cfg.get("dtype", "bfloat16"),

        "max_model_len": model_cfg.get("max_model_len", 32768),

        "tensor_parallel_size": model_cfg.get("tensor_parallel_size", 1),

        "enforce_eager": model_cfg.get("enforce_eager", True),

        "trust_remote_code": model_cfg.get("trust_remote_code", True),

    }

    if model_cfg.get("quantization"):

        llm_kwargs["quantization"] = model_cfg["quantization"]

    if model_cfg.get("kv_cache_dtype"):

        llm_kwargs["kv_cache_dtype"] = model_cfg["kv_cache_dtype"]

    if model_cfg.get("gpu_memory_utilization") is not None:

        llm_kwargs["gpu_memory_utilization"] = model_cfg["gpu_memory_utilization"]



    return LLM(**llm_kwargs)





def _sampling_params(decoding: Dict[str, Any], seed: int):

    from vllm import SamplingParams



    return SamplingParams(

        temperature=decoding.get("temperature", 0.6),

        top_p=decoding.get("top_p", 0.95),

        max_tokens=decoding.get("max_tokens", 32768),

        seed=seed,

    )





def generate_one(

    llm,

    prompt: str,

    decoding: Dict[str, Any],

    seed: int,

    model_path: str,

    use_chat_template: bool = True,

) -> Dict[str, Any]:

    results = generate_chunk(

        llm,

        [prompt],

        decoding=decoding,

        seed=seed,

        model_path=model_path,

        use_chat_template=use_chat_template,

    )

    return results[0]





def generate_chunk(

    llm,

    prompts: List[str],

    decoding: Dict[str, Any],

    seed: int,

    model_path: str,

    use_chat_template: bool = True,

) -> List[Dict[str, Any]]:

    """Run one vLLM.generate() call for a batch of prompts (continuous batching)."""

    if not prompts:

        return []



    rendered = [

        render_prompt(p, model_path, use_chat_template=use_chat_template) for p in prompts

    ]

    params = _sampling_params(decoding, seed)



    vram_before = snapshot_vram_bytes()

    with track_gpu() as stats:

        outputs = llm.generate(rendered, params)

    vram_after = snapshot_vram_bytes()

    peak_vram = max(vram_before, vram_after, stats.peak_vram_bytes) / (1024**3)

    per_latency = stats.latency_sec / len(prompts)



    results: List[Dict[str, Any]] = []

    for rendered_prompt, output in zip(rendered, outputs):

        completion = output.outputs[0].text

        results.append(

            {

                "prompt": rendered_prompt,

                "completion": completion,

                "latency_sec": per_latency,

                "peak_vram_gb": peak_vram,

                "prompt_tokens": len(output.prompt_token_ids),

                "completion_tokens": len(output.outputs[0].token_ids),

            }

        )

    return results





def generate_batch(

    llm,

    prompts: List[str],

    decoding: Dict[str, Any],

    seed: int,

    model_path: str,

    use_chat_template: bool = True,

    batch_size: int = 1,

) -> List[Dict[str, Any]]:

    """Generate in vLLM-native chunks (batch_size prompts per llm.generate call)."""

    if batch_size < 1:

        batch_size = 1

    results: List[Dict[str, Any]] = []

    for start in range(0, len(prompts), batch_size):

        chunk = prompts[start : start + batch_size]

        results.extend(

            generate_chunk(

                llm,

                chunk,

                decoding=decoding,

                seed=seed,

                model_path=model_path,

                use_chat_template=use_chat_template,

            )

        )

    return results


