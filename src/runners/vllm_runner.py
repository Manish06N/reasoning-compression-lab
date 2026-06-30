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





def _safe_metric(metrics: Any, name: str) -> Any:

    if metrics is None:

        return None

    if isinstance(metrics, dict):

        return metrics.get(name)

    return getattr(metrics, name, None)


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

    vram_before_gb = vram_before / (1024**3)

    vram_after_gb = vram_after / (1024**3)

    vram_max_gb = max(vram_before, vram_after, stats.peak_vram_bytes) / (1024**3)

    per_latency = stats.latency_sec / len(prompts)

    per_energy = (stats.energy_joules / len(prompts)) if stats.energy_joules is not None else None



    results: List[Dict[str, Any]] = []

    for rendered_prompt, output in zip(rendered, outputs):

        choice = output.outputs[0]

        completion = choice.text

        prompt_tokens = len(output.prompt_token_ids)

        completion_tokens = len(choice.token_ids)

        total_tokens = prompt_tokens + completion_tokens

        metrics = getattr(output, "metrics", None)

        first_token_time = _safe_metric(metrics, "first_token_time")

        arrival_time = _safe_metric(metrics, "arrival_time")

        time_to_first_token = None

        if first_token_time is not None and arrival_time is not None:

            time_to_first_token = max(0.0, float(first_token_time) - float(arrival_time))

        decode_latency = max(per_latency - time_to_first_token, 0.0) if time_to_first_token is not None else per_latency

        decode_tokens_per_second = completion_tokens / decode_latency if decode_latency > 0 else None

        total_tokens_per_second = total_tokens / per_latency if per_latency > 0 else None

        seconds_per_output_token = per_latency / completion_tokens if completion_tokens else None

        tokens_per_joule = (completion_tokens / per_energy) if per_energy and per_energy > 0 else None

        finish_reason = getattr(choice, "finish_reason", None)

        stop_reason = getattr(choice, "stop_reason", None)

        max_tokens = decoding.get("max_tokens", 32768)

        results.append(

            {

                "prompt": rendered_prompt,

                "completion": completion,

                "latency_sec": per_latency,

                "time_to_first_token_sec": time_to_first_token,

                "peak_vram_gb": vram_max_gb,

                "vram_before_gb": vram_before_gb,

                "vram_after_gb": vram_after_gb,

                "vram_max_gb": vram_max_gb,

                "gpu_util_mean": stats.gpu_util_mean,

                "gpu_util_max": stats.gpu_util_max,

                "power_watts_mean": stats.power_watts_mean,

                "power_watts_max": stats.power_watts_max,

                "energy_joules": per_energy,

                "prompt_tokens": prompt_tokens,

                "completion_tokens": completion_tokens,

                "total_tokens": total_tokens,

                "tokens_per_second": total_tokens_per_second,

                "decode_tokens_per_second": decode_tokens_per_second,

                "seconds_per_output_token": seconds_per_output_token,

                "tokens_per_joule": tokens_per_joule,

                "finish_reason": finish_reason,

                "stop_reason": stop_reason,

                "truncated": completion_tokens >= max_tokens,

                "completion_chars": len(completion),

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


