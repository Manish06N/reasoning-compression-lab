"""Build validated raw inference rows (single- and multi-sample)."""

from __future__ import annotations

from typing import Any, Mapping

from src.schemas.provenance import input_text_hash, provenance_fields
from src.schemas.validate import validate_row_or_raise


def build_raw_response_row(
    *,
    row_base: Mapping[str, Any],
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    prompt_template_file: str,
    run_provenance: dict[str, Any] | None = None,
    batch_size: int = 1,
    sample_index: int | None = None,
    sample_seed: int | None = None,
    n_samples: int | None = None,
    telemetry_method: str = "measured",
    decoding_config_override: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Assemble one raw JSONL row with full provenance and optional schema check."""
    cell_id = cell["cell_id"]
    task = cell["task"]
    model_path = cell["model_path"]
    decoding = cell.get("decoding") or {}

    if run_provenance is None:
        run_provenance = provenance_fields(
            cell,
            prompt_template_file=prompt_template_file,
            batch_size=batch_size,
            n_samples=n_samples,
            max_model_len=cell.get("model", {}).get("max_model_len"),
        )

    row: dict[str, Any] = {
        **dict(row_base),
        **run_provenance,
        "input_text_hash": input_text_hash(str(row_base.get("problem", ""))),
        "prompt_template_file": prompt_template_file,
        "prompt": result["prompt"],
        "completion": result["completion"],
        "latency_sec": result.get("latency_sec"),
        "time_to_first_token_sec": result.get("time_to_first_token_sec"),
        "peak_vram_gb": result.get("peak_vram_gb"),
        "vram_before_gb": result.get("vram_before_gb"),
        "vram_after_gb": result.get("vram_after_gb"),
        "vram_max_gb": result.get("vram_max_gb"),
        "gpu_util_mean": result.get("gpu_util_mean"),
        "gpu_util_max": result.get("gpu_util_max"),
        "power_watts_mean": result.get("power_watts_mean"),
        "power_watts_max": result.get("power_watts_max"),
        "energy_joules": result.get("energy_joules"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "total_tokens": result.get("total_tokens"),
        "tokens_per_second": result.get("tokens_per_second"),
        "decode_tokens_per_second": result.get("decode_tokens_per_second"),
        "seconds_per_output_token": result.get("seconds_per_output_token"),
        "tokens_per_joule": result.get("tokens_per_joule"),
        "finish_reason": result.get("finish_reason"),
        "stop_reason": result.get("stop_reason"),
        "truncated": result.get("truncated"),
        "completion_chars": result.get("completion_chars"),
        "cell_id": cell_id,
        "model_path": model_path,
        "quant_config": cell.get("quant_config"),
        "task": task["task_name"],
        "seed": cell["seed"],
        "batch_size": batch_size,
        "telemetry_method": telemetry_method,
        "decoding_temperature": decoding.get("temperature"),
        "decoding_top_p": decoding.get("top_p"),
        "decoding_max_tokens": decoding.get("max_tokens"),
        "decoding_repetition_penalty": decoding.get("repetition_penalty"),
        "max_model_len": cell.get("model", {}).get("max_model_len"),
    }

    if sample_index is not None:
        row["sample_index"] = sample_index
    if sample_seed is not None:
        row["sample_seed"] = sample_seed
    if n_samples is not None:
        row["n_samples"] = n_samples
    if decoding_config_override:
        row["decoding_config"] = decoding_config_override

    if validate:
        validate_row_or_raise(row)

    return row
