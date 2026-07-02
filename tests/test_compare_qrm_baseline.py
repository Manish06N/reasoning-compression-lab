"""Tests for QRM baseline comparison gate."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compare_summary_includes_provenance_and_gate():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "n": 500,
        "pass_at_1": 0.93,
        "truncation_rate": 0.05,
        "parse_failure_rate": 0.02,
        "completion_tokens_mean": 5000,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)

    assert report["targets_provenance"]["yaml_sha256"]
    assert report["gate"]["reference_pct"] == 93.9
    assert report["gate"]["gate_type"] == "hard"
    assert report["gate"]["sanity_band_pct"] == [88.9, 98.9]
    assert report["hard_passed"] is True


def test_compare_summary_fails_broken_pipeline_band():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "pass_at_1": 0.07,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["hard_passed"] is False


def test_gpqa_sanity_gate_warns_not_hard_fails():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    task_targets = targets["models"]["DeepSeek-R1-Distill-Qwen-7B"]["GPQA-Diamond"]
    band_cfg = task_targets["pass_at_1_pct"]
    assert band_cfg["reference"] == 51.0
    assert band_cfg["tolerance_pp"] == 8.0
    assert task_targets["gate"] == "sanity"

    summary = {
        "cell_id": "level_c_qwen7b_fp8_gpqa_seed0",
        "task": "gpqa_diamond",
        "pass_at_1": 0.60,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["hard_passed"] is True
    pass_check = next(c for c in report["checks"] if c["metric"] == "pass_at_1_pct")
    assert pass_check["status"] == "SANITY_WARN"


def test_llama_math500_uses_table4_not_table1():
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    cfg = targets["models"]["DeepSeek-R1-Distill-Llama-8B"]["MATH-500"]["pass_at_1_pct"]
    assert "Table 4" in cfg["source"]
    assert cfg["reference"] == 91.0


def test_hard_gate_fails_incomplete_cell_n():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "task": "math500",
        "n": 350,
        "pass_at_1": 0.93,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    n_check = next(c for c in report["checks"] if c["metric"] == "n")
    assert n_check["status"] == "FAIL"
    assert report["hard_passed"] is False


def test_llama_gsm8k_marked_unused():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    assert targets["models"]["DeepSeek-R1-Distill-Llama-8B"]["GSM8K"]["status"] == "unused"

    summary = {
        "cell_id": "level_c_llama8b_bf16_gsm8k_seed0",
        "task": "gsm8k",
        "pass_at_1": 0.88,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["passed"] is None
    assert report["checks"][0]["status"] == "SKIP"


def test_resolve_model_key_from_model_id_without_cell_id():
    from scripts.compare_qrm_baseline import _resolve_model_key, compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "ambiguous_cell_name",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "task": "math500",
        "n": 500,
        "pass_at_1": 0.93,
    }
    assert _resolve_model_key(summary, targets) == "DeepSeek-R1-Distill-Qwen-7B"
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["model_key"] == "DeepSeek-R1-Distill-Qwen-7B"
    assert report["hard_passed"] is True


def test_resolve_model_key_llama_from_model_id():
    from scripts.compare_qrm_baseline import _resolve_model_key
    from src.runners.config_utils import load_yaml

    targets = load_yaml(ROOT / "configs/baselines/qrm_literature_targets.yaml")
    summary = {
        "cell_id": "unknown",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "task": "math500",
    }
    assert _resolve_model_key(summary, targets) == "DeepSeek-R1-Distill-Llama-8B"


def test_infer_model_family_from_summary_model_id():
    from scripts.build_paper_tables import infer_model_family

    summary = {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "cell_id": "x"}
    assert infer_model_family(summary, None) == "Qwen-7B"


def test_compare_summary_skips_fp8_against_bf16_target():
    from scripts.compare_qrm_baseline import compare_summary
    from src.runners.config_utils import load_yaml

    targets_path = ROOT / "configs/baselines/qrm_literature_targets.yaml"
    targets = load_yaml(targets_path)
    summary = {
        "cell_id": "level_a_qwen7b_fp8_math500_seed0",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quant_config": "fp8",
        "prompt_profile": "reproduction",
        "task": "math500",
        "n": 500,
        "pass_at_1": 0.93,
    }
    report = compare_summary(summary, targets, targets_path=targets_path)
    assert report["passed"] is None
    assert report["checks"][0]["status"] == "SKIP"
    assert "quant_config mismatch" in report["checks"][0]["message"]


def test_resolve_model_key_qwen15b_cell_id():
    from scripts.compare_qrm_baseline import _resolve_model_key
    from src.runners.config_utils import load_yaml

    targets = load_yaml(ROOT / "configs/baselines/qrm_literature_targets.yaml")
    summary = {"cell_id": "level_c_qwen15b_bf16_math500_seed0", "task": "math500"}
    assert _resolve_model_key(summary, targets) == "DeepSeek-R1-Distill-Qwen-1.5B"


def test_resolve_model_key_gptq3_returns_none():
    from scripts.compare_qrm_baseline import _resolve_model_key
    from src.runners.config_utils import load_yaml

    targets = load_yaml(ROOT / "configs/baselines/qrm_literature_targets.yaml")
    summary = {
        "cell_id": "level_b_qwen7b_gptq3_math500_seed0",
        "quant_config": "gptq3",
        "task": "math500",
    }
    assert _resolve_model_key(summary, targets) is None
