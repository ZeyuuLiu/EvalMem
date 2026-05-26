from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory_eval.adapters import create_adapter_by_system, list_supported_memory_systems
from memory_eval.eval_core import EvaluatorConfig
from memory_eval.pipeline import PipelineConfig, ThreeProbeEvaluationPipeline


def _load_adapter(module_ref: str, class_name: str, config_json: str = ""):
    if module_ref.endswith(".py"):
        module_path = Path(module_ref).resolve()
        spec = importlib.util.spec_from_file_location("runtime_adapter_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load adapter module from {module_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"adapter class {class_name} not found in {module_ref}")
    if not config_json:
        return cls()

    raw_cfg = json.loads(config_json)
    if not isinstance(raw_cfg, dict):
        raise RuntimeError("--adapter-config-json must be a json object")

    # generic path: adapter(**kwargs)
    try:
        return cls(**raw_cfg)
    except TypeError:
        pass

    # common path in this repo: Adapter(config=AdapterConfig(...))
    cfg_name = f"{class_name}Config"
    cfg_cls = getattr(module, cfg_name, None)
    if cfg_cls is not None:
        cfg_obj = cfg_cls(**raw_cfg)
        return cls(config=cfg_obj)
    return cls(config=raw_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full three-probe evaluation pipeline.")
    parser.add_argument("--dataset", default="data/locomo10.json", help="Path to locomo dataset json")
    parser.add_argument("--output", default="outputs/eval_pipeline_results.json", help="Path to output json")
    parser.add_argument("--adapter-module", default="", help="Adapter module import path or .py file path")
    parser.add_argument("--adapter-class", default="", help="Adapter class name")
    parser.add_argument(
        "--memory-system",
        default="",
        help="Registered memory system key, e.g. o_mem. "
        "If provided, adapter-module/class are not required.",
    )
    parser.add_argument(
        "--list-memory-systems",
        action="store_true",
        help="List registered memory-system adapter keys and exit.",
    )
    parser.add_argument(
        "--adapter-config-json",
        default="",
        help='Optional adapter config json. Example: {"use_real_omem":true,"api_key":"...","base_url":"..."}',
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument("--fkey-source", choices=["rule", "llm"], default="rule")
    parser.add_argument("--tau-rank", type=int, default=5)
    parser.add_argument("--tau-snr", type=float, default=0.2)
    parser.add_argument("--neg-noise-threshold", type=float, default=0.15)
    parser.add_argument("--llm-assist", action="store_true", help="Force enable LLM assist")
    parser.add_argument("--no-llm-assist", action="store_true", help="Force disable LLM assist")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--allow-rule-fallback", action="store_true", help="Allow rule fallback when LLM judgement fails")
    parser.add_argument(
        "--allow-adapter-fallback",
        action="store_true",
        help="Allow adapter-level fallback instead of fail-fast on adapter call failure",
    )
    parser.add_argument(
        "--allow-empty-online-answer",
        action="store_true",
        help="Allow empty online answer in generation probe",
    )
    args = parser.parse_args()

    if args.list_memory_systems:
        print(json.dumps(list_supported_memory_systems(), ensure_ascii=False, indent=2))
        return 0

    raw_cfg: dict = {}
    if args.adapter_config_json:
        raw_cfg_obj = json.loads(args.adapter_config_json)
        if not isinstance(raw_cfg_obj, dict):
            raise RuntimeError("--adapter-config-json must be a json object")
        raw_cfg = raw_cfg_obj

    if args.memory_system:
        adapter = create_adapter_by_system(args.memory_system, raw_cfg)
    else:
        if not args.adapter_module or not args.adapter_class:
            raise RuntimeError("either --memory-system or both --adapter-module/--adapter-class are required")
        adapter = _load_adapter(args.adapter_module, args.adapter_class, args.adapter_config_json)

    use_llm_assist = True
    if args.no_llm_assist:
        use_llm_assist = False
    elif args.llm_assist:
        use_llm_assist = True

    eval_cfg = EvaluatorConfig(
        tau_rank=args.tau_rank,
        tau_snr=args.tau_snr,
        neg_noise_score_threshold=args.neg_noise_threshold,
        use_llm_assist=use_llm_assist,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        require_llm_judgement=not args.allow_rule_fallback,
        strict_adapter_call=not args.allow_adapter_fallback,
        disable_rule_fallback=not args.allow_rule_fallback,
        require_online_answer=not args.allow_empty_online_answer,
    )
    pipeline = ThreeProbeEvaluationPipeline(
        PipelineConfig(
            dataset_path=args.dataset,
            output_path=args.output,
            top_k=args.top_k,
            limit=args.limit,
            f_key_mode=args.fkey_source,
            evaluator_config=eval_cfg,
        )
    )
    report = pipeline.run(adapter)
    print("Pipeline finished.")
    print(f"Output: {args.output}")
    print(f"Total samples: {report['summary']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
