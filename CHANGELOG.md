# Changelog

All notable changes to this project are documented in this file.

## [0.6.6] - 2026-03-24

### Changed

1. Increased stable O-Mem async timeout defaults for real evaluation:
   - `OMemAdapterConfig.async_call_timeout_sec`: `120.0 -> 180.0`
2. Increased stable smoke-run timeout to align with longer native O-Mem topic merge stages.

## [0.6.5] - 2026-03-19

### Added

1. Preserved original `system/O-Mem` and created a parallel patched runtime:
   - `system/O-Mem-StableEval`
2. New adapter registry target for patched O-Mem runtime:
   - `o_mem_stable_eval`
3. New code-audit docs for evaluation internals and O-Mem completion review:
   - `docs/code-audit/2026-03-19/eval-layer-bottom-up-implementation-audit.md`
   - `docs/code-audit/2026-03-19/o-mem-adapter-completion-audit.md`
   - `docs/code-audit/2026-03-19/o-mem-stable-eval-internal-fix-notes.md`

### Changed

1. Patched `system/O-Mem-StableEval/memory_chain` to remove infinite retry loops in key LLM-JSON paths.
2. Added robust JSON parsing helpers that tolerate markdown code fences and extracted JSON substrings.
3. Replaced open-ended retry loops in:
   - `memory_manager.py`
   - `episodic_memory.py`
   - `persona_memory.py`
   - `memory.py`
4. Stable-eval path now uses bounded retries and explicit failure instead of endless retry-on-parse-error behavior.

## [0.6.4] - 2026-03-19

### Changed

1. O-Mem adapter adds async fail-fast timeout guard to avoid indefinite hangs:
   - new config: `async_call_timeout_sec` (default `120.0`)
   - `receive_message(...)` in ingestion path is wrapped with timeout
   - `generate_system_response(...)` calls for online/oracle generation are wrapped with timeout
2. Strict runs now fail with explicit timeout errors instead of blocking forever on unstable internal O-Mem retry loops.

## [0.6.3] - 2026-03-19

### Added

1. Two-stage O-Mem evaluation orchestrator:
   - `scripts/run_omem_two_stage_eval.py`
   - Stage A: run baseline online answers on first N questions and compute accuracy
   - Stage B: run strict attribution on all incorrect questions + sampled correct questions
2. Unified output report with explicit model list and hyperparameters:
   - O-Mem generation model
   - baseline correctness-judge model
   - eval-layer LLM model
   - embedding model path and retrieval/runtime hyperparameters

## [0.6.2] - 2026-03-19

### Changed

1. O-Mem adapter retrieval is now native in real mode:
   - `retrieve_original(...)` calls O-Mem `retrieve_from_memory_soft_segmentation(...)`
   - retrieval output is normalized from O-Mem native retrieval payload (context/facts/attributes channels)
2. O-Mem adapter online answer is now native in real mode:
   - `generate_online_answer(...)` now uses O-Mem native retrieval result + `generate_system_response(...)`
3. Added native retrieval cache in run context to keep retrieval/generation consistency per query.
4. Added full bilingual evaluation flow guide for one memory system:
   - `docs/architecture/one-memory-system-full-evaluation-flow-bilingual-v0.6.2.md`

## [0.6.1] - 2026-03-19

### Added

1. Memory-system adapter registry for per-system dedicated implementation:
   - `src/memory_eval/adapters/registry.py`
2. New architecture specification for adapter fidelity and fail-fast protocol:
   - `docs/architecture/adapter-requirements-and-fidelity-protocol-bilingual-v0.6.1.md`
3. New bilingual implementation walkthrough for strict three-probe evaluation:
   - `docs/architecture/three-probe-framework-implementation-bilingual-v0.6.1.md`

### Changed

1. Evaluator strict policy defaults are now enabled:
   - require structured LLM judgement
   - fail on adapter-call exceptions
   - disable rule fallback in strict mode
   - require non-empty online answer by default
2. Encoding probe:
   - strict-mode now raises on LLM judgement failure / invalid state
   - no evaluator-side fallback scan in strict mode
   - adapter hybrid retrieval errors are surfaced when strict mode is enabled
3. Retrieval probe:
   - strict-mode requires POS/NEG LLM judgement success
   - LLM state can be treated as primary state source in strict mode
   - rule fallback is blocked in strict mode
4. Generation probe:
   - strict-mode requires both LLM answer-judge and tri-answer comparison success
   - adapter online-answer errors are surfaced in strict mode
5. O-Mem adapter:
   - real O-Mem ingest failure no longer silently degrades unless explicitly allowed
6. Pipeline:
   - now records per-query `EVAL_ERROR` entries instead of silently dropping context
   - output includes `adapter_manifest` and explicit `errors` list
7. CLI:
   - supports `--memory-system` + registry-based adapter creation
   - supports strict-policy control flags (`--allow-rule-fallback`, etc.)

## [0.6.0] - 2026-03-18

### Added

1. Bilingual three-probe implementation logic document:
   - `docs/architecture/three-probe-implementation-logic-bilingual-v0.6.0.md`
2. Bilingual fixed adapter interface specification:
   - `docs/architecture/adapter-fixed-interface-spec-bilingual-v0.6.0.md`

### Changed

1. Encoding probe now supports adapter-side hybrid candidate retrieval and structured LLM storage judgement.
2. Retrieval probe now uses POS/NEG split LLM judging with token-overlap SNR metrics.
3. Generation probe now supports tri-answer comparison (`A_online`, `A_oracle`, `A_gold`) with comparative evidence output.
4. Adapter protocol expanded with optional enhancement methods:
   - `hybrid_retrieve_candidates(...)`
   - `generate_online_answer(...)`
5. O-Mem adapter updated to implement hybrid candidate retrieval and online-answer export.

## [0.5.2] - 2026-03-18

### Added

1. Bilingual real-run report for O-Mem on LOCOMO sample0:
   - `docs/architecture/o-mem-real-run-sample0-report-v0.5.2.md`

### Changed

1. README design-doc list updated with the new real-run report.

## [0.5.1] - 2026-03-17

### Changed

1. O-Mem adapter audit now scans both `system/O-Mem` and `src/memory_eval/adapters`.
2. O-Mem adapter compliance status updated to reflect protocol implementation in `src/memory_eval/adapters/o_mem_adapter.py`.
3. Fixed duplicate time/speaker prefix issue in O-Mem adapter memory export formatting.
4. Pipeline CLI now supports `--adapter-config-json` and explicit `--no-llm-assist`.

## [0.5.0] - 2026-03-17

### Added

1. End-to-end three-probe pipeline module:
   - `src/memory_eval/pipeline/runner.py`
   - `src/memory_eval/pipeline/__init__.py`
2. Full pipeline CLI script:
   - `scripts/run_eval_pipeline.py`
3. Mock end-to-end pipeline test script:
   - `scripts/test_eval_pipeline_mock.py`

### Changed

1. `memory_eval` package now exports `pipeline`.
2. README updated with full pipeline usage.

## [0.4.2] - 2026-03-17

### Added

1. Adapter-aware unified evaluator entrypoint:
   - `ParallelThreeProbeEvaluator.evaluate_with_adapters(...)`
2. Eval-layer final implementation document:
   - `docs/architecture/eval-layer-final-implementation-v0.4.2.md`
3. O-Mem adapter compliance audit document:
   - `docs/architecture/o-mem-adapter-compliance-audit-v0.4.2.md`
4. O-Mem adapter audit script:
   - `scripts/audit_o_mem_adapter.py`

### Changed

1. README updated with v0.4.2 scope and O-Mem audit command.
2. Evaluation layer now exposes both trace-based and adapter-aware evaluator entry paths for integration.

## [0.4.1] - 2026-03-17

### Added

1. `system/` directory for baseline memory-system source code storage.
2. Bilingual LLM-assist strategy design doc:
   - `docs/architecture/llm-assist-strategy-bilingual-v0.4.1.md`

### Changed

1. `EvaluatorConfig.use_llm_assist` is now enabled by default.
2. README updated for:
   - `system/` directory convention
   - default-on LLM assist behavior

## [0.4.0] - 2026-03-17

### Added

1. Generation probe dedicated module:
   - `src/memory_eval/eval_core/generation.py`
2. Generation adapter protocol:
   - `GenerationAdapterProtocol.generate_oracle_answer(...)`
3. LLM-assisted generation judge:
   - `llm_judge_generation_answer(...)`
4. Independent generation test script:
   - `scripts/test_generation_probe.py`
5. Generation implementation and full three-probe vulnerability review docs:
   - `docs/architecture/generation-probe-implementation-v0.4.0.md`
   - `docs/architecture/three-probe-vulnerability-review-v0.4.0.md`

### Changed

1. Probe wrapper now routes generation logic through dedicated generation module.
2. Eval core exports include generation adapter protocol and generation APIs.
3. README updated with generation probe usage and test command.

## [0.3.0] - 2026-03-17

### Added

1. Retrieval probe dedicated module:
   - `src/memory_eval/eval_core/retrieval.py`
2. Retrieval adapter protocol:
   - `RetrievalAdapterProtocol.retrieve_original(...)`
3. Optional LLM-assist helper module:
   - `src/memory_eval/eval_core/llm_assist.py`
4. Independent retrieval test script:
   - `scripts/test_retrieval_probe.py`
5. Retrieval design document and encoding alignment check docs:
   - `docs/architecture/retrieval-probe-implementation-v0.3.0.md`
   - `docs/architecture/encoding-alignment-check-v0.3.0.md`

### Changed

1. Probe wrappers now call dedicated encoding/retrieval modules.
2. Encoding probe supports optional LLM-assisted fact matching.
3. README updated with retrieval and LLM-assist usage.

## [0.2.1] - 2026-03-17

### Added

1. Dedicated encoding probe implementation with explicit input contract `Q + M + F_key`
2. Adapter-bound encoding interfaces:
   - `EncodingAdapterProtocol.export_full_memory(...)`
   - `EncodingAdapterProtocol.find_memory_records(...)`
3. Encoding-only evaluation APIs:
   - `evaluate_encoding_probe(...)`
   - `evaluate_encoding_probe_with_adapter(...)`
4. Independent validation script:
   - `scripts/test_encoding_probe.py`
5. Detailed implementation doc:
   - `docs/architecture/encoding-probe-implementation-v0.2.1.md`

### Fixed

1. Safer short-token fact matching in encoding probe to avoid false positives (`he` vs `she`).

## [0.2.0] - 2026-03-17

### Added

1. Parallel three-probe evaluation core (`src/memory_eval/eval_core/`)
2. Adapter runtime protocol for evaluator integration
3. Evidence-first attribution output contract with probe-level evidence blocks
4. Runtime query resolution interfaces:
   - `build_locomo_sample_registry(...)`
   - `get_by_question_id(...)`
   - `find_by_query(...)`
5. Time-aware oracle context format update:
   - `<date_time> | <speaker>: <text>`
6. Three-layer probe implementation plan document:
   - `docs/architecture/three-layer-probe-implementation-plan.md`

### Changed

1. LOCOMO sample includes `evidence_with_time` and time-aware `oracle_context`
2. Demo script supports query-level sample resolve and LLM/rule f_key source selection

## [0.1.0] - 2026-03-16

### Added

1. New project bootstrap in `memory-eval-framework/`
2. Traceability docs for requirements and design notes
3. LOCOMO sample builder demo (`scripts/demo_build_locomo_samples.py`)
4. Dataset builder module (`src/memory_eval/dataset/locomo_builder.py`)
5. Versioning files: `VERSION`, `CHANGELOG.md`
