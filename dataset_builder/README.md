# DynaMem-Bench: Dynamic Memory Benchmark Builder

> Construction code for the on-policy dynamic memory benchmark **DynaMem-Bench** used by the **EvalMem** evaluation framework.
> The released benchmark is produced by this builder.
> Detailed design rationale is documented in the paper.

This sub-project contains only the **construction pipeline** (offline pre-generation + on-policy runtime scaffolding). The constructed dataset itself is **not** committed to this anonymous repository; rerunning the pipeline reproduces it from configuration.

---

## 1. Dataset specification

- **10 personas**, each with 6 structured attributes covering: age band, education, occupation, communication style, change propensity, NEG sensitivity.
- **Per persona**: 10 sessions × 20 turns ≈ 200 dialogue turns (produced at runtime).
- **Per persona**: ~80 evaluation questions, ~100 query asks distributed across the 10 sessions.
- **Total scale**: 10 personas × ~100 query asks ≈ **1000 question asks** (≈ 800 unique questions).

Each evaluation sample exposes a seven-tuple compatible with EvalMem (see "Sample schema" below) plus dynamic-state fields (`ask_at_sessions`, `gold_answers`, `f_keys`).

---

## 2. Layout

```
dataset_builder/
├── configs/
│   ├── dataset_config.yaml         # Global parameters
│   ├── persona_dimensions.yaml     # 10 persona presets
│   ├── state_var_library.yaml      # Candidate state-variable library (~25 vars × 6 domains)
│   ├── prompts/                    # LLM prompt templates
│   └── keys.local.example.json     # Copy to keys.local.json and fill in
├── builder/                        # Offline pre-generation (stages 0-7)
│   ├── schemas.py                  # Dataclasses
│   ├── config.py                   # Config loaders
│   ├── llm_client.py               # OpenAI-compatible client w/ cache + retry
│   ├── persona_pool.py             # Stage 0: render personas
│   ├── question_sampler.py         # Stage 1: sample question drafts
│   ├── state_schema.py             # Stage 2: state schema + dual-core + cascade
│   ├── state_evolution.py          # Stage 3: deterministic state evolution
│   ├── exposure_plan.py            # Stage 4: per-session exposure plan
│   ├── seed_utterance.py           # Stage 5: per-session seed utterances
│   ├── eval_questions.py           # Stage 6: evaluation questions (POS/NEG-A/B/C/D)
│   ├── oracle_context.py           # Stage 7: oracle dialogues + contexts
│   └── verifier.py                 # Cross-model verification
├── runtime/                        # Stage 10: on-policy interaction
│   ├── user_simulator.py
│   └── session_runner.py
├── scripts/
│   ├── 01_build_per_persona.py     # Per-persona / batch builder
│   ├── 02_inspect_dataset.py       # Dataset audit script
│   ├── 03_run_verification.py      # Cross-model verification runner
│   ├── 04_runtime_smoke_test.py    # Stage-10 smoke test
│   └── 05_health_check.py          # End-to-end health check
└── tests/                          # Unit tests
```

---

## 3. Quick start

### 3.1 Configure API credentials

Copy `configs/keys.local.example.json` to `configs/keys.local.json` and fill in:

```json
{
  "api_key": "<YOUR_OPENAI_COMPATIBLE_API_KEY>",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "temperature": 0.0
}
```

`keys.local.json` is **gitignored** and never committed.

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

(Python >= 3.9.)

### 3.3 Build the dataset

```bash
# First time: build the persona pool + a single-persona dry run
python scripts/01_build_per_persona.py --build-pool --persona-id p_01

# Single persona (cache-warm reruns are fast)
python scripts/01_build_per_persona.py --persona-id p_01

# Batch build all 10 personas
python scripts/01_build_per_persona.py --all

# Inspect the dataset
python scripts/02_inspect_dataset.py
```

Outputs land in `dataset_builder/data/per_persona/<persona_id>/`. LLM responses are cached under `dataset_builder/cache/llm_responses/` so repeated runs do not consume quota.

---

## 4. Offline construction pipeline (7 stages)

Each persona runs the following pipeline independently and produces 8 JSON artifacts:

| Stage | Input | Output | LLM calls |
|-------|-------|--------|-----------|
| 0 Persona pool | `persona_dimensions.yaml` | `personas.json`, `per_persona/<id>/persona.json` | 1 / persona |
| 1 Question drafts | `persona.json`, `state_var_library.yaml` | `questions_draft.json` | 1 (sample ~180 candidates) |
| 2 State schema | `questions_draft.json`, `state_var_library.yaml` | `state_schema.json` | 1 (cascade detection) |
| 3 State evolution | `state_schema.json` | `state_evolution.json` | 0 (deterministic) |
| 4 Exposure plan | `state_evolution.json` | `exposure_plan.json` | 10 (one per session) |
| 5 Seed utterances | `exposure_plan.json` | `seed_utterances.json` | 10 |
| 6 Evaluation questions | `state_evolution.json`, `exposure_plan.json` | `eval_questions.json` | 3-4 (NEG-A/D + POS top-up) |
| 7 Oracle contexts | `eval_questions.json`, `exposure_plan.json` | `oracle_dialogues.json`, `oracle_contexts.json` | 10 |

Approx. **35 LLM calls / persona**, **~350 / full benchmark** (with caching, repeated runs are nearly free).

---

## 5. Design highlights

1. **6-dimensional structured persona pool**
   - Two evaluation-specific dimensions: `change_propensity` (drives ρ) and `neg_sensitivity` (drives the NEG subtype distribution).

2. **Dual-core variables + three cascade modes**
   - One primary core variable + 1-2 secondary cores per persona.
   - Cascade modes: `immediate`, `delayed`, and **`rollback`** (30% of personas roll back at session 9).

3. **NEG task with four sub-types**
   - `NEG-A` non-existent / `NEG-B` stale / `NEG-C` conflicting / `NEG-D` adversarial.
   - Distributed across sessions (not concentrated at session 10).

4. **Temporal interference set I_t** with three labels:
   - `stale`, `cascaded`, `rollback_reactivated`.

5. **State-tracking multi-session asks**
   - The same question is asked at the end of two different sessions; the gold answer changes with the dynamic state σ_t, directly stressing the system's state-update behaviour.

---

## 6. Algorithms

### 6.1 State evolution (`builder/state_evolution.py`)

Fully deterministic, driven by `random.Random(seed + persona_id)`:

- session 0: initialise σ_0 (avoid extremes).
- sessions 1-5: hold σ_0.
- session 6: primary core flips (pick a different value).
- sessions 7-10: cascade triggers + 30% personas roll back + ρ-driven point updates.

### 6.2 `ask_at` schedule (`builder/eval_questions.py:compute_ask_at_sessions_for_pos`)

For every POS state-tracking question:

1. Earliest askable session = max( exposure session of each `required_var` ).
2. Add every session in which any `required_var` changes.
3. Cap to `max_asks = 2` (keeps total asks ≈ 100).
4. At each `ask_at` session, recompute `gold = compute_gold(σ_t, required_vars)` independently.

### 6.3 NEG distribution (`builder/eval_questions.py:_distribute_neg_to_sessions`)

- NEG-A / NEG-D: uniformly across sessions 1..10.
- NEG-B / NEG-C: across sessions 6..10 (require state changes upstream).

---

## 7. Fairness guarantees (runtime / stage 10)

1. **Fixed seed utterances**: every system sees the exact same first-turn user message in each session.
2. **Mandatory `must_expose` enumeration**: the user simulator must surface every required fact.
3. **System-agnostic** persona / state trajectory / questions / answers.
4. **Read-only probes**: examiners do not pollute the system-under-test's memory.

---

## 8. Status

### Implemented (offline pre-generation)

- Stages 0-7 (persona rendering / question sampling / state schema / state evolution / exposure plan / seed utterance / eval questions / oracle context).
- LLM client (cache + retry + JSON parsing).
- Dataset inspection script.

### Stretch / TODO

- Stage 8: cross-model verification with five auxiliary checks.
- Stage 9: human κ spot-check UI.
- Stage 10: on-policy interaction (`user_simulator` + `session_runner` + read-only adapters); scaffold present in `runtime/`.
- Unit tests.

---

## 9. Sample schema

Every evaluation sample is a seven-tuple (compatible with EvalMem § Framework) plus dynamic fields:

```json
{
  "question_id": "p_01_q_007",
  "persona_id": "p_01",
  "question": "What am I mainly preparing for now?",
  "task_type": "POS",
  "neg_subtype": null,
  "is_state_tracking": true,
  "domain": "study",
  "required_vars": ["current_focus"],
  "ask_at_sessions": [3, 6, 10],
  "gold_answers": {
    "3": "graduate-school exam",
    "6": "internship",
    "10": "internship"
  },
  "f_keys": {
    "3": [["current_focus", "graduate-school exam"]],
    "6": [["current_focus", "internship"]]
  },
  "trap_answers": {},
  "cascade_metadata": null
}
```
