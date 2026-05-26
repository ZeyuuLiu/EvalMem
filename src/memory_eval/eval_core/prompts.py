from __future__ import annotations

import json
from typing import Any, Dict, List


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_only_notice() -> str:
    return (
        "输出要求：\n"
        "1. 只输出一个纯文本 JSON 对象。\n"
        "2. 不要输出 ```json 或任何 Markdown 包裹。\n"
        "3. 不要在 JSON 前后输出解释性文字。\n"
        "4. defect 代码只能从题目允许的枚举中选择，不能自造。\n"
    )


def build_encoding_pos_prompt(query: str, f_key: List[str], evidence_texts: List[str], candidates: List[Dict[str, Any]]) -> str:
    return (
        "你是 EncodingAgent，负责判断 POS 任务中记忆系统是否已经把回答当前问题所需的信息写入到可观察记忆中。\n"
        "变量释义：\n"
        "- Query：当前问题。\n"
        "- F_key：回答该问题必须被支持的关键事实单元，是最小事实约束。\n"
        "- GoldEvidence：来自原始对话的标准证据，用于参照，不要求 Candidates 必须逐字一致。\n"
        "- Candidates：记忆系统当前可导出的候选记忆集合。你只能基于这些候选判断，不能脑补隐藏记忆。\n"
        "状态定义：\n"
        "- EXIST：候选集合中存在足以支撑核心事实的信息。\n"
        "- MISS：候选集合中不存在足以支撑核心事实的信息。对于 POS 任务，这意味着编码失败。\n"
        "- CORRUPT_AMBIG：候选涉及相关内容，但主体、指代、时间锚点或事件归属过于模糊，无法确认就是目标事实。\n"
        "- CORRUPT_WRONG：候选涉及相关内容，但关键事实值错误，例如人、时间、地点、关系、事件结果错了。\n"
        "缺陷代码严格映射：\n"
        "- MISS -> defects=[\"EM\"]，EM = Extraction Miss\n"
        "- CORRUPT_AMBIG -> defects=[\"EA\"]，EA = Extraction Ambiguous\n"
        "- CORRUPT_WRONG -> defects=[\"EW\"]，EW = Extraction Wrong\n"
        "- EXIST -> defects=[]\n"
        "判定原则：\n"
        "1. 以语义支撑为主，不要求精确字符串匹配。\n"
        "2. 允许摘要、改写、时间归一化、代词替换、多条候选联合支撑。\n"
        "3. 候选即使是 summary，只要保留了回答问题所需核心事实，也可判为 EXIST。\n"
        "4. 请同时输出 reasoning、matched_candidate_ids、evidence_snippets、missing_facts，方便后续归因。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"encoding_state\":\"EXIST|MISS|CORRUPT_AMBIG|CORRUPT_WRONG\","
        "\"defects\":[\"EM|EA|EW\"],"
        "\"confidence\":0.0,"
        "\"matched_candidate_ids\":[],"
        "\"reasoning\":\"...\","
        "\"evidence_snippets\":[],"
        "\"missing_facts\":[]}\n"
        f"Query: {query}\n"
        f"F_key: {_dump(f_key)}\n"
        f"GoldEvidence: {_dump(evidence_texts)}\n"
        "Candidates:\n"
        + "\n".join([f"- id={c.get('id','')} text={c.get('text','')}" for c in candidates[:60]])
    )


def build_encoding_neg_prompt(query: str, evidence_texts: List[str], candidates: List[Dict[str, Any]]) -> str:
    return (
        "你是 EncodingAgent，负责判断 NEG 任务中记忆系统是否错误地写入了会诱导错误回答的伪记忆。\n"
        "请特别注意：NEG 任务中的 MISS 是正确状态，不表示失败；它表示记忆库中没有可回答该问题的伪证据。\n"
        "变量释义：\n"
        "- Query：当前问题。\n"
        "- ReferenceEvidenceForContrast：用于说明该问题本不应被回答的对照证据。\n"
        "- Candidates：记忆系统当前可导出的候选记忆集合。\n"
        "状态定义：\n"
        "- MISS：正确状态，表示未发现足以支撑错误回答的伪记忆。\n"
        "- DIRTY：错误状态，表示发现了会诱导具体回答的伪记忆。\n"
        "缺陷代码严格映射：\n"
        "- DIRTY -> defects=[\"DMP\"]，DMP = Dirty Memory Pollution\n"
        "- MISS -> defects=[]\n"
        "判定原则：\n"
        "1. 只有当候选真正提供了可支撑具体回答的伪事实、伪时间、伪事件关联时，才判 DIRTY。\n"
        "2. 只是主题相近、词汇相似、但不足以回答 Query 的候选，仍应判 MISS。\n"
        "3. 请输出 reasoning 和 evidence_snippets，说明为什么这是正确的 NEG-MISS 或错误的 DIRTY。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"encoding_state\":\"DIRTY|MISS\","
        "\"defects\":[\"DMP\"],"
        "\"confidence\":0.0,"
        "\"matched_candidate_ids\":[],"
        "\"reasoning\":\"...\","
        "\"evidence_snippets\":[]}\n"
        f"Query: {query}\n"
        f"ReferenceEvidenceForContrast: {_dump(evidence_texts)}\n"
        "Candidates:\n"
        + "\n".join([f"- id={c.get('id','')} text={c.get('text','')}" for c in candidates[:60]])
    )


def build_retrieval_pos_prompt(
    query: str,
    f_key: List[str],
    evidence_texts: List[str],
    retrieved_items: List[Dict[str, Any]],
    rank_index: int,
    hit_indices: List[int],
    snr: float,
    tau_rank: int,
    tau_snr: float,
) -> str:
    return (
        "你是 RetrievalAgent，负责判断 POS 任务中原记忆系统的原生检索结果是否成功召回了回答问题所需的证据。\n"
        "变量释义：\n"
        "- Query：当前问题。\n"
        "- F_key：问题所依赖的关键事实单元。\n"
        "- GoldEvidence：标准证据，可用于语义参照。\n"
        "- RetrievedItems：原记忆系统真实返回的 top-k 检索结果，你只能依据这些内容判断。\n"
        "- Diagnostics：辅助诊断信息，不是硬规则。\n"
        "  * rank_index：规则匹配估计的最佳命中位置，-1 表示没找到。\n"
        "  * hit_indices：规则匹配认为可能命中的位置。\n"
        "  * snr：估计的信噪比。\n"
        "  * tau_rank / tau_snr：排序和信噪比阈值，可作为 LATE / NOI 参考。\n"
        "状态定义：\n"
        "- HIT：RetrievedItems 中存在足以支撑核心事实的证据。\n"
        "- MISS：RetrievedItems 中不存在足以支撑核心事实的证据。\n"
        "缺陷代码严格映射：\n"
        "- RF = Retrieval Failure。仅当 retrieval_state=MISS 时可输出；后处理会在 encoding=MISS 时自动 suppress。\n"
        "- LATE = 命中了，但最佳支持证据排序明显过后。\n"
        "- NOI = 命中了，但检索集合噪声大、有效证据被淹没。\n"
        "判定原则：\n"
        "1. 以语义支撑为主，不要求词面完全一致。\n"
        "2. 允许摘要、压缩记忆、改写、时间归一化表达作为有效命中。\n"
        "3. 如果多个 retrieved items 联合起来能支撑事实，也可判为 HIT。\n"
        "4. 如果确实未检到能支撑核心事实的证据，应判 MISS，并在 defects 中输出 RF。\n"
        "5. 如果检到了，但最佳证据位置过后，可在 HIT 基础上附加 LATE。\n"
        "6. 如果检到了，但噪声大、信号弱，可在 HIT 基础上附加 NOI。\n"
        "7. 请给出 best_rank、reasoning 和 evidence_snippets，方便后续归因。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"retrieval_state\":\"HIT|MISS\","
        "\"defects\":[\"RF|LATE|NOI\"],"
        "\"matched_ids\":[],"
        "\"best_rank\":-1,"
        "\"confidence\":0.0,"
        "\"reasoning\":\"...\","
        "\"evidence_snippets\":[]}\n"
        f"Query: {query}\n"
        f"F_key: {_dump(f_key)}\n"
        f"GoldEvidence: {_dump(evidence_texts)}\n"
        f"Diagnostics: rank_index={rank_index}, hit_indices={_dump(hit_indices)}, snr={snr:.6f}, tau_rank={tau_rank}, tau_snr={tau_snr}\n"
        "RetrievedItems:\n"
        + "\n".join([f"- id={it.get('id','')} score={it.get('score',0)} text={it.get('text','')}" for it in retrieved_items[:30]])
    )


def build_retrieval_neg_prompt(query: str, retrieved_items: List[Dict[str, Any]]) -> str:
    return (
        "你是 RetrievalAgent，负责判断 NEG 任务中原记忆系统的原生检索结果是否产生了高误导性的伪相关召回。\n"
        "请特别注意：NEG 任务中的 MISS 是正确状态，不表示失败；它表示检索结果不足以诱导具体回答。\n"
        "状态定义：\n"
        "- MISS：正确状态，没有足以误导作答的召回。\n"
        "- NOISE：错误状态，检索结果会显著诱导一个具体但不被证据支持的回答。\n"
        "缺陷代码严格映射：\n"
        "- NOISE -> defects=[\"NIR\"]，NIR = Noise-Induced Retrieval\n"
        "- MISS -> defects=[]\n"
        "判定原则：\n"
        "1. 主题相近不等于 NOISE；只有在会误导具体回答时才判 NOISE。\n"
        "2. 如果只是弱相关、泛相关、背景相关，仍应判 MISS。\n"
        "3. 请输出 reasoning 和 evidence_snippets，说明为什么这是正确的 NEG-MISS 或错误的 NOISE。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"retrieval_state\":\"MISS|NOISE\","
        "\"defects\":[\"NIR\"],"
        "\"confidence\":0.0,"
        "\"reasoning\":\"...\","
        "\"evidence_snippets\":[]}\n"
        f"Query: {query}\n"
        "RetrievedItems:\n"
        + "\n".join([f"- id={it.get('id','')} score={it.get('score',0)} text={it.get('text','')}" for it in retrieved_items[:30]])
    )


def build_generation_pos_answer_prompt(query: str, oracle_context: str, answer_oracle: str, answer_gold: str) -> str:
    return (
        "你是 GenerationAgent，负责 POS 任务下的 oracle correctness 判定。\n"
        "变量释义：\n"
        "- OracleContext：完美证据上下文。\n"
        "- OracleAnswer：系统在完美证据下生成的回答。\n"
        "- GoldAnswer：标准答案。\n"
        "缺陷代码严格映射：\n"
        "- GF = Generation Faithfulness failure：忽视或违背 OracleContext。\n"
        "- GRF = Generation Reasoning Failure：参考了 OracleContext，但推理整合失败。\n"
        "判定原则：\n"
        "1. 若 OracleAnswer 与 GoldAnswer 在核心事实层面语义等价，应判 correct=true 且 substate=NONE。\n"
        "2. 时间表达允许不同格式，只要指向同一时间即可。\n"
        "3. 如果回答明显忽视 OracleContext，优先考虑 GF。\n"
        "4. 如果回答参考了上下文，但关系整合、比较、归纳失败，优先考虑 GRF。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"correct\":true,"
        "\"substate\":\"GF|GRF|NONE\","
        "\"grounded\":true,"
        "\"reason\":\"...\"}\n"
        f"Query: {query}\n"
        f"OracleContext: {oracle_context}\n"
        f"OracleAnswer: {answer_oracle}\n"
        f"GoldAnswer: {answer_gold}\n"
    )


def build_generation_neg_answer_prompt(query: str, oracle_context: str, answer_oracle: str, answer_gold: str) -> str:
    return (
        "你是 GenerationAgent，负责 NEG 任务下的 oracle correctness 判定。\n"
        "请注意：NEG 任务的正确行为是拒答，而不是回答细节。\n"
        "缺陷代码严格映射：\n"
        "- GH = Generation Hallucination：在不应回答时编造了具体内容。\n"
        "判定原则：\n"
        "1. 如果 OracleAnswer 是谨慎拒答或明确说明上下文不足，应判 correct=true 且 substate=NONE。\n"
        "2. 如果 OracleAnswer 给出了具体事实、时间、地点、关系或事件结论，应判 correct=false 且 substate=GH。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"correct\":true,"
        "\"substate\":\"GH|NONE\","
        "\"is_refusal\":true,"
        "\"reason\":\"...\"}\n"
        f"Query: {query}\n"
        f"OracleContext: {oracle_context}\n"
        f"OracleAnswer: {answer_oracle}\n"
        f"ExpectedRefusalStyle: {answer_gold}\n"
    )


def build_generation_pos_comparison_prompt(
    query: str,
    answer_gold: str,
    answer_online: str,
    answer_oracle: str,
    oracle_context: str,
) -> str:
    return (
        "你是 GenerationAgent，负责 POS 任务下 online/oracle/gold 三答案对比。\n"
        "目标：区分在线回答错误、oracle 回答错误，以及错误更偏向上游链路还是生成本身。\n"
        "缺陷代码严格映射：\n"
        "- GF = Generation Faithfulness failure\n"
        "- GRF = Generation Reasoning Failure\n"
        "判定原则：\n"
        "1. 先分别判断 OnlineAnswer 与 GoldAnswer、OracleAnswer 与 GoldAnswer 的语义等价性。\n"
        "2. 如果 OracleAnswer 正确但 OnlineAnswer 错误，说明更偏向上游链路问题，不应轻易把责任判给生成层。\n"
        "3. 如果 OracleAnswer 本身错误，再根据是否忽视上下文区分 GF 或 GRF。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"generation_state\":\"PASS|FAIL\","
        "\"defects\":[\"GF|GRF\"],"
        "\"online_correct\":true,"
        "\"oracle_correct\":true,"
        "\"comparative_judgement\":{\"online_vs_gold\":\"...\",\"oracle_vs_gold\":\"...\",\"online_vs_oracle\":\"...\"},"
        "\"reasoning\":\"...\"}\n"
        f"Query: {query}\n"
        f"GoldAnswer: {answer_gold}\n"
        f"OnlineAnswer: {answer_online}\n"
        f"OracleAnswer: {answer_oracle}\n"
        f"OracleContext: {oracle_context}\n"
    )


def build_generation_neg_comparison_prompt(
    query: str,
    answer_gold: str,
    answer_online: str,
    answer_oracle: str,
    oracle_context: str,
) -> str:
    return (
        "你是 GenerationAgent，负责 NEG 任务下 online/oracle/gold 三答案对比。\n"
        "请注意：NEG 任务的理想行为是拒答。\n"
        "缺陷代码严格映射：\n"
        "- GH = Generation Hallucination\n"
        "判定原则：\n"
        "1. 只要任一答案给出具体编造内容，就应视为 FAIL 并给出 GH。\n"
        "2. 如果两个答案都体现谨慎拒答，则应视为 PASS。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"generation_state\":\"PASS|FAIL\","
        "\"defects\":[\"GH\"],"
        "\"online_correct\":true,"
        "\"oracle_correct\":true,"
        "\"comparative_judgement\":{\"online_vs_gold\":\"...\",\"oracle_vs_gold\":\"...\",\"online_vs_oracle\":\"...\"},"
        "\"reasoning\":\"...\"}\n"
        f"Query: {query}\n"
        f"ExpectedRefusal: {answer_gold}\n"
        f"OnlineAnswer: {answer_online}\n"
        f"OracleAnswer: {answer_oracle}\n"
        f"OracleContext: {oracle_context}\n"
    )


def build_attribution_prompt(
    task_type: str,
    query: str,
    answer_gold: str,
    enc_summary: Dict[str, Any],
    ret_summary: Dict[str, Any],
    gen_summary: Dict[str, Any],
) -> str:
    task_desc = "NEG 任务：当前 query 无原文证据支撑，理想行为是拒答。" if task_type == "NEG" else "POS 任务：当前 query 有原文证据支撑，理想行为是检索并正确回答。"
    return (
        "你是 AttributionAgent，负责综合编码、检索、生成三个探针结果，形成最终的缺陷集合与归因逻辑。\n"
        "请注意：最终答案不是单一主责点，而是缺陷集合（union of defects）。一个样本可以同时存在多个问题。\n"
        "判定原则：\n"
        "1. 编码层回答‘是否被存住’，检索层回答‘是否被取出’，生成层回答‘给定完美证据后还能否答对’。\n"
        "2. 必须依据三层探针证据，不要只看最终答对/答错。\n"
        "3. 如果 encoding_state=MISS，则 RF 应被 suppress，因为没存住时不应归因检索失败。\n"
        "4. POS 任务中，encoding MISS 是失败；NEG 任务中，encoding MISS 是正确观察，不应自动记为缺陷。\n"
        "5. POS retrieval_state=MISS 时，可把 RF 视为候选缺陷，但要结合编码层决定是否保留。\n"
        "6. generation 层的 GH/GF/GRF 可以与上游缺陷同时存在，不能强行压缩成单一主责。\n"
        "7. primary_cause 用于指出最主要责任层，但 defect_union 必须保留所有被证据支持的缺陷。\n"
        "8. secondary_causes 只能来自已有 probe 证据，不要虚构。\n"
        + _json_only_notice()
        + "请只输出符合下列 schema 的 JSON："
        "{\"primary_cause\":\"encoding|retrieval|generation|none\","
        "\"secondary_causes\":[],"
        "\"defect_union\":[],"
        "\"decision_trace\":[],"
        "\"summary\":\"...\"}\n"
        f"TaskDefinition: {task_desc}\n"
        f"Query: {query}\n"
        f"GoldAnswer: {answer_gold}\n"
        f"EncodingSummary: {_dump(enc_summary)}\n"
        f"RetrievalSummary: {_dump(ret_summary)}\n"
        f"GenerationSummary: {_dump(gen_summary)}\n"
    )


def build_correctness_judge_prompt(
    task_type: str,
    question: str,
    answer_gold: str,
    answer_pred: str,
    judge_mode: str = "online",
    oracle_context: str = "",
    retrieved_context: str = "",
) -> str:
    if task_type == "NEG":
        task_desc = (
            "This is a NEG question. The expected behavior is usually to abstain, answer cautiously, "
            "or explicitly state that the available memory/context does not support a concrete answer. "
            "If the answer fabricates unsupported facts, dates, names, places, or events, it should usually be judged WRONG."
        )
    else:
        task_desc = (
            "This is a POS question. Judge whether the generated answer matches the gold answer on the core fact. "
            "Be generous: if the generated answer is longer, more conversational, slightly hedged, or phrased differently, "
            "count it as CORRECT as long as it refers to the same underlying fact."
        )
    mode_desc = (
        "This is ONLINE judging: use only the question, gold answer, generated answer, and the actual retrieved context. "
        "Do not use oracle context to infer missing facts."
        if judge_mode == "online"
        else "This is ORACLE judging: you may use the oracle context as supporting evidence."
    )
    return (
        "Your task is to label a generated answer as CORRECT or WRONG.\n"
        "You will be given a question, a gold answer, a generated answer, and optional supporting context.\n"
        "Judge factual equivalence rather than exact wording.\n"
        "Be generous when the generated answer is longer than the gold answer, uses different wording, "
        "or is slightly cautious, as long as it still conveys the same core fact.\n"
        "For time-related questions, count the answer as CORRECT if it refers to the same date, month, year, "
        "or relative time anchored to the same event, even if the surface format is different.\n"
        "If the generated answer omits the key fact, answers a different question, contradicts the gold answer, "
        "or fabricates unsupported content, judge it as WRONG.\n"
        "For NEG questions, cautious abstention can be CORRECT; unsupported fabrication should be WRONG.\n"
        "Return only strict JSON: "
        "{\"label\":\"CORRECT|WRONG\",\"reason\":\"...\",\"semantic_match\":true|false,"
        "\"temporal_match\":true|false,\"refusal_expected\":true|false,"
        "\"refusal_present\":true|false,\"fabricated\":true|false}\n"
        f"JudgeMode: {judge_mode}\n"
        f"ModeInstruction: {mode_desc}\n"
        f"TaskDefinition: {task_desc}\n"
        f"Question: {question}\n"
        f"CorrectAnswer: {answer_gold}\n"
        f"ModelResponse: {answer_pred}\n"
        f"OracleContext: {oracle_context}\n"
        f"RetrievedContext: {retrieved_context}\n"
    )
