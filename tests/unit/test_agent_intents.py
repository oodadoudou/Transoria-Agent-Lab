from transoria.bridge.handlers.agent_intents import (
    INTENT_COMPOUND_CONFIG,
    INTENT_FALLBACK,
    INTENT_MODEL_PROFILE_COPY,
    INTENT_MODEL_PROFILE_GUIDANCE,
    INTENT_MODEL_UPGRADE,
    INTENT_PROMPT_PRESET,
    INTENT_PROMPT_QUALITY,
    INTENT_ROUTE_ORDER,
    INTENT_TASK_START,
    AgentIntentSignals,
    classify_agent_intent,
    direct_intent_sequence,
)
from transoria.bridge.handlers.agent import _classify_agent_intent


def test_task_start_intent_wins_over_configuration_signals() -> None:
    intent = classify_agent_intent(
        AgentIntentSignals(
            task_start=True,
            compound_config=True,
            prompt_preset=True,
            model_profile_copy=True,
        )
    )

    assert intent.kind == INTENT_TASK_START


def test_compound_config_intent_wins_over_prompt_and_model_signals() -> None:
    intent = classify_agent_intent(
        AgentIntentSignals(
            compound_config=True,
            prompt_quality=True,
            prompt_preset=True,
            model_profile_guidance=True,
            model_profile_copy=True,
        )
    )

    assert intent.kind == INTENT_COMPOUND_CONFIG


def test_prompt_quality_intent_wins_over_prompt_creation() -> None:
    intent = classify_agent_intent(
        AgentIntentSignals(prompt_quality=True, prompt_preset=True)
    )

    assert intent.kind == INTENT_PROMPT_QUALITY


def test_prompt_creation_intent_wins_over_model_signals() -> None:
    intent = classify_agent_intent(
        AgentIntentSignals(
            prompt_preset=True,
            model_profile_guidance=True,
            model_profile_copy=True,
        )
    )

    assert intent.kind == INTENT_PROMPT_PRESET


def test_model_guidance_intent_wins_over_model_copy() -> None:
    intent = classify_agent_intent(
        AgentIntentSignals(model_profile_guidance=True, model_profile_copy=True)
    )

    assert intent.kind == INTENT_MODEL_PROFILE_GUIDANCE


def test_model_copy_intent_is_selected_when_no_higher_signal_matches() -> None:
    intent = classify_agent_intent(AgentIntentSignals(model_profile_copy=True))

    assert intent.kind == INTENT_MODEL_PROFILE_COPY


def test_empty_signals_fall_back_to_llm() -> None:
    intent = classify_agent_intent(AgentIntentSignals())

    assert intent.kind == INTENT_FALLBACK


def test_fallback_direct_sequence_uses_ordered_intent_graph() -> None:
    assert direct_intent_sequence(
        INTENT_FALLBACK,
        allow_task_start=True,
    ) == INTENT_ROUTE_ORDER


def test_direct_sequence_can_disable_task_start_route() -> None:
    assert direct_intent_sequence(
        INTENT_FALLBACK,
        allow_task_start=False,
    ) == (
        INTENT_COMPOUND_CONFIG,
        INTENT_PROMPT_QUALITY,
        INTENT_PROMPT_PRESET,
        INTENT_MODEL_PROFILE_GUIDANCE,
        INTENT_MODEL_PROFILE_COPY,
        INTENT_MODEL_UPGRADE,
    )


def test_text_route_treats_path_and_novel_background_as_task_start() -> None:
    intent = _classify_agent_intent(
        """
        /Users/doudouda/Downloads/Personal_doc/Novels/Translate/Keyword/C-苍白黎明-copy/
        输出和输入放在同一个文件夹里。BL 作品指南

        背景/类型：现代
        作品关键词：严肃、爱恨交织、禁忌关系
        人物介绍
        攻：李承元
        """
    )

    assert intent.kind == INTENT_TASK_START


def test_text_route_keeps_explicit_model_copy_as_model_copy() -> None:
    intent = _classify_agent_intent(
        "按照 DeepSeek-f 的模型配置复制一个新的模型配置，名字叫 DeepSeek 4 Pro，model_id 用 deepseek-v4-pro。"
    )

    assert intent.kind == INTENT_MODEL_PROFILE_COPY


def test_text_route_keeps_glossary_review_prompt_creation_as_prompt_preset() -> None:
    intent = _classify_agent_intent(
        "请新建一套术语审查 Prompt，名字叫小说术语审查预设，重点检查人名一致性和 ABO 设定继承。"
    )

    assert intent.kind == INTENT_PROMPT_PRESET


def test_text_route_prefers_workflow_start_when_model_words_are_incidental() -> None:
    intent = _classify_agent_intent(
        "用我现在配置好的 DeepSeek 模型跑术语表 workflow，input 是 /Users/me/book，output 同目录，背景：现代 BL。"
    )

    assert intent.kind == INTENT_TASK_START
