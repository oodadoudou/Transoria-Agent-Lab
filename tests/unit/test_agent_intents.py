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
from transoria.bridge.handlers.agent_response_routing import classify_message_intent


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
    intent = classify_message_intent(
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
    intent = classify_message_intent(
        "按照 DeepSeek-f 的模型配置复制一个新的模型配置，名字叫 DeepSeek 4 Pro，model_id 用 deepseek-v4-pro。"
    )

    assert intent.kind == INTENT_MODEL_PROFILE_COPY


def test_text_route_keeps_glossary_review_prompt_creation_as_prompt_preset() -> None:
    intent = classify_message_intent(
        "请新建一套术语审查 Prompt，名字叫小说术语审查预设，重点检查人名一致性和 ABO 设定继承。"
    )

    assert intent.kind == INTENT_PROMPT_PRESET


def test_text_route_prefers_workflow_start_when_model_words_are_incidental() -> None:
    intent = classify_message_intent(
        "用我现在配置好的 DeepSeek 模型跑术语表 workflow，input 是 /Users/me/book，output 同目录，背景：现代 BL。"
    )

    assert intent.kind == INTENT_TASK_START


def test_text_route_prefers_workflow_start_when_model_context_is_noisy() -> None:
    intent = classify_message_intent(
        "先别管刚才那个 DeepSeek 4 Pro 模型创建草案了，直接处理术语：/Users/me/book，输出和输入一样，背景是现代 BL。"
    )

    assert intent.kind == INTENT_TASK_START


def test_text_route_keeps_glossary_workflow_above_compound_config_words() -> None:
    intent = classify_message_intent(
        "提取术语，input 是 /Users/me/book，output 同目录，并发沿用当前模型配置，背景是现代 BL。"
    )

    assert intent.kind == INTENT_TASK_START


def test_text_route_detects_compound_config_batch_request() -> None:
    intent = classify_message_intent(
        "请准备配置修改草案：把模型 agy-cli-f 的并发数改成 4，把翻译 Prompt「默认」重命名为「标准中文翻译预设」，并把当前阶段配置保存成「测试复合配置」预设。"
    )

    assert intent.kind == INTENT_COMPOUND_CONFIG


def test_text_route_keeps_translation_quality_advice_above_prompt_creation() -> None:
    intent = classify_message_intent(
        "现在翻译效果不好，文风很怪而且有人名不一致，我想修改 prompt，你帮我诊断一下应该怎么改。"
    )

    assert intent.kind == INTENT_PROMPT_QUALITY


def test_text_route_treats_prompt_design_for_bad_translation_as_quality_request() -> None:
    intent = classify_message_intent(
        "翻译效果不好，我想让你帮我设计一个新的翻译 prompt，文风要更自然，人名一致，不要现在就启动任务。"
    )

    assert intent.kind == INTENT_PROMPT_QUALITY


def test_text_route_handles_vague_model_setup_help() -> None:
    intent = classify_message_intent(
        "我想加一个新模型，但是不知道 base url 和 model id 怎么填，你帮我配置一下。"
    )

    assert intent.kind == INTENT_MODEL_PROFILE_GUIDANCE


def test_text_route_handles_existing_inventory_model_upgrade() -> None:
    intent = classify_message_intent(
        "我想把工作模型换成模型库里更强的模型，你帮我看一下并生成修改方案。"
    )

    assert intent.kind == INTENT_MODEL_UPGRADE
