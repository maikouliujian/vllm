# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.entrypoints.openai.protocol import DeltaMessage
from collections.abc import Sequence

class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        Qwen3 has stricter requirements - it needs both start and end tokens
        to be present, unlike other models that work with just the end token.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        # todo
        # Check if the model output contains both <think> and </think> tokens.
        # if self.start_token not in model_output or self.end_token not in model_output:
        #   return None, model_output

        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if self.end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        从流式输出中提取推理内容。
        支持场景：
        1. 标准 <think> ... </think>
        2. 只有 </think> 没有开始标记（隐含推理）
        3. 标记位于 Token 边界
        """

        # 1. 基础过滤：如果是纯粹的特殊 Token ID，且不携带文本，直接跳过
        if not delta_text and len(delta_token_ids) == 1:
            if delta_token_ids[0] in [self.start_token_id, self.end_token_id]:
                return None

        # 2. 状态判定（基于已确认的 Token 历史）
        has_started_before = self.start_token_id in previous_token_ids if self.start_token_id is not None else False
        has_ended_before = self.end_token_id in previous_token_ids if self.end_token_id is not None else False

        # 3. 当前 Delta 判定
        has_start_in_delta = self.start_token_id in delta_token_ids if self.start_token_id is not None else False
        has_end_in_delta = self.end_token_id in delta_token_ids if self.end_token_id is not None else False

        # --- 场景 A：已经彻底进入正文阶段 ---
        if has_ended_before:
            return DeltaMessage(content=delta_text)

        # --- 场景 B：当前 Delta 中发现了结束标记 (强制转折点) ---
        if has_end_in_delta:
            # 找到字符串中的位置
            end_idx = delta_text.find(self.end_token)

            if end_idx != -1:
                reasoning_part = delta_text[:end_idx]
                content_part = delta_text[end_idx + len(self.end_token):]
            else:
                # 特殊情况：ID 存在但文本找不到（可能 Token 包含多余字符或编码差异）
                # 此时保守处理：如果有 content 倾向则归为 content
                reasoning_part = ""
                content_part = delta_text

            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None
            )

        # --- 场景 C：已经在显式推理中 ---
        if has_started_before:
            return DeltaMessage(reasoning=delta_text)

        # --- 场景 D：当前 Delta 发现了开始标记 ---
        if has_start_in_delta:
            start_idx = delta_text.find(self.start_token)
            if start_idx != -1:
                content_pre = delta_text[:start_idx]
                reasoning_post = delta_text[start_idx + len(self.start_token):]
            else:
                content_pre = ""
                reasoning_post = delta_text

            return DeltaMessage(
                content=content_pre if content_pre else None,
                reasoning=reasoning_post if reasoning_post else None
            )

        # --- 场景 E：无标地带 (关键逻辑) ---
        # 如果你确定这是推理模型（如 R1），且还没遇到结束标，
        # 即使没看到 <think>，我们也认为当前是推理。
        # 通用兜底：既无标记也非强制推理模型，视为正文
        return DeltaMessage(content=delta_text)
