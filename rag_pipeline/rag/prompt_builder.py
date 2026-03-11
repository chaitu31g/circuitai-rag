"""Prompt construction for datasheet-grounded QA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are an electronics engineer answering from datasheet context. "
    "If the context includes figure descriptions or diagram summaries that are "
    "not relevant to the question, ignore them completely. "
    "If the answer is not present in the provided context, respond with: "
    "'The datasheet context does not specify this value.' "
    "Do not provide estimated or general electronics knowledge."
)

SPEC_EXTRACTION_SYSTEM_PROMPT = (
    "You are an electronics engineer extracting specifications from datasheet context.\n\n"
    "Rules:\n\n"
    "1. Extract numeric specifications directly from the context.\n"
    "2. If multiple values exist, summarize them clearly.\n"
    "3. If a constraint exists (min/max), state it explicitly.\n"
    "4. Preserve units and conditions.\n"
    "5. Do NOT infer or estimate missing values.\n"
    "6. If the context contains figure descriptions or diagram summaries that are not relevant "
    "to the question, ignore them — extract only from textual specification content.\n"
    "7. If no value exists, say: 'The datasheet context does not specify this value.'"
)

JSON_SPEC_SYSTEM_PROMPT = (
    "You are an electronics engineer extracting specifications from datasheet context.\n\n"
    "Task:\n"
    "Extract ALL constraints related to the queried parameter and output ONLY valid JSON.\n\n"
    "Rules:\n\n"
    "1. Extract each constraint separately.\n"
    "2. Preserve parameter name (ICBO, VCBO, VBRCBO, etc.).\n"
    "3. Include value, unit, and conditions.\n"
    "4. Do NOT infer missing values.\n"
    "5. If the context includes figure descriptions or diagram summaries that are not relevant, ignore them.\n"
    "6. If no constraint exists, return an empty JSON array.\n"
    "7. Output ONLY JSON (no explanations)."
)

RAG_ANSWER_SYSTEM_PROMPT = (
    "You are an electronics engineer answering using datasheet context.\n\n"
    "Instructions:\n\n"
    "1. Use ONLY the provided context.\n"
    "2. If numeric constraints exist, state them directly.\n"
    "3. If multiple constraints exist, summarize them clearly.\n"
    "4. Preserve units and conditions.\n"
    "5. Do NOT say the value is unspecified if constraints are present.\n"
    "6. Do NOT invent values.\n"
    "7. If the context contains figure descriptions or diagram summaries that are "
    "not relevant to the question, ignore them — base your answer on the textual content only.\n\n"
    "Answer style:\n"
    "Concise technical datasheet explanation using retrieved values."
)


@dataclass
class PromptParts:
    system: str
    user: str
    full_prompt: str


class DatasheetPromptBuilder:
    """Builds prompts that constrain answers to retrieved datasheet context."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    @staticmethod
    def _to_full_prompt(system: str, user: str) -> str:
        # Keep role tags explicit for small local models running via Ollama.
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"

    def build(
        self,
        query: str,
        context: str,
        context_chunk_ids: Optional[List[str]] = None,
    ) -> PromptParts:
        refs = ", ".join(context_chunk_ids or [])
        refs_line = f"\nContext chunk IDs: {refs}" if refs else ""

        user_prompt = (
            "Use only the provided datasheet context.\n"
            "Rules:\n"
            "- Do not invent specifications, values, or conditions.\n"
            "- If figure descriptions or diagram summaries in the context are not "
            "relevant to the question, ignore them.\n"
            "- If missing from context, reply exactly: "
            "'The datasheet context does not specify this value.'\n"
            "- Cite the relevant context details in your reasoning.\n"
            "- Keep the final answer concise and technical.\n\n"
            f"Question:\n{query}\n\n"
            f"Datasheet Context:\n{context}{refs_line}"
        )

        full_prompt = self._to_full_prompt(
            system=self.system_prompt,
            user=user_prompt,
        )
        return PromptParts(system=self.system_prompt, user=user_prompt, full_prompt=full_prompt)

    def build_spec_extraction_prompt(
        self,
        query: str,
        context: str,
        context_chunk_ids: Optional[List[str]] = None,
    ) -> PromptParts:
        refs = ", ".join(context_chunk_ids or [])
        refs_line = f"\nContext chunk IDs: {refs}" if refs else ""
        user_prompt = (
            "Extract only specifications relevant to the question from datasheet context.\n"
            "Return output in this exact structure for each constraint found:\n"
            "Parameter:\n"
            "Value:\n"
            "Conditions:\n"
            "Source snippet:\n\n"
            "If multiple constraints exist, list each as a separate block.\n"
            "If no value is present in context, output exactly:\n"
            "The datasheet context does not specify this value.\n\n"
            f"Question:\n{query}\n\n"
            f"Datasheet Context:\n{context}{refs_line}"
        )
        full_prompt = self._to_full_prompt(
            system=SPEC_EXTRACTION_SYSTEM_PROMPT,
            user=user_prompt,
        )
        return PromptParts(
            system=SPEC_EXTRACTION_SYSTEM_PROMPT,
            user=user_prompt,
            full_prompt=full_prompt,
        )

    def build_json_spec_prompt(
        self,
        query: str,
        context: str,
        context_chunk_ids: Optional[List[str]] = None,
    ) -> PromptParts:
        refs = ", ".join(context_chunk_ids or [])
        refs_line = f"\nContext chunk IDs: {refs}" if refs else ""
        user_prompt = (
            "Return ONLY valid JSON in this exact schema:\n"
            "[\n"
            "  {\n"
            '    "parameter": "",\n'
            '    "value": "",\n'
            '    "unit": "",\n'
            '    "limit_type": "",\n'
            '    "conditions": "",\n'
            '    "source_text": ""\n'
            "  }\n"
            "]\n\n"
            "limit_type examples: max, min, typical, breakdown.\n"
            "Do not include markdown fences or any text outside JSON.\n\n"
            f"Question:\n{query}\n\n"
            f"Datasheet Context:\n{context}{refs_line}"
        )
        full_prompt = self._to_full_prompt(
            system=JSON_SPEC_SYSTEM_PROMPT,
            user=user_prompt,
        )
        return PromptParts(
            system=JSON_SPEC_SYSTEM_PROMPT,
            user=user_prompt,
            full_prompt=full_prompt,
        )

    def build_rag_answer_prompt(
        self,
        query: str,
        context: str,
        context_chunk_ids: Optional[List[str]] = None,
    ) -> PromptParts:
        refs = ", ".join(context_chunk_ids or [])
        refs_line = f"\nContext chunk IDs: {refs}" if refs else ""
        user_prompt = (
            "Answer the question only from datasheet context.\n"
            "If constraints are present, report them directly with units and conditions.\n"
            "If multiple constraints are present, summarize each clearly in one concise explanation.\n"
            "If the context contains figure descriptions or diagram summaries not relevant to the question, ignore them.\n"
            "Only say the value is not specified when no relevant constraint exists in context.\n\n"
            f"Question:\n{query}\n\n"
            f"Datasheet Context:\n{context}{refs_line}"
        )
        full_prompt = self._to_full_prompt(
            system=RAG_ANSWER_SYSTEM_PROMPT,
            user=user_prompt,
        )
        return PromptParts(
            system=RAG_ANSWER_SYSTEM_PROMPT,
            user=user_prompt,
            full_prompt=full_prompt,
        )
