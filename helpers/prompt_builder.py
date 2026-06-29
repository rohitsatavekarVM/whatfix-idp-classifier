"""Centralized prompt building for LLM operations"""
from typing import Any, Dict, List, Tuple, Optional
from andromeda import HumanMessage, SystemMessage
from .document_config import DocumentTypeConfig


class PromptBuilder:
    """Builds prompts for various document processing tasks"""
    
    @staticmethod
    def build_messages(system_prompt: str, user_prompt: str):
        """Create message list for LLM"""
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    
    @staticmethod
    def build_classification_prompt(text: str) -> Tuple[str, str]:
        """Build primary classification prompt"""
        type_definitions = "\n".join([
            f"{k}, {v}" for k, v in DocumentTypeConfig.TYPE_DEFINITIONS.items()
        ])
        
        system_prompt = (
            "You need to classify the kind of insurance information in the page's raw text "
            "in one word."
        )
        user_prompt = (
            f"{text}\n\n"
            "Carefully analyze the context of the above raw text.\n"
            "Understand the kind of information it contains. The text is extracted with OCR,\n"
            "hence may contain errors or jumbled words.\n\n"
            "Category, Definition\n"
            f"{type_definitions}\n"
            "OTHER, Uncategorized documents\n\n"
            "Return only a valid JSON object with keys: document_type, confidence.\n"
            "document_type must be one of the allowed categories.\n"
            "confidence must be a number between 0.0 and 1.0.\n"
            "Do not include subtype or evidence fields in this response.\n\n"
            "Which of the above categories does the raw text belong to?\n"
            "Output only JSON."
        )
        return system_prompt, user_prompt
    
    @staticmethod
    def build_subclassification_prompt(text: str, hl_type: str) -> Tuple[str, str]:
        """Build sub-classification prompt"""
        sub_defs = DocumentTypeConfig.SUB_TYPE_DEFINITIONS.get(hl_type, {})
        lines = [f"{k}, {v}" for k, v in sub_defs.items()]
        lines.append("OTHER, Uncategorized documents")
        sub_types_list = "\n".join(lines)
        
        system_prompt = (
            "You need to classify the kind of insurance information in the page's raw text "
            "in one word."
        )
        user_prompt = (
            f"{text}\n\n"
            f"Carefully analyze the context of the above raw text from a group of pages classified as type {hl_type}.\n"
            "Understand the kind of information it contains. The text is extracted with OCR,\n"
            "hence may contain errors or jumbled words.\n\n"
            "Category, Definition\n"
            f"{sub_types_list}\n\n"
            "Return only a valid JSON object with keys: sub_type, confidence.\n"
            "sub_type must be one of the allowed categories.\n"
            "confidence must be a number between 0.0 and 1.0.\n"
            "Do not include high-level document type or evidence fields in this response.\n\n"
            "Which of the above categories based on definition does the raw text belong to?\n"
            "Output only JSON."
        )
        return system_prompt, user_prompt

    @staticmethod
    def build_page_range_classification_prompt(
        page_block: str,
        allowed_types: List[str],
        page_numbers: List[int],
    ) -> Tuple[str, str]:
        """Build prompt for page-range classification with strict page boundaries."""
        allowed = ", ".join(allowed_types)
        pages = ", ".join(str(p) for p in page_numbers)
        min_page = min(page_numbers) if page_numbers else 0
        max_page = max(page_numbers) if page_numbers else 0
        system_prompt = (
            "You classify ordered OCR pages into contiguous page ranges.\n"
            "Understand the kind of information it contains. The text is extracted with OCR,\n"
            "hence may contain errors or jumbled words.\n\n"
            "Return only ranges using the exact page numbers.\n"
            "Prefer --- PAGE x --- instances instead of other page numbers that might be present in the text.\n"
            "Do not invent pages, do not skip pages, and do not overlap ranges.\n"
            "Every provided page must be covered exactly once.\n"
            "If a title page, separator, cover page, or trailing/filler page sits between pages of the same main document, "
            "assign it to that same surrounding document type when context indicates continuity.\n"
            "Use only allowed document_type values. "
            "Prefer longest contiguous ranges with the same document_type; split only when document_type changes. "
            "If the content appears to continue onto the next page, assign both pages to the same document type whenever possible. "
            "Reason about the correct document type based on the text, boundaries, and context. "
            "Return a valid JSON object with a key named types. Each item in types must include start_page, end_page, document_type, and confidence. "
            "The confidence value must be between 0.0 and 1.0."
        )
        user_prompt = (
            f"Allowed document types and definitions: {allowed}\n"
            f"Valid page numbers: [{pages}]\n"
            f"Page span minimum: {min_page}, maximum: {max_page}\n\n"
            "Task:\n"
            "1) Read all pages in order.\n"
            "2) Segment into contiguous ranges.\n"
            "3) Output exact start_page/end_page values from valid page numbers only.\n"
            "4) Ensure full coverage with no overlaps and no gaps.\n\n"
            f"{page_block}\n"
        )
        return system_prompt, user_prompt
    
    @staticmethod
    def build_summarize_prompt(
        text: str,
        sub_type: str,
        hl_type: str,
        schema_hint: Optional[str],
        include_actions: bool,
        summary_format: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Build summarization prompt"""
        base = (
            "You are an assistant to an insurance adjuster reviewing insurance claim documentation.\n"
            "Provide a structured markdown summary of the key information in the page's raw text. Use short paragraphs and bullet points; prioritize readability and clarity.\n"
            "Make the summary visually clear by breaking up details with blank lines or line breaks as appropriate. Avoid crowding information—each major point or fact should be visually distinct.\n"
            "Focus on the information relevant to this document type and "
            "summarize from the adjuster's perspective. Do not refer to 'this document' or include generic transitions. Focus directly on key facts, and keep events in clear chronological order. Do not repeat information or introduce conflicting details.\n"
            "Avoid titles, headings, or tables—present only the summary. Use concise language and bullet lists. Eliminate filler words and do not start with phrases like 'Here is a summary.'\n"
            "If the source text includes densely packed or run-together numbers, codes, or facts, separate and space them for clarity in the summary (for example: 'Claim reference: xxx', 'Loss date: xxx'). Ensure each event, vehicle, party, or monetary value appears on a new line or as a distinct bullet.\n"
            "When summarizing key figures (like expenses or totals), present them in a direct, business-appropriate format. For all key figure labels (for example, 'Medical Specials:'), always make the label bold, as in '**Medical Specials:** $xxx in medical expenses incurred to date.' Never hallucinate information beyond the given text.\n\n"
            "Return only valid JSON with keys: text, confidence.\n"
            "text should be the structured summary. confidence should be a number between 0.0 and 1.0 based on how directly the source text supports the summary; use 0.0 only when no reliable supported summary can be produced.\n"
            "Do not include any additional keys. Output only JSON.\n"
        )
        if summary_format:
            base += f"\nSummary format:\n{summary_format}\n"

        if schema_hint:
            base += f"\nFocus areas hint: {schema_hint}\n"

        suffix = ""
        if include_actions:
            suffix = (
                "Lastly, provide one or two action items for the insurance adjuster in the response with heading **Recommended next best actions:**"
                "assuming the given information is accurate. Explicitly mention deadlines, amounts, or other "
                f"critical details relevant for document type {sub_type or hl_type}. Do not include notes."
            )
        
        return base + suffix, f"{text}\n"
    
    @staticmethod
    def build_claim_selection_prompt(overview: str, candidates: str) -> Tuple[str, str]:
        """Build claim number selection prompt"""
        system_prompt = (
            "You are an assistant to an insurance adjuster who is reviewing a document.\n"
            "You need to extract the correct claim number from the list of potential numbers from the document."
            # "Claim number must be 7-15 characters only."
        )
        
        first_line = (overview or "").split("\n")[0]
        user_prompt = (
            f"Overview: {first_line}\n\n"
            f"List of potential claim numbers: {candidates}\n\n"
            "Understand the above overview to identify the key names and information.\n"
            "You are given a list of potential claim numbers extracted from the document.\n"
            "These are simply extracted by looking for numbers in the text along with 10 words before and after.\n"
            "Based on the context of the overview, extract the correct claim number from the list.\n"
            "Claim number must be 7-20 characters only. It is not the same as full file name or path. Ex: A00341299\n"
            "Reply in one word with the correct claim number from the list."
        )
        return system_prompt, user_prompt
