from getpass import getuser
import os
import shutil
from pathlib import Path
import yaml

linux_user = getuser()

# BASE_PATH = f"/home/{linux_user}/ET-AI-IDP_V3.0/app"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load config.yaml
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")
config_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config_data = yaml.safe_load(f) or {}

_paths = config_data.get("paths", {})
_models = config_data.get("models", {})
_providers = config_data.get("providers", {})
_api = config_data.get("api", {})
_clustering = config_data.get("clustering", {})

RESULTS_DIR = os.getenv("RESULTS_DIR", str(Path(BASE_PATH) / _paths.get("results_dir", "results")))
TEMP_DIR = str(Path(BASE_PATH) / _paths.get("temp_dir", "temp_files"))


def _resolve_tesseract_cmd() -> str:
    cmd = os.getenv("TESSERACT_CMD") or _paths.get("tesseract_cmd")
    if cmd:
        if os.path.isabs(cmd):
            if os.path.exists(cmd):
                return cmd
        else:
            return cmd

    system_cmd = shutil.which("tesseract")
    if system_cmd:
        return system_cmd

    return "/usr/bin/tesseract"


TESSERACT_CMD = _resolve_tesseract_cmd()
PRIMARY_LLM = _models.get("primary_llm", "gpt-oss:20b")
SECONDARY_LLM = _models.get("secondary_llm", "llama3.1:8b")
FALLBACK_LLM = _models.get("fallback_llm", "gemma")
EMBEDDING_MODEL = _models.get("embedding", "text-embedding-3-small")

DEFAULT_PROVIDER = _providers.get("default", "litellm")
EMBEDDING_PROVIDER = _providers.get("embedding", DEFAULT_PROVIDER)
CLUSTERING_CONFIG = _clustering
# Max recursion cycles for the extraction agent ReAct loop
RECURSION_LIMIT = 25


def model_other_args(model_name: str, **kwargs):
    """Return provider arguments required by a specific model."""
    if model_name == "gpt-oss:20b":
        kwargs["use_responses_api"] = True
    return kwargs

OVERVIEW_PROMPT = """
summarize the above text with only case specific information. Any generic info should be ignored.
If there is no such case specific information, write one line about what the text is about.
Do not include any headings. Answer as a short paragraph.
Do not include any prefixes like "Based on the provided text, here's a summary".
Your answer MUST be a short paragraph. Maximum of 100 words.
Do not use bullet points, subheadings or any formatting. Only respond as a short paragraph.
"""


STRUCTURE_PROMPT = """
You are an AI specialized in extracting information from insurance documents.
Your task is to structure the given text without missing any information.
'overview' is mandatory in your output and must be a string of 40-50 words.

OUTPUT FORMAT:
You must respond in a json format with keys as a string and values as a dictionary of key (string) value (string) pairs.
Do not use lists.

RULES:
DO NOT DEVIATE FROM THE STRUCTURED OUTPUT RULES FOR TOOL_CALLS!!
DO NOT REPEAT INFORMATION IN MULTIPLE CATEGORIES, ESPECIALLY 'other_information'.
ONLY INCLUDE INFORMATION RELEVANT TO AN INSURANCE PROVIDER OR CUSTOMER, MEDICAL PROVIDER OR PATIENT.
DO NOT INCLUDE GENERIC INFORMATION SUCH AS NOTES, FOOTER TEXT, HEADER TEXT, LONG FORM INFORMATION.
DO NOT INCLUDE ANYTHING THAT IS NOT SPECIFIED OR UNKNOWN, IT IS OKAY IF THE OUTPUT IS EMPTY.
DO NOT INCLUDE MISSING INFORMATION SUCH AS "[Unreadable]".
IF MAJORITY OF THE INFORMATION IS UNKNWON, MISSING, UNREADABLE or NOT SPECIFIED, ONLY PROVIDE A SUMMARY.
IF THE OVERVIEW IS GIVEN AND SPECIFIES THE TEXT IS MISSING DATA, DEFAULT ALL KEYS TO EMTPY DICT AND RETURN ONLY OVERVIEW.
""" 


SUMMARIZER_PROMPT = """
You will be given page by page text extracted with OCR from complex documents. Since the text might not be ordered or formatted correctly,
you will need to use spatial analysis, logic and common sense to correctly decipher. If a given keyword does not have a
matching value in the text, you should NOT specify it.

You task is to read the text given to you, decipher and summarize all the key information
in Key value pairs WITHOUT redacting anything. Finally, provide an brief overview of the important information.

[X] if available usually represents a check box. [] Indicates unchecked. It can sometimes be deformed in many ways.
Analyze the text logically and map relationships as best as you can by following the below mentioned rules.
Do not make logical mistakes like "Self (Spouse)". Do not provide default values like "XYZ" or "John Doe" for missing info.

RULES:
1. Only give me what you can find
2. Overview must be one or two sentences summarizing only the given text
3. No more than 10 key value pairs per text
4. Do not include missing or incomplete information
5. Do not redact information
6. Do not include information that doesn't make sense
7. If you identify mispelled words or names, correct it in your output
8. If the given text does not have any key information relevant to an insured, only provide an overview
9. Do not return any placeholders with dummy content like [Unreadable] or [XYZ]
10. When given ICD codes and descriptions explicitly, use those in your response
11. If given context from previous pages -- do not include it in your response, only use it to understand the text
"""
