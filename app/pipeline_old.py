import json
import os
import re
from typing import Dict, Any, Tuple

import pandas as pd
import pdfplumber
import requests

from openai import OpenAI
from .extraction_templates import TEMPLATE_MAP, GENERIC_TEMPLATE

# Config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Ollama Model: {OLLAMA_MODEL}")

USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _extract_json_from_text(text: str) -> Any:
    fenced = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Simple cleanup attempt for common trailing comma issues
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return json.loads(text)


def _build_system_prompt() -> str:
    return (
        "You are a strict data extraction engine. "
        "You CANNOT speak to the user. You MUST output ONLY valid JSON. "
        "Do not include any preamble, analysis, or explanation. "
        "Your output must start with '{' and end with '}'."
    )


def _build_classification_prompt(doc_text: str) -> str:
    return f"""
Analyze the following document text and classify it into one of the following categories:
- "profit_and_loss": Financial statements showing income, expenses, and profits (P&L, Income Statement).
- "balance_sheet": Financial statements showing assets, liabilities, and equity.
- "cash_flow": Cash Flow statements.
- "generic": Any other type of document or if unsure.

Output a JSON object with:
- "document_type": The classification label.
- "confidence_score": A number between 0.0 and 1.0 indicating your confidence.
- "reasoning": A brief explanation of why you chose this category.

DOCUMENT CONTENT (Truncated):
{doc_text[:4000]}
"""


def _build_extraction_prompt(doc_text: str, template: Dict[str, Any], doc_type: str) -> str:
    template_str = json.dumps(template, indent=2)
    
    specific_instructions = ""
    if doc_type == "profit_and_loss":
        specific_instructions = (
            "IMPORTANT for PROFIT & LOSS:\n"
            "- Extract data for ALL available time periods (years/quarters) found in the columns.\n"
            "- Map tabular data to the 'annual' and 'quarterly' lists in the JSON.\n"
            "- Be precise with 'sales_cr', 'net_profit_cr', etc. Convert values to Crores if needed, or extract as is if already in Crores."
        )

    return f"""
You are an expert financial data extractor.
The document type has been identified as: {doc_type}

Your task is to extract structured data from the document into the provided JSON Schema.

{specific_instructions}

CRITICAL RULES:
1. Output ONLY valid JSON.
2. FOLLOW the TEMPLATE structure exactly.
3. If a field is not found, use null or 0.0 as appropriate.
4. Extract ALL rows/items for lists like 'annual' or 'rows'.
5. Do NOT add any markdown formatting like ```json ... ```. Just raw JSON.
6. Do NOT explain your answer.

TEMPLATE:
{template_str}

DOCUMENT CONTENT:
{doc_text}

JSON OUTPUT:
"""


# ---------- File loaders ----------

def _detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in [".xls", ".xlsx"]:
        return "excel"
    return "unknown"


def _load_pdf_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            parts.append(f"--- PAGE {i+1} ---\n{text}")
    return "\n\n".join(parts)


def _load_excel_text(path: str, max_rows_per_sheet: int = 300) -> str:
    xl = pd.ExcelFile(path)
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        # Convert all columns to string to avoid serialization issues
        df = df.astype(str)
        if len(df) > max_rows_per_sheet:
            df = df.head(max_rows_per_sheet)
        parts.append(f"### SHEET: {sheet} ###")
        parts.append(df.to_csv(index=False))
    return "\n\n".join(parts)


# ---------- Ollama + OpenAI callers ----------

def _call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """
    Returns (raw_response_text, model_name_used)
    """
    # 1. Try Ollama
    try:
        url = f"{OLLAMA_HOST}/v1/chat/completions"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], f"ollama:{OLLAMA_MODEL}"
    except Exception as e:
        print(f"Ollama error: {e}")
        if not USE_OPENAI_FALLBACK:
            raise e

    # 2. Fallback to OpenAI
    print("Falling back to OpenAI...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set but OpenAI fallback requested.")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content, f"openai:{OPENAI_MODEL}"


# ---------- Public pipeline ----------

def process_document(document_path: str, template_path: str = None) -> Tuple[dict, str]:
    """
    Returns (json_result, model_used)
    """
    # 1. Load Text
    file_type = _detect_file_type(document_path)
    if file_type == "pdf":
        doc_text = _load_pdf_text(document_path)
    elif file_type == "excel":
        doc_text = _load_excel_text(document_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or Excel.")

    # 2. Classify Document
    print("Classifying document...")
    sys_prompt = "You are a document classifier. Output strict JSON."
    class_prompt = _build_classification_prompt(doc_text)
    
    class_resp, class_model = _call_llm(sys_prompt, class_prompt)
    try:
        class_json = _extract_json_from_text(class_resp)
        doc_type = class_json.get("document_type", "generic")
        confidence = class_json.get("confidence_score", 0.0)
        print(f"Classified as: {doc_type} (Confidence: {confidence})")
    except Exception as e:
        print(f"Classification failed, defaulting to generic. Error: {e}")
        doc_type = "generic"
        confidence = 0.0
        class_json = {"document_type": "generic", "confidence_score": 0.0, "reasoning": "Classification failed"}

    # 3. Select Template
    # Normalize doc_type
    doc_type_key = doc_type.lower().replace(" ", "_")
    template = TEMPLATE_MAP.get(doc_type_key)
    
    # Fallback to key containing "profit" matching
    if not template and "profit" in doc_type_key:
        template = TEMPLATE_MAP["profit_and_loss"]
    
    # If generic or unknown, use the user-provided template file (format_template.json)
    if not template:
        print("Using generic/user provided template.")
        if template_path and os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    template = json.load(f)
            except Exception as e:
                print(f"Failed to load user template: {e}")
                template = GENERIC_TEMPLATE
        else:
            template = GENERIC_TEMPLATE

    # 4. Extract Data
    print(f"Extracting data using template for: {doc_type}...")
    ext_sys_prompt = _build_system_prompt()
    print('ext_sys_prompt', ext_sys_prompt)
    ext_user_prompt = _build_extraction_prompt(doc_text, template, doc_type_key)
    print('ext_user_prompt', ext_user_prompt)
    
    raw_result, model_used = _call_llm(ext_sys_prompt, ext_user_prompt)
    print('raw_result', raw_result)
    json_result = _extract_json_from_text(raw_result)
    print('json_result', json_result)

    # 5. Merge Classification Metadata
    # Safely inject metadata
    if isinstance(json_result, dict):
        if "metadata" in json_result and isinstance(json_result["metadata"], dict):
            json_result["metadata"]["classification"] = class_json
        else:
            json_result["classification"] = class_json
    else:
        # If result is not a dict (e.g. list), wrap it
        json_result = {
            "data": json_result,
            "classification": class_json
        }

    return json_result, model_used
