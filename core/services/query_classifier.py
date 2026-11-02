"""Query classification and semantic query expansion for ForgeAssist AI.

Classifies maintenance queries without external LLM dependencies, extracts
technical entities (error codes, parameters, components), and provides domain-specific
synonym expansion for improved BM25/keyword retrieval.
"""

import re
from typing import Dict, List, Set, Any

# Category constants
CATEGORY_ERROR_CODE = "ERROR_CODE"
CATEGORY_OPERATING_PARAM = "OPERATING_PARAM"
CATEGORY_MAINTENANCE = "MAINTENANCE"
CATEGORY_SAFETY = "SAFETY"
CATEGORY_TROUBLESHOOTING = "TROUBLESHOOTING"
CATEGORY_MACHINE_STATE = "MACHINE_STATE"
CATEGORY_GENERAL = "GENERAL"
CATEGORY_UNSUPPORTED = "UNSUPPORTED"

# Code detection regex (e.g. E17, E-17, E 17, F08, A01, ERR-102)
ERROR_CODE_PATTERN = re.compile(
    r"\b([A-Za-z]{1,4}[\s\-_]?\d{1,4})\b"
)

# Common off-topic / unsupported triggers (including explicitly excluded manual topics)
UNSUPPORTED_PATTERNS = [
    re.compile(r"\b(weather|football|cricket|movie|president|politics|stock market|capital of|joke|recipe)\b", re.IGNORECASE),
    re.compile(r"\b(who (is|are) you|what is your name|how are you|tell me about yourself)\b", re.IGNORECASE),
    re.compile(r"\b(cost|price|pricing|how much does|supplier phone|supplier contact|wiring diagram|circuit drawing|service contract)\b", re.IGNORECASE),
]

# Category keyword indicators
PARAM_KEYWORDS = {
    "temperature", "pressure", "voltage", "current", "speed", "flow",
    "range", "normal", "max", "maximum", "min", "minimum", "operating",
    "threshold", "limit", "tolerance", "bar", "psi", "celsius", "rpm",
    "degree", "degrees", "specification", "specs", "capacity", "level"
}

MAINTENANCE_KEYWORDS = {
    "maintenance", "service", "interval", "intervals", "schedule", "scheduled",
    "inspect", "inspection", "check", "checks", "routine", "daily", "weekly",
    "monthly", "annual", "annually", "hours", "replace", "replacement",
    "grease", "lubricate", "lubrication", "oil change", "filter", "clean"
}

SAFETY_KEYWORDS = {
    "safety", "precaution", "precautions", "warning", "warnings", "danger",
    "hazard", "hazards", "ppe", "protective", "eyewear", "gloves", "interlock",
    "interlocks", "lockout", "tagout", "loto", "emergency", "stop", "e-stop",
    "isolate", "isolation", "safe", "safeguard", "risk"
}

TROUBLESHOOTING_KEYWORDS = {
    "troubleshoot", "troubleshooting", "diagnose", "diagnosis", "cause", "causes",
    "problem", "issue", "fault", "defect", "fail", "failure", "failing",
    "low pressure", "high pressure", "high temperature", "overheating", "vibration",
    "noise", "leak", "leaking", "action", "remedy", "solution", "symptom"
}

MACHINE_STATE_KEYWORDS = {
    "startup", "start-up", "start", "starting", "shutdown", "shut-down", "stop",
    "standby", "idle", "power on", "power off", "initialize", "initialization",
    "ready state", "warm up", "cycle"
}

# Domain synonym mapping for industrial machinery manuals
SYNONYM_MAP: Dict[str, List[str]] = {
    "temperature": ["temp", "thermal", "degrees", "oil temperature", "operating temperature"],
    "oil": ["hydraulic fluid", "fluid", "lubricant", "oil level"],
    "pressure": ["hydraulic pressure", "system pressure", "discharge pressure", "bar", "psi"],
    "maintenance": ["service", "servicing", "schedule", "scheduled", "scheduled maintenance", "preventive maintenance", "inspection", "routine", "intervals"],
    "interval": ["schedule", "frequency", "periodic", "routine", "hours", "daily", "weekly", "monthly"],
    "inspect": ["inspection", "check", "examine", "verify", "monitor"],
    "safety": ["precaution", "warning", "caution", "danger", "hazard", "protective"],
    "troubleshoot": ["troubleshooting", "diagnose", "diagnosis", "fault finding", "cause", "remedy"],
    "operating": ["operation", "operational", "recommended", "normal"],
    "range": ["limits", "specifications", "tolerances", "parameters", "threshold"],
    "discharge": ["outlet", "exhaust", "output"],
    "intake": ["inlet", "suction", "input"],
    "hydraulic": ["fluid", "oil", "hydro"],
    "compressor": ["compression", "air compressor", "pump"],
    "filter": ["filtration", "strainer", "element"],
    "leak": ["leakage", "fluid loss", "seepage"],
    "fault": ["error", "alarm", "code", "defect"],
    "check": ["verify", "inspect", "examine", "measure"],
    "low": ["insufficient", "drop", "loss"],
    "high": ["excessive", "elevated", "over"],
}


def extract_error_codes(question: str) -> List[str]:
    """Extract standard industrial error/fault codes like E17, E-17, F08, E08."""
    matches = ERROR_CODE_PATTERN.findall(question)
    cleaned_codes = []
    # Ignore machine model identifiers, document tokens, and filter part numbers
    ignored_prefixes = ("REV", "PAGE", "DOC", "HP", "AC", "CNC", "ISO", "MOD", "RF", "AF")
    for match in matches:
        cleaned = re.sub(r"[\s\-_]", "", match).upper()
        if re.fullmatch(r"[A-Z]{1,3}\d{1,4}", cleaned):
            if not any(cleaned.startswith(pfx) for pfx in ignored_prefixes):
                cleaned_codes.append(cleaned)
    return list(dict.fromkeys(cleaned_codes))


def expand_query_terms(question: str) -> Set[str]:
    """Expand query terms using domain-specific industrial maintenance synonyms."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    expanded = set(words)

    # Check multi-word phrases first
    lower_q = question.lower()
    if "oil temp" in lower_q or "oil temperature" in lower_q:
        expanded.update(["hydraulic", "fluid", "temperature", "operating", "range", "normal"])
    if "hydraulic pressure" in lower_q:
        expanded.update(["pressure", "bar", "system", "operating", "normal"])
    if "air compressor" in lower_q:
        expanded.update(["compressor", "discharge", "temperature", "pressure"])

    for word in words:
        if word in SYNONYM_MAP:
            for syn in SYNONYM_MAP[word]:
                expanded.update(re.findall(r"[a-z0-9]+", syn.lower()))

    return expanded


def classify_question(question: str) -> Dict[str, Any]:
    """
    Classify a question into one of the designated categories and extract metadata.
    Does not use LLMs.
    """
    q_clean = question.strip()
    q_lower = q_clean.lower()
    tokens = set(re.findall(r"[a-z0-9]+", q_lower))

    # 1. Check unsupported / off-topic
    for pat in UNSUPPORTED_PATTERNS:
        if pat.search(q_clean):
            return {
                "category": CATEGORY_UNSUPPORTED,
                "confidence": 0.95,
                "error_codes": [],
                "keywords": list(tokens),
                "expanded_terms": list(tokens),
            }

    # 2. Check for explicit error/fault codes
    codes = extract_error_codes(q_clean)
    if codes or any(term in tokens for term in ["e17", "e24", "e31", "e42", "e08"]):
        # Consolidate found codes
        all_codes = list(codes)
        for term in ["e17", "e24", "e31", "e42", "e08"]:
            if term in tokens and term.upper() not in all_codes:
                all_codes.append(term.upper())

        return {
            "category": CATEGORY_ERROR_CODE,
            "confidence": 0.98,
            "error_codes": all_codes,
            "keywords": [c.lower() for c in all_codes] + ["error", "fault", "alarm"],
            "expanded_terms": list(expand_query_terms(q_clean)),
        }

    # 3. Score matching categories based on token overlap
    scores = {
        CATEGORY_SAFETY: len(tokens & SAFETY_KEYWORDS),
        CATEGORY_MAINTENANCE: len(tokens & MAINTENANCE_KEYWORDS),
        CATEGORY_TROUBLESHOOTING: len(tokens & TROUBLESHOOTING_KEYWORDS),
        CATEGORY_OPERATING_PARAM: len(tokens & PARAM_KEYWORDS),
        CATEGORY_MACHINE_STATE: len(tokens & MACHINE_STATE_KEYWORDS),
    }

    # Priority adjustments for compound queries
    if ("check" in tokens or "checks" in tokens or "when" in tokens or "cause" in tokens or "diagnose" in tokens) and ("low" in tokens or "drop" in tokens or "fail" in tokens or "abnormal" in tokens):
        scores[CATEGORY_TROUBLESHOOTING] += 5

    if "precaution" in tokens or "hazard" in tokens or "ppe" in tokens or "safety" in tokens:
        scores[CATEGORY_SAFETY] += 4

    if "interval" in tokens or "schedule" in tokens or "routine" in tokens or "servicing" in tokens:
        scores[CATEGORY_MAINTENANCE] += 4

    if ("temperature" in tokens or "pressure" in tokens or "range" in tokens) and not ("check" in tokens or "checks" in tokens):
        scores[CATEGORY_OPERATING_PARAM] += 3

    best_category, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score == 0:
        category = CATEGORY_GENERAL
        confidence = 0.50
    else:
        category = best_category
        confidence = min(0.60 + (best_score * 0.1), 0.95)

    expanded = expand_query_terms(q_clean)

    return {
        "category": category,
        "confidence": round(confidence, 2),
        "error_codes": [],
        "keywords": list(tokens),
        "expanded_terms": list(expanded),
    }
