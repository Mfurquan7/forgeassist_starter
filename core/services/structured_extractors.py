"""Generic structured information extractors for industrial machinery manuals.

Provides reusable, pattern-based parsers for:
- Operating parameters, ranges, and thresholds
- Maintenance intervals and schedules
- Safety precautions, warnings, and PPE requirements
- Troubleshooting symptoms, checks, and corrective actions
- Inspection checklist items

All extractors are domain-generic and do NOT hardcode machine-specific codes or text.
"""

import re
from typing import List, Dict, Any, Optional

# ============================================================
# OPERATING PARAMETERS & THRESHOLDS
# ============================================================

# Patterns for normal operating ranges, e.g.:
# "Normal hydraulic pressure: 120 to 140 bar"
# "Operating temperature range: 40 - 60 °C"
# "Discharge pressure: 7.5 - 8.5 bar"
RANGE_PATTERNS = [
    re.compile(
        r"(?:normal|recommended|operating|acceptable|standard)?\s*"
        r"([a-z\s]{3,35}?)\s*(?:range|level|pressure|temperature|speed|voltage|flow)?\s*"
        r"(?:is|are|:|=|\bbetween\b)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:to|-|–|and)\s*(\d+(?:\.\d+)?)\s*"
        r"([°\w/%]+(?:\s+[a-zA-Z]+)?)"
        r"(?:\s+(during|at|under)\s+([^.,;\n]+))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"([a-z\s]{3,35}?)\s*:\s*(\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?\s*[°\w/%]+(?:\s+[a-zA-Z]+)?)",
        re.IGNORECASE,
    ),
]

# Patterns for threshold limits (max/min/trigger limits)
THRESHOLD_PATTERNS = [
    re.compile(
        r"(maximum|minimum|max|min|high|low|trip|alarm|shutdown)\s+"
        r"([a-z\s]{3,30}?)\s*(?:is|are|:|=)?\s*"
        r"(\d+(?:\.\d+)?\s*[°\w/%]+(?:\s+[a-zA-Z]+)?)",
        re.IGNORECASE,
    ),
]


def extract_operating_parameters(text: str) -> List[Dict[str, Any]]:
    """Extract operating parameters and numerical ranges from text."""
    results = []
    seen = set()

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check range pattern 1
        for match in RANGE_PATTERNS[0].finditer(line_clean):
            subject, val_min, val_max, unit, cond_prep, condition = match.groups()
            param_name = subject.strip()
            # Filter noise
            if len(param_name) < 3 or param_name.lower().startswith(("the ", "a ", "and ")):
                param_name = re.sub(r"^(the|a|an|and)\s+", "", param_name, flags=re.IGNORECASE)
            
            value_str = f"{val_min} to {val_max} {unit}".strip()
            cond_str = f"{cond_prep} {condition}".strip() if condition else ""

            key = (param_name.lower(), value_str.lower())
            if key not in seen and len(param_name) > 2:
                seen.add(key)
                results.append({
                    "parameter": param_name.title(),
                    "value": value_str,
                    "condition": cond_str,
                    "raw_text": match.group(0).strip(),
                })

        # Check threshold patterns
        for match in THRESHOLD_PATTERNS[0].finditer(line_clean):
            limit_type, subject, val_unit = match.groups()
            param_name = f"{limit_type.capitalize()} {subject.strip()}".strip()
            key = (param_name.lower(), val_unit.lower())
            if key not in seen:
                seen.add(key)
                results.append({
                    "parameter": param_name,
                    "value": val_unit.strip(),
                    "condition": "",
                    "raw_text": match.group(0).strip(),
                })

    return results


# ============================================================
# MAINTENANCE SCHEDULES & INTERVALS
# ============================================================

INTERVAL_PATTERNS = [
    re.compile(r"\b(daily|weekly|monthly|quarterly|semi-annually|annually|yearly)\b", re.IGNORECASE),
    re.compile(r"\b(every\s+\d+\s*(?:operating\s+)?(?:hours|hrs|days|weeks|months|cycles))\b", re.IGNORECASE),
    re.compile(r"\b(\d+\s*(?:hours|hrs|cycles))\s*(?:maintenance|service|inspection|interval)\b", re.IGNORECASE),
]


def extract_maintenance_items(text: str) -> List[Dict[str, Any]]:
    """Extract structured maintenance tasks and their scheduled intervals."""
    items = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    current_interval = None
    current_tasks = []
    raw_lines = []

    def flush_interval():
        nonlocal current_interval, current_tasks, raw_lines
        if current_interval and current_tasks:
            items.append({
                "interval": current_interval,
                "items": list(dict.fromkeys(current_tasks)),
                "raw_text": "\n".join(raw_lines),
            })
        current_tasks = []
        raw_lines = []

    for line in lines:
        matched_interval = None
        for pat in INTERVAL_PATTERNS:
            m = pat.search(line)
            if m:
                matched_interval = m.group(1).capitalize()
                break

        # Check if line indicates a new interval header
        if matched_interval and (":" in line or len(line) < 60 or "maintenance" in line.lower() or "inspection" in line.lower()):
            flush_interval()
            current_interval = matched_interval
            raw_lines.append(line)

            # Check if tasks are on the same line after a colon
            if ":" in line:
                after_colon = line.split(":", 1)[1].strip()
                if after_colon:
                    # Split comma or semicolon separated tasks
                    parts = [p.strip() for p in re.split(r"[;,]", after_colon) if p.strip()]
                    current_tasks.extend(parts)
            continue

        # Check list items under current interval
        if current_interval:
            # Bullet, dash, number, or short sentence
            bullet_match = re.match(r"^[-*•\d.)\]\s]+(.*)", line)
            if bullet_match:
                task = bullet_match.group(1).strip()
                if len(task) > 4:
                    current_tasks.append(task)
                    raw_lines.append(line)
            elif any(line.lower().startswith(action) for action in ["check", "inspect", "replace", "clean", "verify", "lubricate", "drain", "test"]):
                current_tasks.append(line)
                raw_lines.append(line)
            elif len(line) < 100 and not line.endswith(":"):
                # Could be a single task line
                current_tasks.append(line)
                raw_lines.append(line)
            else:
                # Different section reached
                flush_interval()
                current_interval = None

    flush_interval()

    # Also handle table-formatted rows like "Daily | Check oil level and leakages"
    for line in lines:
        if "|" in line or "\t" in line:
            cells = [c.strip() for c in (line.split("|") if "|" in line else line.split("\t")) if c.strip()]
            if len(cells) >= 2:
                # Look for interval in cell 0
                for pat in INTERVAL_PATTERNS:
                    m = pat.search(cells[0])
                    if m:
                        interval = m.group(1).capitalize()
                        task_desc = cells[1]
                        items.append({
                            "interval": interval,
                            "items": [task_desc],
                            "raw_text": line,
                        })
                        break

    return items


# ============================================================
# SAFETY REQUIREMENTS & PRECAUTIONS
# ============================================================

SAFETY_HEADER_RE = re.compile(
    r"\b(DANGER|WARNING|CAUTION|NOTICE|SAFETY\s+NOTE|SAFETY\s+INSTRUCTION)\b",
    re.IGNORECASE,
)

PPE_TERMS = ["safety glasses", "protective eyewear", "gloves", "ear protection", "safety shoes", "hard hat", "face shield", "respirator", "ppe"]
LOCKOUT_TERMS = ["lockout", "tagout", "loto", "de-energize", "isolation", "disconnect power", "bleed pressure", "depressurize"]


def extract_safety_items(text: str) -> List[Dict[str, Any]]:
    """Extract safety warnings, precautions, and rules."""
    precautions = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        match = SAFETY_HEADER_RE.search(line)

        # 1. Header with warning/caution/danger
        if match:
            level = match.group(1).upper()
            content = line[match.end():].lstrip(":- ").strip()
            if not content and idx + 1 < len(lines):
                content = lines[idx + 1]

            if content:
                precautions.append({
                    "category": level.lower(),
                    "text": content,
                    "raw_text": f"{level}: {content}",
                })
            continue

        # 2. PPE requirements
        found_ppes = [ppe for ppe in PPE_TERMS if ppe in line_lower]
        if found_ppes and ("wear" in line_lower or "use" in line_lower or "required" in line_lower):
            precautions.append({
                "category": "ppe",
                "text": line,
                "raw_text": line,
            })
            continue

        # 3. Lockout / Tagout / Interlock procedures
        if any(term in line_lower for term in LOCKOUT_TERMS):
            precautions.append({
                "category": "lockout",
                "text": line,
                "raw_text": line,
            })
            continue

        # 4. Mandatory safety statements ("Never...", "Always...", "Do not operate...")
        if line.startswith(("Never ", "NEVER ", "Always ", "ALWAYS ", "Do not operate", "DO NOT OPERATE", "Ensure all guards")):
            precautions.append({
                "category": "general",
                "text": line,
                "raw_text": line,
            })

    return precautions


# ============================================================
# TROUBLESHOOTING & OBSERVATIONS
# ============================================================

TROUBLE_PATTERN = re.compile(
    r"(?:if|when|in case of)\s+([^,]+?),\s*(?:then\s+)?(.+)",
    re.IGNORECASE,
)


def extract_troubleshooting_items(text: str) -> List[Dict[str, Any]]:
    """Extract symptom-cause-remedy troubleshooting entries."""
    entries = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    current_symptom = None
    current_checks = []
    current_actions = []

    for line in lines:
        m = TROUBLE_PATTERN.search(line)
        if m:
            symptom = m.group(1).strip()
            action = m.group(2).strip()
            entries.append({
                "symptom": symptom,
                "checks": [action],
                "actions": [action],
                "raw_text": line,
            })
            continue

        # Table or label format: Problem: ... Check: ... Action: ...
        lower = line.lower()
        if lower.startswith(("problem:", "symptom:", "fault:", "observation:")):
            if current_symptom:
                entries.append({
                    "symptom": current_symptom,
                    "checks": current_checks,
                    "actions": current_actions,
                    "raw_text": f"Symptom: {current_symptom}",
                })
                current_checks = []
                current_actions = []
            current_symptom = line.split(":", 1)[1].strip()
        elif lower.startswith(("check:", "checks:", "inspection:", "possible cause:")):
            item = line.split(":", 1)[1].strip()
            if item:
                current_checks.append(item)
        elif lower.startswith(("action:", "remedy:", "solution:", "corrective action:")):
            item = line.split(":", 1)[1].strip()
            if item:
                current_actions.append(item)

    if current_symptom:
        entries.append({
            "symptom": current_symptom,
            "checks": current_checks,
            "actions": current_actions,
            "raw_text": f"Symptom: {current_symptom}",
        })

    return entries


# ============================================================
# COMBINED EXTRACTOR
# ============================================================

def extract_all_structured(text: str) -> Dict[str, Any]:
    """Run all domain-generic structured extractors on the given passage."""
    return {
        "operating_parameters": extract_operating_parameters(text),
        "maintenance": extract_maintenance_items(text),
        "safety": extract_safety_items(text),
        "troubleshooting": extract_troubleshooting_items(text),
    }
