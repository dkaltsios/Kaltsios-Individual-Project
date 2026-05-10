"""
demo/parse_metadata.py – extracts the 19 metadata fields from a free-text
clinical description.

Primary path: OpenAI GPT-4o-mini with JSON-schema structured output.
Fallback path: legacy regex pipeline (used when the API key is absent or the
               API call fails for any reason).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# ── Shared constants ───────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "age":                   45,
    "sex":                   "male",
    "fitzpatrick":           2,
    "lesion_size_mm":        5.0,
    "diameter_2":            4.0,
    "anatomical_site_clean": "trunk",
    "bleed":                 0,
    "hurt":                  0,
    "itch":                  0,
    "changed":               0,
    "grew":                  0,
    "elevation":             0,
    "smoking":               0,
    "alcohol_consumption":   0,
    "cancer_history":        0,
    "skin_cancer_history":   0,
    "pesticide":             0,
    "has_piped_water":       1,
    "has_sewage_system":     1,
}

KNOWN_SITES = [
    "face", "trunk", "upper extremity", "lower extremity",
    "scalp", "neck", "palms/soles", "unknown",
]

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}


@dataclass
class ParseResult:
    metadata:      dict[str, Any]
    found:         list[str] = field(default_factory=list)
    defaults_used: list[str] = field(default_factory=list)
    parser_used:   str = "regex"
    error:         str | None = None


# ── OpenAI structured-output parser ──────────────────────────────────────────

_JSON_SCHEMA = {
    "name": "clinical_metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "age":                   {"type": "integer",               "description": "Patient age in years (1-110)"},
            "sex":                   {"type": "string",  "enum": ["male", "female"]},
            "fitzpatrick":           {"type": "integer",               "description": "Fitzpatrick skin type 1-6"},
            "lesion_size_mm":        {"type": "number",                "description": "Largest lesion diameter in mm"},
            "diameter_2":            {"type": "number",                "description": "Second lesion diameter in mm (width if given as AxB mm, else 80% of size)"},
            "anatomical_site_clean": {
                "type": "string",
                "enum": ["face", "trunk", "upper extremity", "lower extremity",
                         "scalp", "neck", "palms/soles", "unknown"],
            },
            "bleed":               {"type": "integer", "enum": [0, 1]},
            "hurt":                {"type": "integer", "enum": [0, 1]},
            "itch":                {"type": "integer", "enum": [0, 1]},
            "changed":             {"type": "integer", "enum": [0, 1]},
            "grew":                {"type": "integer", "enum": [0, 1]},
            "elevation":           {"type": "integer", "enum": [0, 1]},
            "smoking":             {"type": "integer", "enum": [0, 1]},
            "alcohol_consumption": {"type": "integer", "enum": [0, 1]},
            "cancer_history":      {"type": "integer", "enum": [0, 1]},
            "skin_cancer_history": {"type": "integer", "enum": [0, 1]},
            "pesticide":           {"type": "integer", "enum": [0, 1]},
            "has_piped_water":     {"type": "integer", "enum": [0, 1]},
            "has_sewage_system":   {"type": "integer", "enum": [0, 1]},
        },
        "required": list(DEFAULTS.keys()),
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT = """\
You are a clinical data-extraction assistant. Given a patient description,
extract exactly the 19 structured fields listed in the JSON schema.

Guidelines:
- Binary fields (0/1): set 1 if the condition is present or reported, 0 if
  negated ("no bleeding", "denies smoking") or simply not mentioned.
- anatomical_site_clean: map body-part synonyms to the nearest canonical value
  (e.g. "back" → "trunk", "arm" → "upper extremity", "leg" → "lower extremity",
  "cheek/forehead/nose/ear/lip" → "face", "palm/sole/finger/toe" → "palms/soles").
- For fields not inferable from the text, use: age=45, sex="male",
  fitzpatrick=2, lesion_size_mm=5.0, diameter_2=4.0,
  anatomical_site_clean="trunk", all binary fields=0,
  has_piped_water=1, has_sewage_system=1.
"""


def _parse_with_openai(text: str, api_key: str) -> ParseResult:
    import site, sys  # ensure user site-packages is on the path
    for _p in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
        temperature=0,
    )

    raw = json.loads(response.choices[0].message.content)

    # Build result dict, coercing types to match DEFAULTS
    metadata = dict(DEFAULTS)
    found: list[str] = []
    defaults_used: list[str] = []

    for key, default in DEFAULTS.items():
        val = raw.get(key)
        if val is None:
            defaults_used.append(key)
            continue
        # Light type coercion
        try:
            if isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
            else:
                val = str(val)
        except (ValueError, TypeError):
            defaults_used.append(key)
            continue

        # Range checks
        if key == "age" and not (1 <= val <= 110):
            defaults_used.append(key)
            continue
        if key == "fitzpatrick" and not (1 <= val <= 6):
            defaults_used.append(key)
            continue
        if key in ("lesion_size_mm", "diameter_2") and not (0.0 < val <= 200):
            defaults_used.append(key)
            continue
        if key == "anatomical_site_clean" and val not in KNOWN_SITES:
            defaults_used.append(key)
            continue

        metadata[key] = val
        # Only count as "found" if it differs from the default
        if val != default:
            found.append(key)
        else:
            defaults_used.append(key)

    return ParseResult(
        metadata=metadata,
        found=found,
        defaults_used=defaults_used,
        parser_used="openai",
    )


# ── Legacy regex parser (fallback) ────────────────────────────────────────────

def _age(text: str) -> int | None:
    patterns = [
        r"(\d{1,3})\s*[-–]?\s*year(?:s)?[-\s]?old",
        r"age[d]?\s*(?:of\s*)?:?\s*(\d{1,3})",
        r"(\d{1,3})\s*yo\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 110:
                return v
    return None


def _sex(text: str) -> str | None:
    if re.search(r"\bfemale\b|\bwoman\b|\bgirl\b|\bshe\b|\bher\b", text, re.IGNORECASE):
        return "female"
    if re.search(r"\bmale\b|\bman\b|\bboy\b|\bhe\b|\bhis\b", text, re.IGNORECASE):
        return "male"
    return None


def _fitzpatrick(text: str) -> int | None:
    m = re.search(
        r"(?:fitzpatrick|skin\s+type|type)\s*:?\s*(vi|iv|v|iii|ii|i)\b",
        text, re.IGNORECASE,
    )
    if m:
        return ROMAN.get(m.group(1).lower())
    m = re.search(
        r"(?:fitzpatrick|skin\s+type|type)\s*:?\s*([1-6])\b",
        text, re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


def _size(text: str) -> float | None:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*mm\b",
        r"size\s*(?:of\s*)?:?\s*(\d+(?:\.\d+)?)",
        r"diameter\s*(?:of\s*)?:?\s*(\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0.1 <= v <= 200:
                return v
    return None


def _diameter2(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
    if m:
        return float(m.group(2))
    m = re.search(r"diameter\s*2\s*:?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _site(text: str) -> str | None:
    for site in sorted(KNOWN_SITES, key=len, reverse=True):
        if re.search(re.escape(site), text, re.IGNORECASE):
            return site
    synonyms = {
        "back":    "trunk",   "chest":   "trunk",   "abdomen": "trunk",
        "stomach": "trunk",   "arm":     "upper extremity",
        "forearm": "upper extremity",    "hand": "upper extremity",
        "leg":     "lower extremity",    "foot": "lower extremity",
        "feet":    "lower extremity",    "thigh": "lower extremity",
        "shin":    "lower extremity",    "cheek": "face",
        "forehead":"face",               "nose":  "face",
        "ear":     "face",               "lip":   "face",
        "palm":    "palms/soles",        "sole":  "palms/soles",
        "finger":  "palms/soles",        "toe":   "palms/soles",
    }
    for word, canonical in synonyms.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            return canonical
    return None


def _binary_symptom(text: str, keywords: list[str]) -> int | None:
    neg_pattern = r"(?:no|not|without|denies|negative for|no\s+history\s+of)\s+(?:\w+\s+){0,3}"
    for kw in keywords:
        if re.search(neg_pattern + re.escape(kw), text, re.IGNORECASE):
            return 0
        if re.search(rf"\b{re.escape(kw)}", text, re.IGNORECASE):
            return 1
    return None


SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "bleed":               ["bleed", "bleeding", "bleeds", "blood"],
    "hurt":                ["hurt", "hurts", "pain", "painful", "tender", "sore"],
    "itch":                ["itch", "itching", "itchy", "pruritic", "pruritus"],
    "changed":             ["changed", "changing", "change", "altered", "alteration"],
    "grew":                ["grew", "growing", "growth", "enlarged", "enlarging", "increase"],
    "elevation":           ["elevated", "elevation", "raised", "nodular", "bump"],
    "smoking":             ["smok", "smoker", "tobacco", "cigarette"],
    "alcohol_consumption": ["alcohol", "drink", "drinking", "ethanol"],
    "cancer_history":      ["cancer history", "history of cancer", "malignancy"],
    "skin_cancer_history": ["skin cancer", "melanoma history", "previous melanoma",
                            "history of skin cancer", "previous skin cancer"],
    "pesticide":           ["pesticide", "herbicide", "insecticide", "agricultural chemical"],
    "has_piped_water":     ["piped water", "running water", "tap water"],
    "has_sewage_system":   ["sewage", "sewer", "drainage system"],
}


def _parse_with_regex(text: str) -> ParseResult:
    result = dict(DEFAULTS)
    found: list[str] = []
    defaults_used: list[str] = []

    def _apply(key: str, value):
        if value is not None:
            result[key] = value
            found.append(key)
        else:
            defaults_used.append(key)

    _apply("age",                   _age(text))
    _apply("sex",                   _sex(text))
    _apply("fitzpatrick",           _fitzpatrick(text))
    _apply("anatomical_site_clean", _site(text))

    size = _size(text)
    _apply("lesion_size_mm", size)
    d2 = _diameter2(text)
    _apply("diameter_2", d2 if d2 is not None else (size * 0.8 if size is not None else None))

    for key, keywords in SYMPTOM_KEYWORDS.items():
        _apply(key, _binary_symptom(text, keywords))

    return ParseResult(metadata=result, found=found, defaults_used=defaults_used, parser_used="regex")


# ── Public entry point ─────────────────────────────────────────────────────────

def parse_clinical_text(text: str, api_key: str | None = None) -> ParseResult:
    """Parse free-text clinical description into a metadata dictionary.

    Uses the OpenAI API when an *api_key* is provided (or ``OPENAI_API_KEY``
    env var is set).  Falls back silently to the regex pipeline otherwise.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")

    if key:
        try:
            return _parse_with_openai(text, key)
        except Exception as exc:
            result = _parse_with_regex(text)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

    return _parse_with_regex(text)


EXAMPLE_TEXT = (
    "45-year-old male patient, Fitzpatrick type II, with a 6×4 mm lesion on the trunk. "
    "The lesion has been itching and has changed in appearance over the past 3 months. "
    "It is slightly elevated. Patient has a history of smoking. No cancer history."
)
