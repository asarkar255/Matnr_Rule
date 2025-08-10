# app_matnr_scan_llm.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Set
import re, json, os

# ---------- Optional OpenAI client (LLM assist) ----------
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
try:
    if USE_LLM:
        from openai import OpenAI
        openai_client = OpenAI()
    else:
        openai_client = None
except Exception:
    USE_LLM = False
    openai_client = None

app = FastAPI(title="MATNR Scanner (ECC/S4) + LLM assist")

# ---------- Input model (matches your format) ----------
class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: int
    end_line: int
    code: str

# ---------- Regex knowledge (ECC-era, S/4 impact) ----------
# Declarations
DECL_CHAR_LEN_PAREN = re.compile(
    r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*\((\d+)\)\s*TYPE\s*C\b",
    re.IGNORECASE,
)
DECL_CHAR_LEN_EXPL = re.compile(
    r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s*C\b[^.\n]*?\bLENGTH\b\s*(\d+)",
    re.IGNORECASE,
)
DECL_TYPE_MATNR = re.compile(
    r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s+matnr\b",
    re.IGNORECASE,
)
DECL_LIKE_MATNR = re.compile(
    r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*(LIKE|TYPE)\s+.*?-?matnr\b",
    re.IGNORECASE,
)

# Usage patterns
MATNR_COMPON = re.compile(r"\b(\w+)-matnr\b", re.IGNORECASE)
OFFSET_LEN_ON_COMP = re.compile(r"\b(\w+-matnr)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
OFFSET_LEN_ON_VAR = re.compile(r"\b(\w+)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
CONCATENATE_STMT = re.compile(r"\bCONCATENATE\b(.+?)\bINTO\b", re.IGNORECASE | re.DOTALL)
STRING_OP_AND = re.compile(r"(.+?)\s*&&\s*(.+?)")
STRING_TEMPLATE = re.compile(r"\|\{[^}]*\b(matnr|MATNR)\b[^}]*\}\|")
SELECT_INTO_SINGLE = re.compile(
    r"\bSELECT\b(?:\s+SINGLE)?\b[^.]*?\bmatnr\b[^.]*?\bINTO\b\s+@?(\w+)\b", re.IGNORECASE | re.DOTALL
)
MOVE_STMT = re.compile(r"\bMOVE\b\s+(.+?)\s+\bTO\b\s+(\w+)\s*\.", re.IGNORECASE)
ASSIGNMENT = re.compile(r"\b(\w+)\s*=\s*([^\.\n]+)\.", re.IGNORECASE)
COMPARE_STMT = re.compile(r"\bIF\b\s+(.+?)\.\s*", re.IGNORECASE | re.DOTALL)
SIMPLE_COMPARISON = re.compile(
    r"(\w+(?:-matnr)?)\s*(=|<>|NE|EQ|LT|LE|GT|GE)\s*('?[\w\-]+'?|\w+(?:-matnr)?)", re.IGNORECASE
)

# ---------- Helpers ----------
def line_of_offset(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1

def snippet_at(text: str, start: int, end: int) -> str:
    s = max(0, start - 60)
    e = min(len(text), end + 60)
    return text[s:e].replace("\n", "\\n")

DECL_SPLIT_LINES = re.compile(r"\.", re.DOTALL)

def build_symbol_table(full_src: str) -> Dict[str, Dict]:
    """
    Build a best-effort symbol table across ALL units in the request.
    Captures: TYPE c LENGTH n, (n) TYPE c, TYPE matnr, LIKE *-matnr.
    """
    table: Dict[str, Dict] = {}
    for stmt in DECL_SPLIT_LINES.split(full_src):
        s = stmt.strip()
        if not s:
            continue
        m = DECL_CHAR_LEN_PAREN.search(s)
        if m:
            var = m.group(2)
            ln = int(m.group(3))
            table[var.lower()] = {"kind": "char", "len": ln, "source": "decl"}
        m = DECL_CHAR_LEN_EXPL.search(s)
        if m:
            var = m.group(2)
            ln = int(m.group(3))
            table[var.lower()] = {"kind": "char", "len": ln, "source": "decl"}
        m = DECL_TYPE_MATNR.search(s)
        if m:
            var = m.group(2)
            table[var.lower()] = {"kind": "matnr", "len": 40, "source": "decl"}
        m = DECL_LIKE_MATNR.search(s)
        if m:
            var = m.group(2)
            table[var.lower()] = {"kind": "matnr", "len": 40, "source": "decl"}
    return table

def looks_like_matnr_token(tok: str) -> bool:
    return bool(re.search(r"-matnr\b", tok, re.IGNORECASE))

def is_char_len_lt_40(symtab: Dict[str, Dict], var: str, default_none=True) -> Optional[bool]:
    info = symtab.get(var.lower())
    if not info:
        return None if default_none else False
    if info["kind"] == "char":
        return info.get("len", 0) < 40
    if info["kind"] == "matnr":
        return False
    return None

def pack_issue(
    unit: Unit,
    issue_type,
    message,
    severity,
    line,
    code_snippet,
    suggestion,
    meta=None,
):
    meta = meta or {}
    base = unit.model_dump()
    base.pop("code", None)
    base.update(
        {
            "issue_type": issue_type,
            "severity": severity,
            "line": line,
            "message": message,
            "suggestion": suggestion,
            "snippet": code_snippet.strip(),
            "meta": meta,
        }
    )
    return base

def _is_matnr_expr(symtab: Dict[str, Dict], expr: str) -> Tuple[bool, Optional[bool]]:
    """
    Returns (is_matnr_like, is_llm_inferred)
    """
    expr = expr.strip()
    if looks_like_matnr_token(expr):
        return True, False
    mv = re.match(r"^(\w+)$", expr)
    if mv:
        v = mv.group(1).lower()
        info = symtab.get(v)
        if not info:
            return False, None
        return info.get("kind") == "matnr", (info.get("source") == "llm")
    return False, None

# ---------- LLM assist ----------
LLM_CACHE: Dict[str, Dict] = {}

def collect_unknown_matnr_candidates(units: List[Unit], symtab: Dict[str, Dict]) -> Set[str]:
    """
    Find variable names used in MATNR contexts whose type is unknown.
    """
    suspects: Set[str] = set()
    for u in units:
        src = u.code or ""

        # From SELECT ... INTO dest
        for m in SELECT_INTO_SINGLE.finditer(src):
            dest = m.group(1).lower()
            if dest not in symtab:
                suspects.add(dest)

        # From MOVE/assignment if src or dest looks like matnr but is unknown
        for m in MOVE_STMT.finditer(src):
            src_exp, dest = m.group(1).strip(), m.group(2).lower()
            mv = re.match(r"^(\w+)$", src_exp)
            if mv and mv.group(1).lower() not in symtab and looks_like_matnr_token(src_exp) is False:
                suspects.add(mv.group(1).lower())
            if dest not in symtab:
                suspects.add(dest)

        for m in ASSIGNMENT.finditer(src):
            dest, src_exp = m.group(1).lower(), m.group(2).strip()
            if dest not in symtab:
                suspects.add(dest)
            mv = re.match(r"^(\w+)$", src_exp)
            if mv and mv.group(1).lower() not in symtab and looks_like_matnr_token(src_exp) is False:
                suspects.add(mv.group(1).lower())

        # From comparisons
        for m in COMPARE_STMT.finditer(src):
            cond = m.group(1)
            for cmpm in SIMPLE_COMPARISON.finditer(cond):
                left, right = cmpm.group(1), cmpm.group(3)
                for token in (left, right):
                    v = re.match(r"^(\w+)$", token or "")
                    if v:
                        vn = v.group(1).lower()
                        if vn not in symtab:
                            suspects.add(vn)
    return suspects

def llm_infer_types(units: List[Unit], candidates: Set[str]) -> Dict[str, Dict]:
    """
    Ask GPT-5 to guess which candidate variables are MATNR (len 40).
    Return dict: var -> {"kind": "matnr"|"char"|"unknown", "len": int, "confidence": float}
    """
    if not USE_LLM or not candidates:
        return {}

    # Cache key (simple)
    key = hash((tuple(sorted(candidates)), tuple(u.code for u in units)))
    if key in LLM_CACHE:
        return LLM_CACHE[key]

    # Compact context
    MAX_CHARS = 15000
    joined = "\n\n".join(f"--- {u.pgm_name}/{u.inc_name}/{u.type}/{u.name} ---\n{u.code}" for u in units)
    context = joined[:MAX_CHARS]

    prompt = f"""
You are an ABAP code reviewer. Identify which variables are material numbers (MATNR) compatible with S/4HANA 40-char semantics.

Given the ABAP code (snippets from multiple includes) and a list of variables with unknown declarations, infer each variable's likely type.
Return strict JSON mapping lowercase variable -> {{"kind":"matnr"|"char"|"unknown","len":int,"confidence":0..1}}.
- "matnr" implies length 40.
- "char" implies classical character with stated length.
- Use evidence like moves to/from <struct>-matnr, SELECT ... INTO var (field matnr), comparisons with -matnr, etc.
- If insufficient evidence, use "unknown" with low confidence.

Unknown variables (lowercase): {sorted(list(candidates))}

ABAP context {context}:
Return ONLY JSON.
""".strip()

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise static analyzer for ABAP."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        LLM_CACHE[key] = parsed
        return parsed
    except Exception:
        return {}

def merge_llm_hints(symtab: Dict[str, Dict], hints: Dict[str, Dict], min_conf: float = 0.7) -> Dict[str, Dict]:
    merged = dict(symtab)
    for var, info in (hints or {}).items():
        try:
            kind = info.get("kind", "unknown")
            conf = float(info.get("confidence", 0.0))
            if kind == "matnr" and conf >= min_conf:
                merged[var.lower()] = {"kind": "matnr", "len": 40, "source": "llm", "confidence": conf}
            elif kind == "char" and conf >= min_conf:
                ln = int(info.get("len", 0)) or 0
                merged[var.lower()] = {"kind": "char", "len": ln, "source": "llm", "confidence": conf}
        except Exception:
            continue
    return merged

# ---------- Core scanning ----------
def scan_unit(unit: Unit, symtab: Dict[str, Dict]) -> Dict:
    src = unit.code or ""
    findings = []

    # 2) Concatenation (CONCATENATE / && / string template)
    for m in CONCATENATE_STMT.finditer(src):
        seg = m.group(0)
        if re.search(r"\b(matnr|MATNR)\b|-matnr\b", seg):
            findings.append(
                pack_issue(
                    unit,
                    "ConcatenationDetected",
                    "MATNR used in CONCATENATE; concatenation uses full technical length (40).",
                    "warning",
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    "For display: use CONVERSION_EXIT_MATN1_OUTPUT. Avoid persisting concatenated MATNR; prefer structured storage.",
                )
            )
    for m in STRING_OP_AND.finditer(src):
        seg = m.group(0)
        if re.search(r"\b(matnr|MATNR)\b|-matnr\b", seg):
            findings.append(
                pack_issue(
                    unit,
                    "ConcatenationDetected",
                    "MATNR used with && operator.",
                    "warning",
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    "Same guidance as CONCATENATE.",
                )
            )
    for m in STRING_TEMPLATE.finditer(src):
        findings.append(
            pack_issue(
                unit,
                "ConcatenationDetected",
                "MATNR used in string template.",
                "info",
                line_of_offset(src, m.start()),
                snippet_at(src, m.start(), m.end()),
                "If only UI formatting, OK with MATN1 output conversion; avoid storing templates.",
            )
        )

    # 3) Offset/length access
    for m in OFFSET_LEN_ON_COMP.finditer(src):
        token, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        findings.append(
            pack_issue(
                unit,
                "OffsetLengthAccess",
                f"Offset/length on MATNR component: +{off}({ln}).",
                "warning",
                line_of_offset(src, m.start()),
                snippet_at(src, m.start(), m.end()),
                "Avoid offset/length on MATNR; don’t rely on fixed length.",
            )
        )
    for m in OFFSET_LEN_ON_VAR.finditer(src):
        var, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        info = symtab.get(var.lower(), {})
        if info.get("kind") == "matnr":
            findings.append(
                pack_issue(
                    unit,
                    "OffsetLengthAccess",
                    f"Offset/length on MATNR variable {var}: +{off}({ln}).",
                    "warning",
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    "Avoid offset/length on MATNR.",
                )
            )

    # 4) Old SELECT dest type conflict
    for m in SELECT_INTO_SINGLE.finditer(src):
        dest = m.group(1)
        dshort = is_char_len_lt_40(symtab, dest)
        meta = {}
        if dshort is True:
            findings.append(
                pack_issue(
                    unit,
                    "OldSelectDestTypeConflict",
                    f"SELECT ... MATNR INTO {dest} where {dest} is CHAR < 40.",
                    "error",
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    f"Change {dest} TYPE MATNR (40) or map to a 40-char field.",
                    meta,
                )
            )
        elif dshort is None:
            findings.append(
                pack_issue(
                    unit,
                    "OldSelectDestTypeConflict",
                    f"SELECT ... MATNR INTO {dest}; destination type unknown.",
                    "info",
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    "Ensure destination can hold 40 chars (TYPE MATNR).",
                    meta,
                )
            )

    # 5) Old move/assignment length conflict
    for m in MOVE_STMT.finditer(src):
        src_exp = m.group(1).strip()
        dest = m.group(2)
        is_mat, llm_flag = _is_matnr_expr(symtab, src_exp)
        if is_mat:
            dshort = is_char_len_lt_40(symtab, dest)
            meta = {"llm_inferred": bool(llm_flag)}
            if dshort is True:
                findings.append(
                    pack_issue(
                        unit,
                        "OldMoveLengthConflict",
                        f"MOVE from MATNR expr to {dest} (CHAR < 40).",
                        "error",
                        line_of_offset(src, m.start()),
                        snippet_at(src, m.start(), m.end()),
                        f"Change {dest} TYPE MATNR.",
                        meta,
                    )
                )
            elif dshort is None:
                findings.append(
                    pack_issue(
                        unit,
                        "OldMoveLengthConflict",
                        f"MOVE from MATNR expr to {dest} (type unknown).",
                        "warning",
                        line_of_offset(src, m.start()),
                        snippet_at(src, m.start(), m.end()),
                        "Verify destination length; use TYPE MATNR.",
                        meta,
                    )
                )
    for m in ASSIGNMENT.finditer(src):
        dest, src_exp = m.group(1), m.group(2)
        is_mat, llm_flag = _is_matnr_expr(symtab, src_exp)
        if is_mat:
            dshort = is_char_len_lt_40(symtab, dest)
            meta = {"llm_inferred": bool(llm_flag)}
            if dshort is True:
                findings.append(
                    pack_issue(
                        unit,
                        "OldMoveLengthConflict",
                        f"Assignment from MATNR expr to {dest} (CHAR < 40).",
                        "error",
                        line_of_offset(src, m.start()),
                        snippet_at(src, m.start(), m.end()),
                        f"Change {dest} TYPE MATNR.",
                        meta,
                    )
                )
            elif dshort is None:
                findings.append(
                    pack_issue(
                        unit,
                        "OldMoveLengthConflict",
                        f"Assignment from MATNR expr to {dest} (type unknown).",
                        "warning",
                        line_of_offset(src, m.start()),
                        snippet_at(src, m.start(), m.end()),
                        "Ensure destination supports 40 chars.",
                        meta,
                    )
                )

    # 1 & 6) Compare length conflicts
    for m in COMPARE_STMT.finditer(src):
        cond = m.group(1)
        for cmpm in SIMPLE_COMPARISON.finditer(cond):
            left, op, right = cmpm.group(1), cmpm.group(2), cmpm.group(3)
            left_is_mat, left_llm = _is_matnr_expr(symtab, left)
            right_is_mat, right_llm = _is_matnr_expr(symtab, right)
            if not (
                left_is_mat
                or right_is_mat
                or looks_like_matnr_token(left)
                or looks_like_matnr_token(right)
            ):
                continue
            is_lit = lambda s: bool(re.match(r"^'.*'$", s)) or s.isdigit()
            other = right if (left_is_mat or looks_like_matnr_token(left)) else left
            other_short = None if is_lit(other) else is_char_len_lt_40(symtab, other, default_none=True)
            sev = "warning" if is_lit(other) else ("error" if other_short is True else "info")
            msg = (
                "Comparison between MATNR and literal."
                if is_lit(other)
                else ("Comparison between MATNR and short CHAR var." if other_short is True else "Comparison with MATNR; verify other side length.")
            )
            findings.append(
                pack_issue(
                    unit,
                    "CompareLengthConflict",
                    msg,
                    sev,
                    line_of_offset(src, m.start()),
                    snippet_at(src, m.start(), m.end()),
                    "Use TYPE MATNR on the other side or normalize via conversion exit.",
                    {"llm_inferred": bool(left_llm or right_llm)},
                )
            )

    res = unit.model_dump()
    res["matnr_findings"] = findings
    return res

def analyze_units(units: List[Unit]) -> List[Dict]:
    # Build baseline symtab from all code in this request
    flat_src = "\n".join(u.code or "" for u in units)
    symtab = build_symbol_table(flat_src)

    # LLM assist for unknowns used in MATNR contexts
    candidates = collect_unknown_matnr_candidates(units, symtab)
    hints = llm_infer_types(units, candidates)
    symtab2 = merge_llm_hints(symtab, hints, min_conf=0.7)

    # Scan using the merged table
    return [scan_unit(u, symtab2) for u in units]

@app.post("/scan-matnr")
def scan_matnr(units: List[Unit]):
    return analyze_units(units)

@app.get("/health")
def health():
    return {"ok": True, "use_llm": USE_LLM, "model": OPENAI_MODEL if USE_LLM else None}
