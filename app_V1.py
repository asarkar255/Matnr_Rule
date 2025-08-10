# app_matnr_scan.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import json

app = FastAPI(title="MATNR Scanner (ECC/S4 readiness)")

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
# Decls
DECL_CHAR_LEN_PAREN = re.compile(r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*\((\d+)\)\s*TYPE\s*C\b", re.IGNORECASE)
DECL_CHAR_LEN_EXPL  = re.compile(r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s*C\b[^.\n]*?\bLENGTH\b\s*(\d+)", re.IGNORECASE)
DECL_TYPE_MATNR     = re.compile(r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s+matnr\b", re.IGNORECASE)
DECL_LIKE_MATNR     = re.compile(r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*(LIKE|TYPE)\s+.*?-?matnr\b", re.IGNORECASE)

# Usage
MATNR_COMPON          = re.compile(r"\b(\w+)-matnr\b", re.IGNORECASE)
OFFSET_LEN_ON_COMP    = re.compile(r"\b(\w+-matnr)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
OFFSET_LEN_ON_VAR     = re.compile(r"\b(\w+)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
CONCATENATE_STMT      = re.compile(r"\bCONCATENATE\b(.+?)\bINTO\b", re.IGNORECASE | re.DOTALL)
STRING_OP_AND         = re.compile(r"(.+?)\s*&&\s*(.+?)")
STRING_TEMPLATE       = re.compile(r"\|\{[^}]*\b(matnr|MATNR)\b[^}]*\}\|")
SELECT_INTO_SINGLE    = re.compile(r"\bSELECT\b(?:\s+SINGLE)?\b[^.]*?\bmatnr\b[^.]*?\bINTO\b\s+@?(\w+)\b", re.IGNORECASE | re.DOTALL)
MOVE_STMT             = re.compile(r"\bMOVE\b\s+(.+?)\s+\bTO\b\s+(\w+)\s*\.", re.IGNORECASE)
ASSIGNMENT            = re.compile(r"\b(\w+)\s*=\s*([^\.\n]+)\.", re.IGNORECASE)
COMPARE_STMT          = re.compile(r"\bIF\b\s+(.+?)\.\s*", re.IGNORECASE | re.DOTALL)
SIMPLE_COMPARISON     = re.compile(r"(\w+(?:-matnr)?)\s*(=|<>|NE|EQ|LT|LE|GT|GE)\s*('?[\w\-]+'?|\w+(?:-matnr)?)", re.IGNORECASE)

# ---------- Helpers ----------
def line_of_offset(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1

def snippet_at(text: str, start: int, end: int) -> str:
    s = max(0, start-60); e = min(len(text), end+60)
    return text[s:e].replace("\n", "\\n")

# symbol table: var -> {"kind":"char","len":n} or {"kind":"matnr","len":40}
DECL_SPLIT_LINES = re.compile(r"\.", re.DOTALL)

def build_symbol_table(full_src: str) -> Dict[str, Dict]:
    table: Dict[str, Dict] = {}
    for stmt in DECL_SPLIT_LINES.split(full_src):
        s = stmt.strip()
        if not s:
            continue
        m = DECL_CHAR_LEN_PAREN.search(s)
        if m:
            var = m.group(2)
            ln  = int(m.group(3))
            table[var.lower()] = {"kind": "char", "len": ln}
        m = DECL_CHAR_LEN_EXPL.search(s)
        if m:
            var = m.group(2)
            ln  = int(m.group(3))
            table[var.lower()] = {"kind": "char", "len": ln}
        m = DECL_TYPE_MATNR.search(s)
        if m:
            var = m.group(2)
            table[var.lower()] = {"kind": "matnr", "len": 40}
        m = DECL_LIKE_MATNR.search(s)
        if m:
            var = m.group(2)
            table[var.lower()] = {"kind": "matnr", "len": 40}
    return table

def looks_like_matnr_token(tok: str) -> bool:
    return bool(re.search(r"-matnr\b", tok, re.IGNORECASE))

def is_char_len_lt_40(symtab: Dict[str, Dict], var: str, default_none=True) -> Optional[bool]:
    info = symtab.get(var.lower())
    if not info:
        return None if default_none else False
    if info["kind"] == "char":
        return info["len"] < 40
    if info["kind"] == "matnr":
        return False
    return None

def pack_issue(unit: Unit, issue_type, message, severity, line, code_snippet, suggestion, meta=None):
    meta = meta or {}
    base = unit.model_dump()
    base.pop("code", None)
    base.update({
        "issue_type": issue_type,
        "severity": severity,
        "line": line,
        "message": message,
        "suggestion": suggestion,
        "snippet": code_snippet.strip(),
        "meta": meta
    })
    return base

def _is_matnr_expr(symtab: Dict[str, Dict], expr: str) -> bool:
    expr = expr.strip()
    if looks_like_matnr_token(expr):
        return True
    mv = re.match(r"^(\w+)$", expr)
    if mv:
        v = mv.group(1)
        return symtab.get(v.lower(), {}).get("kind") == "matnr"
    return False

def scan_unit(unit: Unit, symtab: Dict[str, Dict]) -> Dict:
    src = unit.code or ""
    findings = []

    # 2) Concatenation
    for m in CONCATENATE_STMT.finditer(src):
        seg = m.group(0)
        if re.search(r"\b(matnr|MATNR)\b|-matnr\b", seg):
            findings.append(pack_issue(
                unit, "ConcatenationDetected",
                "MATNR used in CONCATENATE; concatenation uses full technical length (40).",
                "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                "For display: use CONVERSION_EXIT_MATN1_OUTPUT. Avoid persisting concatenated MATNR; prefer structured storage."
            ))
    for m in STRING_OP_AND.finditer(src):
        seg = m.group(0)
        if re.search(r"\b(matnr|MATNR)\b|-matnr\b", seg):
            findings.append(pack_issue(
                unit, "ConcatenationDetected",
                "MATNR used with && operator.",
                "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                "Same guidance as CONCATENATE."
            ))
    for m in STRING_TEMPLATE.finditer(src):
        findings.append(pack_issue(
            unit, "ConcatenationDetected",
            "MATNR used in string template.",
            "info", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
            "If only UI formatting, OK with MATN1 output conversion; avoid storing templates."
        ))

    # 3) Offset/length access
    for m in OFFSET_LEN_ON_COMP.finditer(src):
        token, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        findings.append(pack_issue(
            unit, "OffsetLengthAccess",
            f"Offset/length on MATNR component: +{off}({ln}).",
            "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
            "Avoid offset/length on MATNR; don’t rely on fixed length."
        ))
    for m in OFFSET_LEN_ON_VAR.finditer(src):
        var, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        if symtab.get(var.lower(), {}).get("kind") == "matnr":
            findings.append(pack_issue(
                unit, "OffsetLengthAccess",
                f"Offset/length on MATNR variable {var}: +{off}({ln}).",
                "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                "Avoid offset/length on MATNR."
            ))

    # 4) Old SELECT dest MATNR type conflict
    for m in SELECT_INTO_SINGLE.finditer(src):
        dest = m.group(1)
        dshort = is_char_len_lt_40(symtab, dest)
        if dshort is True:
            findings.append(pack_issue(
                unit, "OldSelectDestTypeConflict",
                f"SELECT ... MATNR INTO {dest} where {dest} is CHAR < 40.",
                "error", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                f"Change {dest} TYPE MATNR (40) or map to a 40-char field."
            ))
        elif dshort is None:
            findings.append(pack_issue(
                unit, "OldSelectDestTypeConflict",
                f"SELECT ... MATNR INTO {dest}; destination type unknown.",
                "info", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                "Ensure destination can hold 40 chars (TYPE MATNR)."
            ))

    # 5) Old move length conflict (MOVE / assignment)
    for m in MOVE_STMT.finditer(src):
        src_exp = m.group(1).strip()
        dest = m.group(2)
        if _is_matnr_expr(symtab, src_exp):
            dshort = is_char_len_lt_40(symtab, dest)
            if dshort is True:
                findings.append(pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"MOVE from MATNR expr to {dest} (CHAR < 40).",
                    "error", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                    f"Change {dest} TYPE MATNR."
                ))
            elif dshort is None:
                findings.append(pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"MOVE from MATNR expr to {dest} (type unknown).",
                    "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                    "Verify destination length; use TYPE MATNR."
                ))
    for m in ASSIGNMENT.finditer(src):
        dest, src_exp = m.group(1), m.group(2)
        if _is_matnr_expr(symtab, src_exp):
            dshort = is_char_len_lt_40(symtab, dest)
            if dshort is True:
                findings.append(pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"Assignment from MATNR expr to {dest} (CHAR < 40).",
                    "error", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                    f"Change {dest} TYPE MATNR."
                ))
            elif dshort is None:
                findings.append(pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"Assignment from MATNR expr to {dest} (type unknown).",
                    "warning", line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                    "Ensure destination supports 40 chars."
                ))

    # 1 & 6) Compare length conflicts
    for m in COMPARE_STMT.finditer(src):
        cond = m.group(1)
        for cmpm in SIMPLE_COMPARISON.finditer(cond):
            left, op, right = cmpm.group(1), cmpm.group(2), cmpm.group(3)
            # detect which side is MATNR (var or component)
            left_is_matnr = looks_like_matnr_token(left) or (symtab.get(left.lower(), {}).get("kind") == "matnr")
            right_is_matnr = looks_like_matnr_token(right) or (symtab.get(right.lower(), {}).get("kind") == "matnr")
            if not (left_is_matnr or right_is_matnr):
                continue
            # literals?
            is_lit = lambda s: bool(re.match(r"^'.*'$", s)) or s.isdigit()
            other = right if left_is_matnr else left
            other_short = None if is_lit(other) else is_char_len_lt_40(symtab, other, default_none=True)
            sev = "warning" if is_lit(other) else ("error" if other_short is True else "info")
            msg = "Comparison between MATNR and literal." if is_lit(other) else \
                  ("Comparison between MATNR and short CHAR var." if other_short is True else "Comparison with MATNR; verify other side length.")
            findings.append(pack_issue(
                unit, "CompareLengthConflict",
                msg, sev, line_of_offset(src, m.start()), snippet_at(src, m.start(), m.end()),
                "Use TYPE MATNR on the other side or normalize via conversion exit."
            ))

    # return same unit + findings
    res = unit.model_dump()
    res["matnr_findings"] = findings
    return res

def analyze_units(units: List[Unit]) -> List[Dict]:
    # build symtab from the concatenated code (best-effort cross-unit)
    flat_src = "\n".join(u.code or "" for u in units)
    symtab = build_symbol_table(flat_src)
    return [scan_unit(u, symtab) for u in units]

@app.post("/scan-matnr")
def scan_matnr(units: List[Unit]):
    return analyze_units(units)

@app.get("/health")
def health():
    return {"ok": True}
