# app_matnr_scan_with_decl.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import re

app = FastAPI(
    title="MATNR Scanner (ECC-ready) — usage + declaration-site findings, multi-line decls",
    version="1.2"
)

# ========= Models =========
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    line: Optional[int] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: int = 0
    end_line: int = 0
    code: str
    # optional input; we always produce our own findings
    matnr_findings: Optional[List[Finding]] = None

# ========= Regex Knowledge (MATNR focus) =========
# Declarations
DECL_CHAR_LEN_PAREN = re.compile(
    r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*\((\d+)\)\s*TYPE\s*C\b",
    re.IGNORECASE
)
DECL_CHAR_LEN_EXPL  = re.compile(
    r"\b(DATA|CONSTANTS|FIELD-SYMBOLS|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s*C\b[^.\n]*?\bLENGTH\b\s*(\d+)",
    re.IGNORECASE
)
DECL_TYPE_MATNR     = re.compile(
    r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*TYPE\s+matnr\b",
    re.IGNORECASE
)
DECL_LIKE_MATNR     = re.compile(
    r"\b(DATA|PARAMETERS|STATICS)\b[^.\n]*?\b(\w+)\b\s*(LIKE|TYPE)\s+.*?-?matnr\b",
    re.IGNORECASE
)

# Usage
MATNR_COMPON          = re.compile(r"\b(\w+)-matnr\b", re.IGNORECASE)
OFFSET_LEN_ON_COMP    = re.compile(r"\b(\w+-matnr)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
OFFSET_LEN_ON_VAR     = re.compile(r"\b(\w+)\s*\+\s*(\d+)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
CONCATENATE_STMT      = re.compile(r"\bCONCATENATE\b(.+?)\bINTO\b", re.IGNORECASE | re.DOTALL)
STRING_OP_AND         = re.compile(r"(.+?)\s*&&\s*(.+?)")
STRING_TEMPLATE       = re.compile(r"\|.*?\{[^}]*\b(matnr|MATNR)\b[^}]*\}\|", re.DOTALL)
SELECT_INTO_SINGLE    = re.compile(r"\bSELECT\b(?:\s+SINGLE)?\b[^.]*?\bmatnr\b[^.]*?\bINTO\b\s+@?(\w+)\b", re.IGNORECASE | re.DOTALL)
MOVE_STMT             = re.compile(r"\bMOVE\b\s+(.+?)\s+\bTO\b\s+(\w+)\s*\.", re.IGNORECASE)
ASSIGNMENT            = re.compile(r"\b(\w+)\s*=\s*([^\.\n]+)\.", re.IGNORECASE)
COMPARE_STMT          = re.compile(r"\bIF\b\s+(.+?)\.\s*", re.IGNORECASE | re.DOTALL)
SIMPLE_COMPARISON     = re.compile(r"(\w+(?:-matnr)?)\s*(=|<>|NE|EQ|LT|LE|GT|GE)\s*('?[\w\-]+'?|\w+(?:-matnr)?)", re.IGNORECASE)

# Multi-line declaration support (colon header with entries across lines)
def iter_statements_with_offsets(src: str):
    """Yield (stmt, start_off, end_off) for each '.'-terminated statement."""
    buf = []
    start = 0
    for i, ch in enumerate(src):
        buf.append(ch)
        if ch == ".":
            yield "".join(buf), start, i + 1
            buf = []
            start = i + 1
    if buf:
        yield "".join(buf), start, len(src)

def smart_split_commas(s: str):
    parts, cur, q = [], [], False
    for ch in s:
        if ch == "'":
            q = not q
            cur.append(ch)
        elif ch == "," and not q:
            parts.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]

DECL_HEADER_COLON = re.compile(r"^\s*(DATA|STATICS|CONSTANTS|PARAMETERS)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
DECL_ENTRY = re.compile(
    r"^\s*(?P<var>\w+)\s*(?:"
    r"TYPE\s+(?P<dtype>\w+)(?:\s+LENGTH\s+(?P<len>\d+))?(?:\s+DECIMALS\s+(?P<dec>\d+))?"
    r"|LIKE\s+(?P<like>\w+)"
    r"|\((?P<charlen>\d+)\)\s*TYPE\s*C"
    r")?",
    re.IGNORECASE
)

# ========= Helpers =========
def line_of_offset(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1

def snippet_at(text: str, start: int, end: int) -> str:
    s = max(0, start - 60); e = min(len(text), end + 60)
    return text[s:e].replace("\n", "\\n")

def looks_like_matnr_token(tok: str) -> bool:
    return bool(re.search(r"-matnr\b", tok, re.IGNORECASE)) or tok.strip().upper() == "MATNR"

# symbol table: var -> {"kind":"char"/"matnr","len":n}
DECL_SPLIT = re.compile(r"\.", re.DOTALL)

def build_symbol_table(full_src: str) -> Dict[str, Dict]:
    st: Dict[str, Dict] = {}
    for stmt, _, _ in iter_statements_with_offsets(full_src):
        s = stmt.strip()
        if not s:
            continue
        # single-line patterns
        m = DECL_CHAR_LEN_PAREN.search(s)
        if m:
            st[m.group(2).lower()] = {"kind": "char", "len": int(m.group(3))}
        m = DECL_CHAR_LEN_EXPL.search(s)
        if m:
            st[m.group(2).lower()] = {"kind": "char", "len": int(m.group(3))}
        m = DECL_TYPE_MATNR.search(s)
        if m:
            st[m.group(2).lower()] = {"kind": "matnr", "len": 40}
        m = DECL_LIKE_MATNR.search(s)
        if m:
            st[m.group(2).lower()] = {"kind": "matnr", "len": 40}

        # multi-line colon header
        mcol = DECL_HEADER_COLON.match(s)
        if not mcol:
            continue
        body = mcol.group(2)
        if body.endswith("."):
            body = body[:-1]
        for ent in smart_split_commas(body):
            em = DECL_ENTRY.match(ent)
            if not em:
                continue
            var = (em.group("var") or "").lower()
            if not var:
                continue
            if em.group("charlen"):
                st[var] = {"kind":"char","len":int(em.group("charlen"))}
                continue
            dtype = (em.group("dtype") or "").lower()
            like  = (em.group("like") or "").lower()
            if dtype == "matnr" or like == "matnr" or re.search(r"-matnr\b", dtype, re.IGNORECASE):
                st[var] = {"kind": "matnr", "len": 40}
                continue
            if dtype == "c":
                ln = int(em.group("len")) if em.group("len") else None
                if ln is not None:
                    st[var] = {"kind":"char","len":ln}
    return st

def _is_matnr_expr(symtab: Dict[str, Dict], expr: str) -> bool:
    expr = (expr or "").strip()
    if looks_like_matnr_token(expr):
        return True
    mv = re.match(r"^(\w+)$", expr)
    if mv:
        v = mv.group(1)
        return symtab.get(v.lower(), {}).get("kind") == "matnr"
    return False

def is_char_len_lt_40(symtab: Dict[str, Dict], var: str, default_none=True) -> Optional[bool]:
    info = symtab.get((var or "").lower())
    if not info:
        return None if default_none else False
    if info["kind"] == "char":
        ln = info.get("len")
        return (ln is not None and ln < 40)
    if info["kind"] == "matnr":
        return False
    return None

# ========= Declaration index (cross-include, multi-line aware) =========
class DeclSite:
    __slots__ = ("var","unit_idx","line","text")
    def __init__(self, var: str, unit_idx: int, line: int, text: str):
        self.var = var
        self.unit_idx = unit_idx
        self.line = line
        self.text = text

DECL_LINE_PATTERNS = [
    re.compile(r"^\s*(DATA|STATICS|CONSTANTS|PARAMETERS)\s*:\s*(\w+)\b.*\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*(DATA|STATICS|CONSTANTS|PARAMETERS)\s+(\w+)\b.*\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*FIELD-SYMBOLS\s*<(\w+)>\b.*\.\s*$", re.IGNORECASE),
]

def build_declaration_index(units: List[Unit]) -> Dict[str, List[DeclSite]]:
    idx: Dict[str, List[DeclSite]] = {}
    for uidx, u in enumerate(units):
        src = u.code or ""
        for stmt, s_off, _ in iter_statements_with_offsets(src):
            stripped = stmt.strip()
            # single-line
            for pat in DECL_LINE_PATTERNS:
                m = pat.match(stripped)
                if m:
                    var = (m.group(1) if pat.pattern.startswith(r"^\s*FIELD-SYMBOLS") else m.group(2)).lower()
                    if var:
                        idx.setdefault(var, []).append(DeclSite(var, uidx, line_of_offset(src, s_off), stripped))
                    break
            # multi-line colon header
            mcol = DECL_HEADER_COLON.match(stripped)
            if not mcol:
                continue
            body = mcol.group(2)
            if body.endswith("."):
                body = body[:-1]
            entries = smart_split_commas(body)
            # locate body start inside the statement for per-entry line calc
            body_rel_off = stripped.find(body)
            stmt_abs_start = s_off + (len(stmt) - len(stripped))  # adjust for leading whitespace
            rel = 0
            for ent in entries:
                if not ent:
                    continue
                subpos = body.find(ent, rel)
                if subpos < 0:
                    subpos = rel
                ent_abs_off = stmt_abs_start + body_rel_off + subpos
                rel = subpos + len(ent)
                em = DECL_ENTRY.match(ent)
                if not em:
                    continue
                var = (em.group("var") or "").lower()
                if not var:
                    continue
                idx.setdefault(var, []).append(DeclSite(var, uidx, line_of_offset(src, ent_abs_off), ent.strip()))
    return idx

# ========= Packaging helpers =========
def pack_issue(unit: Unit, issue_type, message, severity, start, end, suggestion, meta=None):
    src = unit.code or ""
    return {
        "pgm_name": unit.pgm_name,
        "inc_name": unit.inc_name,
        "type": unit.type,
        "name": unit.name,
        "class_implementation": unit.class_implementation,
        "start_line": unit.start_line,
        "end_line": unit.end_line,
        "issue_type": issue_type,
        "severity": severity,
        "line": line_of_offset(src, start),
        "message": message,
        "suggestion": suggestion or "",
        "snippet": snippet_at(src, start, end),
        "meta": meta or {}
    }

def pack_decl_issue(decl_unit: Unit, decl_line: int, decl_text: str,
                    issue_type: str, message: str, severity: str, suggestion: str, meta=None):
    return {
        "pgm_name": decl_unit.pgm_name,
        "inc_name": decl_unit.inc_name,
        "type": decl_unit.type,
        "name": decl_unit.name,
        "class_implementation": decl_unit.class_implementation,
        "start_line": decl_unit.start_line,
        "end_line": decl_unit.end_line,
        "issue_type": issue_type,
        "severity": severity,
        "line": decl_line,
        "message": message,
        "suggestion": suggestion or "",
        "snippet": decl_text,
        "meta": meta or {}
    }

def _emit_decl_mirrors_for_dest(dest_token: str,
                                usage_issue_type: str,
                                usage_severity: str,
                                usage_unit: Unit,
                                usage_line: int,
                                decl_index: Dict[str, List[DeclSite]],
                                units: List[Unit],
                                mirror_buckets: Dict[int, List[Dict[str, Any]]],
                                too_small: Optional[bool]):
    if not re.match(r"^[A-Za-z_]\w*$", dest_token or ""):
        return
    decls = decl_index.get(dest_token.lower()) or []
    if not decls:
        return
    for d in decls:
        decl_unit = units[d.unit_idx]
        if too_small is True:
            msg = f"Declaration of '{dest_token}' appears too small for 40-char MATNR used in {usage_unit.inc_name}/{usage_unit.name} at line {usage_line}."
            sev = usage_severity  # likely 'error'
            sug = "Change declaration to TYPE MATNR (40) or widen CHAR to 40."
            itype = "DeclarationMatnrSizeRisk"
        else:
            msg = f"Declaration of '{dest_token}' may be insufficient for 40-char MATNR used in {usage_unit.inc_name}/{usage_unit.name} at line {usage_line} (destination type unknown)."
            sev = "warning" if usage_severity != "info" else "info"
            sug = "Verify declaration supports 40 chars (TYPE MATNR)."
            itype = "DeclarationMatnrCapacityUnknown"
        mirror = pack_decl_issue(decl_unit, d.line, d.text, itype, msg, sev, sug, {})
        mirror_buckets.setdefault(d.unit_idx, []).append(mirror)

# ========= Scanner =========
def scan_unit(unit_idx: int,
              unit: Unit,
              symtab: Dict[str, Dict],
              decl_index: Dict[str, List[DeclSite]],
              units: List[Unit],
              mirror_buckets: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:

    src = unit.code or ""
    findings: List[Dict[str, Any]] = []

    # 2) Concatenation
    for m in CONCATENATE_STMT.finditer(src):
        seg = m.group(0)
        if re.search(r"\bmatnr\b|-matnr\b", seg, re.IGNORECASE):
            findings.append(pack_issue(
                unit, "ConcatenationDetected",
                "MATNR used in CONCATENATE; string ops use full technical length (40).",
                "warning", m.start(), m.end(),
                "For display: use CONVERSION_EXIT_MATN1_OUTPUT. Avoid persisting concatenated MATNR."
            ))
    for m in STRING_OP_AND.finditer(src):
        seg = m.group(0)
        if re.search(r"\bmatnr\b|-matnr\b", seg, re.IGNORECASE):
            findings.append(pack_issue(
                unit, "ConcatenationDetected",
                "MATNR used with && operator.",
                "warning", m.start(), m.end(),
                "Same guidance as CONCATENATE."
            ))
    for m in STRING_TEMPLATE.finditer(src):
        findings.append(pack_issue(
            unit, "ConcatenationDetected",
            "MATNR used in string template.",
            "info", m.start(), m.end(),
            "If only UI formatting, OK with MATN1 output conversion; avoid storing templates."
        ))

    # 3) Offset/length access
    for m in OFFSET_LEN_ON_COMP.finditer(src):
        token, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        findings.append(pack_issue(
            unit, "OffsetLengthAccess",
            f"Offset/length on MATNR component: +{off}({ln}).",
            "warning", m.start(), m.end(),
            "Avoid offset/length on MATNR; don’t rely on fixed length."
        ))
    for m in OFFSET_LEN_ON_VAR.finditer(src):
        var, off, ln = m.group(1), int(m.group(2)), int(m.group(3))
        if _is_matnr_expr(symtab, var):
            findings.append(pack_issue(
                unit, "OffsetLengthAccess",
                f"Offset/length on MATNR variable {var}: +{off}({ln}).",
                "warning", m.start(), m.end(),
                "Avoid offset/length on MATNR."
            ))

    # 4) Old SELECT dest MATNR type conflict
    for m in SELECT_INTO_SINGLE.finditer(src):
        dest = m.group(1)
        dshort = is_char_len_lt_40(symtab, dest)
        if dshort is True:
            usage = pack_issue(
                unit, "OldSelectDestTypeConflict",
                f"SELECT ... MATNR INTO {dest} where {dest} is CHAR < 40.",
                "error", m.start(), m.end(),
                f"Change {dest} TYPE MATNR (40) or map to a 40-char field."
            )
            findings.append(usage)
            _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                        usage["line"], decl_index, units, mirror_buckets, True)
        elif dshort is None:
            usage = pack_issue(
                unit, "OldSelectDestTypeConflict",
                f"SELECT ... MATNR INTO {dest}; destination type unknown.",
                "info", m.start(), m.end(),
                "Ensure destination can hold 40 chars (TYPE MATNR)."
            )
            findings.append(usage)
            _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                        usage["line"], decl_index, units, mirror_buckets, False)

    # 5) Old move length conflict (MOVE and '=')
    for m in MOVE_STMT.finditer(src):
        src_exp = m.group(1).strip()
        dest = m.group(2)
        if _is_matnr_expr(symtab, src_exp):
            dshort = is_char_len_lt_40(symtab, dest)
            if dshort is True:
                usage = pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"MOVE from MATNR expr to {dest} (CHAR < 40).",
                    "error", m.start(), m.end(),
                    f"Change {dest} TYPE MATNR."
                )
                findings.append(usage)
                _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                            usage["line"], decl_index, units, mirror_buckets, True)
            elif dshort is None:
                usage = pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"MOVE from MATNR expr to {dest} (type unknown).",
                    "warning", m.start(), m.end(),
                    "Verify destination length; use TYPE MATNR."
                )
                findings.append(usage)
                _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                            usage["line"], decl_index, units, mirror_buckets, False)

    for m in ASSIGNMENT.finditer(src):
        dest, src_exp = m.group(1), m.group(2)
        if _is_matnr_expr(symtab, src_exp):
            dshort = is_char_len_lt_40(symtab, dest)
            if dshort is True:
                usage = pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"Assignment from MATNR expr to {dest} (CHAR < 40).",
                    "error", m.start(), m.end(),
                    f"Change {dest} TYPE MATNR."
                )
                findings.append(usage)
                _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                            usage["line"], decl_index, units, mirror_buckets, True)
            elif dshort is None:
                usage = pack_issue(
                    unit, "OldMoveLengthConflict",
                    f"Assignment from MATNR expr to {dest} (type unknown).",
                    "warning", m.start(), m.end(),
                    "Ensure destination supports 40 chars."
                )
                findings.append(usage)
                _emit_decl_mirrors_for_dest(dest, usage["issue_type"], usage["severity"], unit,
                                            usage["line"], decl_index, units, mirror_buckets, False)

    # 1) & 6) Compare length conflicts
    for m in COMPARE_STMT.finditer(src):
        cond = m.group(1)
        for cmpm in SIMPLE_COMPARISON.finditer(cond):
            left, op, right = cmpm.group(1), cmpm.group(2), cmpm.group(3)
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
            usage = pack_issue(
                unit, "CompareLengthConflict",
                msg, sev, m.start(), m.end(),
                "Use TYPE MATNR on the other side or normalize via conversion exit."
            )
            findings.append(usage)
            # Declaration-site mirror for the non-literal side if it’s a simple var
            if not is_lit(other) and re.match(r"^[A-Za-z_]\w*$", other or ""):
                _emit_decl_mirrors_for_dest(other, usage["issue_type"], usage["severity"], unit,
                                            usage["line"], decl_index, units, mirror_buckets,
                                            (other_short is True))

    # return same unit + findings
    res = unit.model_dump()
    res["matnr_findings"] = findings
    return res

# ========= Orchestrator =========
def analyze_units(units: List[Unit]) -> List[Dict[str, Any]]:
    # Build global symbol table (multi-include) and declaration index with exact lines
    flat_src = "\n".join(u.code or "" for u in units)
    symtab = build_symbol_table(flat_src)
    decl_index = build_declaration_index(units)

    # Scan each unit; collect declaration mirrors per unit index
    mirror_buckets: Dict[int, List[Dict[str, Any]]] = {}
    out = []
    for idx, u in enumerate(units):
        out.append(scan_unit(idx, u, symtab, decl_index, units, mirror_buckets))

    # Inject declaration-site mirrors into corresponding unit results
    for uidx, mirrors in mirror_buckets.items():
        if mirrors and uidx < len(out):
            out[uidx].setdefault("matnr_findings", []).extend(mirrors)
    return out

# ========= API =========
@app.post("/scan-matnr")
def scan_matnr(units: List[Unit]):
    return analyze_units(units)

@app.get("/health")
def health():
    return {"ok": True}
