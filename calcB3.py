# calcB3.py
# App Streamlit: múltiplos PDFs B3/XP, rateio por valor, PM, cotações (Yahoo),
# Banco/Carteira com posições; TOTAL descontando vendas; pop-up de movimentações
# diretamente na tabela (popover) sem navegar; tabelas centralizadas e
# auto-atualização de cotações (30s invisível). Integração Gmail + PDF com senha.

import io
import re
import math
import unicodedata
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from time import time as _now

import pandas as pd
import streamlit as st

# --- Auto refresh invisível (30s) ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

AUTO_REFRESH_MS = 30_000  # 30s
if st_autorefresh:
    st_autorefresh(interval=AUTO_REFRESH_MS, limit=None, key="auto_refresh_30s")

# --- PDF parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# --- Quotes
try:
    import yfinance as yf
except Exception:
    yf = None

# --- Gmail / Google
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# =============================================================================
# Utils
# =============================================================================
TZ = ZoneInfo("America/Sao_Paulo")
UTC = timezone.utc

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def brl(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def parse_brl_number(s: str) -> float:
    return float(s.replace(".", "").replace(",", "."))

def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _fmt_dt_local(dt) -> str:
    try:
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None:
                dt = dt.tz_localize(UTC)
            return dt.astimezone(TZ).strftime("%d/%m/%Y %H:%M")
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        try:
            return pd.to_datetime(dt, utc=True).tz_convert(TZ).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return str(dt)

# =============================================================================
# Senhas para PDF (via Secrets)
# =============================================================================
def _norm_pwd_variants(raw: str) -> list[str]:
    """Gera variações úteis: remove pontuação; datas dd/mm/aaaa -> ddmmaaaa e yyyymmdd."""
    v = str(raw or "").strip()
    if not v:
        return []
    outs = []

    outs.append(v)  # original

    # sem pontuação
    v_nopunct = re.sub(r"[^\w]", "", v, flags=re.UNICODE)
    if v_nopunct and v_nopunct not in outs:
        outs.append(v_nopunct)

    # datas dd/mm/aaaa
    m = re.fullmatch(r"(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})", v)
    if m:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        for k in (f"{d}{mth}{y}", f"{y}{mth}{d}"):
            if k not in outs:
                outs.append(k)

    return outs

def _collect_pdf_passwords_from_secrets() -> list[str]:
    def _pull(path):
        v = st.secrets
        for k in path:
            try:
                v = v.get(k)
            except Exception:
                return None
            if v is None:
                return None
        return v

    candidates = []
    try:
        for path in [("pdf_password",), ("pdf_passwords",)]:
            v = _pull(path)
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                candidates += [str(x) for x in v if str(x).strip()]
            else:
                if str(v).strip():
                    candidates.append(str(v))
        for path in [("pdf","password"), ("pdf","passwords")]:
            v = _pull(path)
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                candidates += [str(x) for x in v if str(x).strip()]
            else:
                if str(v).strip():
                    candidates.append(str(v))
    except Exception:
        pass

    final = []
    for raw in candidates:
        for v in _norm_pwd_variants(raw):
            if v not in final:
                final.append(v)
    return final

_PWD_LIST = _collect_pdf_passwords_from_secrets()
_PWD_SIG = sha1("||".join(_PWD_LIST).encode()) if _PWD_LIST else ""

# =============================================================================
# PDF → texto (com suporte a senha)
# =============================================================================
def extract_text_from_pdf(file_bytes: bytes, passwords: Optional[list[str]] = None) -> tuple[str, str]:
    """
    Tenta extrair texto usando pdfplumber, PyMuPDF (fitz) e PyPDF2.
    Suporta PDFs protegidos com senha (lista de tentativas).
    Retorna (texto, "engine usada").
    """
    pwds = [None] + (passwords or [])

    # 1) pdfplumber
    if pdfplumber is not None:
        for p in pwds:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes), password=p) as pdf:
                    # a) strict
                    text = ""
                    for page in pdf.pages:
                        text += (page.extract_text(x_tolerance=1, y_tolerance=1) or "") + "\n"
                    if text.strip():
                        return text, f"pdfplumber (strict tol=1/1){' + pwd' if p else ''}"
            except Exception:
                pass
            try:
                with pdfplumber.open(io.BytesIO(file_bytes), password=p) as pdf:
                    # b) default
                    text = ""
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
                    if text.strip():
                        return text, f"pdfplumber (default tol){' + pwd' if p else ''}"
            except Exception:
                pass
            try:
                with pdfplumber.open(io.BytesIO(file_bytes), password=p) as pdf:
                    # c) reconstructed words
                    chunks = []
                    for page in pdf.pages:
                        words = page.extract_words(use_text_flow=True) or []
                        if words:
                            words_sorted = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))
                            line_y, line_words = None, []
                            for w in words_sorted:
                                y = round(w.get("top", 0), 1)
                                txt = w.get("text", "")
                                if line_y is None:
                                    line_y = y
                                if abs(y - line_y) > 0.8:
                                    chunks.append(" ".join(line_words)); line_words = [txt]; line_y = y
                                else:
                                    line_words.append(txt)
                            if line_words:
                                chunks.append(" ".join(line_words))
                    text = "\n".join(chunks)
                    if text.strip():
                        return text, f"pdfplumber (reconstructed words){' + pwd' if p else ''}"
            except Exception:
                pass

    # 2) PyMuPDF (fitz)
    if fitz is not None:
        for p in pwds:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf", password=(p if p else None))
                text = "\n".join(pg.get_text("text") for pg in doc)
                if text.strip():
                    return text, f"PyMuPDF{' + pwd' if p else ''}"
            except Exception:
                continue

    # 3) PyPDF2
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if getattr(reader, "is_encrypted", False):
                opened = False
                for p in pwds:
                    try:
                        res = reader.decrypt("" if p is None else p)
                        if res == 1:
                            opened = True
                            break
                    except Exception:
                        continue
                if not opened:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            if text.strip():
                return text, "PyPDF2"
        except Exception:
            pass

    return "", "none"

# =============================================================================
# Tickers
# =============================================================================
import re as _re
FII_EXACT = _re.compile(r"\b([A-Z]{4}11[A-Z]?)\b")
WITH_11   = _re.compile(r"\b([A-Z]{3,6}11[A-Z]?)\b")
SHARES    = _re.compile(r"\b([A-Z]{4}[3456][A-Z]?)\b")
BDR_2D    = _re.compile(r"\b([A-Z]{4}\d{2}[A-Z]?)\b")

def extract_ticker_from_text(text: str) -> str | None:
    t = strip_accents(text).upper()
    for rx in (FII_EXACT, WITH_11, SHARES, BDR_2D):
        m = rx.search(t)
        if m:
            return m.group(1)
    return None

# Heurística por nome (para quando o PDF não traz o ticker)
NAME2TICKER_PATTERNS = [
    (_re.compile(r"\bALLOS\b", _re.I), "ALSO3"),
    (_re.compile(r"\bAMBEV\b", _re.I), "ABEV3"),
    (_re.compile(r"\b(AREZZO|AZZAS\s*2154)\b", _re.I), "ARZZ3"),
    (_re.compile(r"\bMARCOPOLO\b", _re.I), "POMO3"),
    (_re.compile(r"\bMELNICK\b", _re.I), "MELK3"),
    (_re.compile(r"\bVULCABRAS\b", _re.I), "VULC3"),
]
def guess_ticker_from_name(spec: str) -> str | None:
    s = strip_accents(spec).upper()
    for rx, tkr in NAME2TICKER_PATTERNS:
        if rx.search(s):
            return tkr
    return None

def derive_from_on_pn(text: str) -> str | None:
    t = strip_accents(text).upper()
    m = _re.search(r"\b([A-Z]{3,6})\s+(ON|PN)\b", t)
    if not m:
        return None
    base, cls = m.group(1), m.group(2)
    # Observação: este é um último recurso; muitos nomes não batem com o código-base.
    return f"{base}3" if cls == "ON" else f"{base}4"

def clean_b3_tickers(lst) -> list:
    out = []
    for t in lst:
        if not isinstance(t, str):
            continue
        t = strip_accents(t).upper().strip()
        if not t:
            continue
        if _re.fullmatch(r"[A-Z]{3,6}\d{1,2}[A-Z]?", t):
            out.append(t)
    return sorted(set(out))

# =============================================================================
# Parsers & headers
# =============================================================================
def parse_trades_b3style(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    """
    Parser B3: robusto a PDF com senha (onde '@' pode sumir/mudar) e a linhas “grudadas”.
    Captura TODAS as ocorrências no texto inteiro.
    """
    text = (text or "").replace("\xa0", " ").replace("\u202f", " ")
    pat = _re.compile(
        r"1?-?BOVESPA\s+"
        r"(?P<cv>[CV])\s+VISTA\s+"
        r"(?P<spec>.+?)\s+"
        r"(?:[@#]\s*)?"                        # '@', '@#' ou ausente
        r"(?P<qty>\d{1,7})\s+"
        r"(?P<price>\d{1,3},\d{2})\s+"
        r"(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})\s+"
        r"(?P<dc>[CD])",
        flags=_re.IGNORECASE | _re.DOTALL
    )

    recs = []
    for m in pat.finditer(text):
        cv = m.group("cv").upper()
        spec = _re.sub(r"\s+", " ", m.group("spec")).strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))
        dc = m.group("dc").upper()

        ticker = extract_ticker_from_text(spec) or extract_ticker_from_text(m.group(0))
        if not ticker:
            key = _re.sub(r"[^A-Z]", "", strip_accents(spec).upper())
            ticker = name_to_ticker_map.get(key)
        if not ticker:
            ticker = guess_ticker_from_name(spec) or derive_from_on_pn(spec) or ""

        recs.append({
            "Ativo": ticker, "Nome": spec,
            "Operação": "Compra" if cv == "C" else "Venda",
            "Quantidade": qty, "Preço_Unitário": price,
            "Valor": value if cv == "C" else -value, "Sinal_DC": dc
        })

    return pd.DataFrame(recs)

def parse_trades_generic_table(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    text = (text or "").replace("\xa0", " ").replace("\u202f", " ")
    pat = _re.compile(
        r"(?P<cv>\b[CV]\b|\bCompra\b|\bVenda\b).*?"
        r"(?P<spec>.+?)\s+"
        r"(?:[@#]\s*)?(?P<qty>\d{1,7})\s+"
        r"(?P<price>\d{1,3},\d{2})\s+"
        r"(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})",
        flags=_re.IGNORECASE | _re.DOTALL
    )

    recs = []
    for m in pat.finditer(text):
        cv_raw = m.group("cv").strip().upper()
        cv = "C" if cv_raw.startswith("C") else "V"
        spec = _re.sub(r"\s+", " ", m.group("spec")).strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))

        ticker = extract_ticker_from_text(spec) or extract_ticker_from_text(m.group(0))
        if not ticker:
            key = _re.sub(r"[^A-Z]", "", strip_accents(spec).upper())
            ticker = name_to_ticker_map.get(key)
        if not ticker:
            ticker = guess_ticker_from_name(spec) or derive_from_on_pn(spec) or ""

        recs.append({
            "Ativo": ticker, "Nome": spec,
            "Operação": "Compra" if cv == "C" else "Venda",
            "Quantidade": qty, "Preço_Unitário": price,
            "Valor": value if cv == "C" else -value, "Sinal_DC": ""
        })
    return pd.DataFrame(recs)

def parse_trades_any(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    df = parse_trades_b3style(text, name_to_ticker_map)
    if df.empty:
        df = parse_trades_generic_table(text, name_to_ticker_map)
    return df

def detect_layout(text: str) -> str:
    t = strip_accents(text).lower()
    xp_hits = sum(k in t for k in ["data da consulta","data de referencia","conta xp","codigo assessor",
                                   "corretagem / despesas","ouvidoria:","xp investimentos cctvm","xpi.com.br"])
    b3_hits = sum(k in t for k in ["nota de negociacao","resumo dos negocios","total custos / despesas",
                                   "total bovespa / soma","cblc","liquido para","1-bovespa"])
    if xp_hits >= max(2, b3_hits + 1): return "XP"
    if b3_hits >= max(2, xp_hits + 1): return "B3"
    try:
        if not parse_trades_b3style(text, {}).empty: return "B3"
    except Exception: pass
    return "XP" if "conta xp" in t or "data da consulta" in t else "B3"

def parse_header_dates_and_net(text: str):
    lines = text.splitlines()
    norms = [strip_accents(l).lower() for l in lines]
    pairs = list(zip(lines, norms))
    data_pregao = None
    for raw, norm in pairs[:200]:
        if any(_re.search(p, norm) for p in [r"data\s*do\s*preg[aã]o", r"data\s*da\s*negocia[cç][aã]o", r"\bnegocia[cç][aã]o\b"]):
            md = _re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: data_pregao = md.group(1); break
    if not data_pregao:
        for raw, norm in pairs[:120]:
            if any(k in norm for k in ["consulta","referenc","liquido","l\u00edquido"]): continue
            if _re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", raw): continue
            md = _re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: data_pregao = md.group(1); break
    liquido_para_data = None; liquido_para_valor = None
    for raw, norm in pairs[::-1]:
        if "liquido para" in norm or "l\u00edquido para" in norm:
            md = _re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: liquido_para_data = md.group(1)
            vals = _re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2})", raw)
            if vals: liquido_para_valor = vals[-1]
            break
    return data_pregao, liquido_para_data, liquido_para_valor

# =============================================================================
# Custos
# =============================================================================
def parse_cost_components(text: str) -> dict:
    comp = {}
    atom = {
        "liquidacao": r"Taxa\s*de\s*liquida[cç][aã]o\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "emolumentos": r"Emolumentos\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "registro": r"Taxa\s*de\s*Registro\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "transf_ativos": r"Taxa\s*de\s*Transf\.\s*de\s*Ativos\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "corretagem": r"Corretag\w*\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "taxa_operacional": r"(?:Taxa|Tarifa)\s*Operacion\w+\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "iss": r"\bISS\b.*?(\d{1,3}(?:\.\d{3})*,\d{2})",
        "impostos": r"\bImpostos\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "outros": r"\bOutros\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
    }
    for k, p in atom.items():
        m = _re.search(p, text, flags=_re.IGNORECASE)
        if m: comp[k] = parse_brl_number(m.group(1))
    totals = {
        "total_bovespa_soma": r"Total\s*Bovespa\s*/\s*Soma\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "total_custos_despesas": r"Total\s*(?:Custos|Corretagem)\s*/\s*Despesas\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "taxas_b3": r"Taxas?\s*B3\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*,\d{2})",
    }
    for k, p in totals.items():
        m = _re.search(p, text, flags=_re.IGNORECASE)
        if m: comp[k] = parse_brl_number(m.group(1))
    m_irrf = _re.search(r"I\.?R\.?R\.?F\.?.*?(\d{1,3}(?:\.\d{3})*,\d{2})", text, flags=_re.IGNORECASE)
    if m_irrf: comp["_irrf"] = parse_brl_number(m_irrf.group(1))
    return comp

def compute_rateable_total(fees: dict) -> tuple[float, dict]:
    used = {}
    liq = fees.get("liquidacao", 0.0); used["liquidacao"] = liq
    reg = fees.get("registro", 0.0);   used["registro"] = reg
    tb = fees.get("total_bovespa_soma")
    if tb is None:
        tb = fees.get("emolumentos", 0.0) + fees.get("transf_ativos", 0.0)
        used["total_bovespa_soma_reconstr"] = tb
    else:
        used["total_bovespa_soma"] = tb
    tcd = fees.get("total_custos_despesas")
    if tcd is None:
        base = fees.get("corretagem", 0.0) or fees.get("taxa_operacional", 0.0)
        tcd = base + fees.get("iss", 0.0) + fees.get("impostos", 0.0) + fees.get("outros", 0.0)
        used["total_custos_despesas_reconstr"] = tcd
    else:
        used["total_custos_despesas"] = tcd
    total = round(liq + reg + tb + tcd, 2); used["__total_rateio"] = total
    return total, used

# =============================================================================
# Quotes (Yahoo)
# =============================================================================
def guess_yf_symbols_for_b3(ticker: str) -> list:
    return [f"{ticker.upper()}.SA", ticker.upper()]

def _pick_symbol_with_data(ticker: str) -> tuple[str|None, str]:
    if yf is None: return None, "yfinance ausente"
    last_err = ""
    for sym in guess_yf_symbols_for_b3(ticker):
        try:
            hd = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=False)
            if not hd.empty and "Close" in hd and not hd["Close"].dropna().empty:
                return sym, ""
        except Exception as e:
            last_err = str(e)
    return None, ("sem histórico diário" if not last_err else last_err)

@st.cache_data(show_spinner=False, ttl=25)
def fetch_quotes_yahoo_for_tickers(tickers: list, ref_date: datetime | None = None, _salt: int = 0) -> pd.DataFrame:
    cols = ["Ticker","Símbolo","Último","Último (quando)","Fechamento (pregão)","Pregão (data)","Motivo"]
    rows = []
    if yf is None or not tickers: return pd.DataFrame(columns=cols)
    for t in tickers:
        sym, note = _pick_symbol_with_data(t)
        last_px = last_dt = close_px = close_dt = None
        motivo = note; last_from_close = False
        if sym:
            for intr in ["1m","5m","15m"]:
                try:
                    h = yf.Ticker(sym).history(period="5d", interval=intr, auto_adjust=False)
                    if not h.empty and "Close" in h and not h["Close"].dropna().empty:
                        s = h["Close"].dropna()
                        last_px = float(s.iloc[-1]); idx = s.index[-1]
                        if getattr(idx,"tzinfo",None) is None: idx = pd.Timestamp(idx).tz_localize(timezone.utc)
                        last_dt = idx.tz_convert(TZ); break
                except Exception: pass
            try:
                if ref_date is None:
                    hd = yf.Ticker(sym).history(period="10d", interval="1d", auto_adjust=False)
                    if not hd.empty and "Close" in hd and not hd["Close"].dropna().empty:
                        s = hd["Close"].dropna()
                        close_px = float(s.iloc[-1]); idx = s.index[-1]
                        if getattr(idx,"tzinfo",None) is None: idx = pd.Timestamp(idx).tz_localize(timezone.utc)
                        close_dt = idx.tz_convert(TZ)
                else:
                    start = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
                    end   = (ref_date + timedelta(days=7)).strftime("%Y-%m-%d")
                    hd = yf.Ticker(sym).history(start=start, end=end, auto_adjust=False)
                    if not hd.empty and "Close" in hd and not hd["Close"].dropna().empty:
                        s = hd["Close"].dropna()
                        idxs = s.index[s.index.date <= ref_date.date()]
                        if len(idxs)>0: close_px = float(s.loc[idxs[-1]]); idx = idxs[-1]
                        else: close_px = float(s.iloc[-1]); idx = s.index[-1]
                        if getattr(idx,"tzinfo",None) is None: idx = pd.Timestamp(idx).tz_localize(timezone.utc)
                        close_dt = idx.tz_convert(TZ)
            except Exception: pass
            if last_px is None and close_px is not None:
                last_px, last_dt, last_from_close = close_px, close_dt, True
            if last_px is None and not motivo:
                motivo = "sem intraday/fechamento"
        rows.append({
            "Ticker": t, "Símbolo": sym or "",
            "Último": last_px,
            "Último (quando)": (_fmt_dt_local(last_dt)+(" (fechamento)" if last_from_close and last_dt else "")) if last_dt else "",
            "Fechamento (pregão)": close_px,
            "Pregão (data)": close_dt.date().strftime("%d/%m/%Y") if close_dt else "",
            "Motivo": motivo,
        })
    _ = _salt
    return pd.DataFrame(rows, columns=cols)

# =============================================================================
# DB (SQLite)
# =============================================================================
DB_PATH = Path("carteira.db")

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def db_init():
    conn = db_connect()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS ingestions (
        filehash TEXT PRIMARY KEY,
        filename TEXT,
        layout   TEXT,
        data_pregao TEXT,
        liquido_para_data TEXT,
        liquido_para_valor REAL,
        extractor TEXT,
        total_rateio REAL,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS trades_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filehash TEXT,
        data_pregao TEXT,
        ativo TEXT,
        operacao TEXT,
        quantidade INTEGER,
        preco_unit REAL,
        valor REAL,
        sinal_dc TEXT,
        nome TEXT,
        FOREIGN KEY(filehash) REFERENCES ingestions(filehash) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS trades_agg (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filehash TEXT,
        data_pregao TEXT,
        ativo TEXT,
        operacao TEXT,
        quantidade INTEGER,
        valor REAL,
        preco_medio REAL,
        custos REAL,
        total REAL,
        FOREIGN KEY(filehash) REFERENCES ingestions(filehash) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS fees_components (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filehash TEXT,
        key TEXT,
        value REAL,
        FOREIGN KEY(filehash) REFERENCES ingestions(filehash) ON DELETE CASCADE
    );
    -- Tokens do Gmail (um registro só)
    CREATE TABLE IF NOT EXISTS gmail_tokens (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        access_token TEXT,
        refresh_token TEXT,
        token_uri TEXT,
        client_id TEXT,
        client_secret TEXT,
        scopes TEXT,
        expiry TEXT
    );
    """)
    conn.commit()
    conn.close()

def db_already_ingested(filehash: str) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM ingestions WHERE filehash = ?", (filehash,))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def db_delete_ingestion_with_snapshot(filehash: str) -> dict | None:
    conn = db_connect(); cur = conn.cursor()
    snap = {"ing": None, "raw": [], "agg": [], "fees": []}
    cur.execute("SELECT * FROM ingestions WHERE filehash=?", (filehash,))
    row = cur.fetchone()
    if not row:
        conn.close(); return None
    snap["ing"] = row
    cur.execute("SELECT * FROM trades_raw WHERE filehash=?", (filehash,))
    snap["raw"] = cur.fetchall()
    cur.execute("SELECT * FROM trades_agg WHERE filehash=?", (filehash,))
    snap["agg"] = cur.fetchall()
    cur.execute("SELECT * FROM fees_components WHERE filehash=?", (filehash,))
    snap["fees"] = cur.fetchall()
    cur.execute("DELETE FROM ingestions WHERE filehash=?", (filehash,))
    conn.commit(); conn.close()
    return snap

def db_restore_ingestion_from_snapshot(snap: dict):
    if not snap or not snap.get("ing"):
        return False
    conn = db_connect(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO ingestions (filehash, filename, layout, data_pregao, liquido_para_data, liquido_para_valor, extractor, total_rateio, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, tuple(snap["ing"][0:9]))
    for r in snap.get("raw", []):
        cur.execute("""
            INSERT INTO trades_raw (id, filehash, data_pregao, ativo, operacao, quantidade, preco_unit, valor, sinal_dc, nome)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, r)
    for r in snap.get("agg", []):
        cur.execute("""
            INSERT INTO trades_agg (id, filehash, data_pregao, ativo, operacao, quantidade, valor, preco_medio, custos, total)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, r)
    for r in snap.get("fees", []):
        cur.execute("""
            INSERT INTO fees_components (id, filehash, key, value)
            VALUES (?,?,?,?)
        """, r)
    conn.commit(); conn.close()
    return True

def db_save_ingestion(res: dict, filehash: str, filename: str):
    conn = db_connect(); cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO ingestions
        (filehash, filename, layout, data_pregao, liquido_para_data, liquido_para_valor, extractor, total_rateio, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        filehash, filename, res.get("layout"), res.get("data_pregao"), res.get("liquido_para_data"),
        parse_brl_number(res["liquido_para_valor"]) if res.get("liquido_para_valor") else None,
        res.get("extractor"), res.get("total_rateio"), datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    ))
    fees = res.get("fees") or {}
    for k, v in fees.items():
        if k.startswith("_"): 
            continue
        cur.execute("INSERT INTO fees_components (filehash, key, value) VALUES (?,?,?)", (filehash, k, float(v)))
    df_raw = res.get("df_valid")
    if df_raw is not None and not df_raw.empty:
        for _, r in df_raw.iterrows():
            cur.execute("""
                INSERT INTO trades_raw (filehash, data_pregao, ativo, operacao, quantidade, preco_unit, valor, sinal_dc, nome)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                filehash, res.get("data_pregao"), str(r.get("Ativo") or ""), str(r.get("Operação") or ""),
                int(r.get("Quantidade") or 0), float(r.get("Preço_Unitário") or 0.0),
                float(r.get("Valor") or 0.0), str(r.get("Sinal_DC") or ""), str(r.get("Nome") or "")
            ))
    df_out = res.get("out")
    if df_out is not None and not df_out.empty:
        for _, r in df_out.iterrows():
            cur.execute("""
                INSERT INTO trades_agg (filehash, data_pregao, ativo, operacao, quantidade, valor, preco_medio, custos, total)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                filehash, r["Data do Pregão"], r["Ativo"], r["Operação"],
                int(r["Quantidade"]), float(r["Valor"]), 
                float(r["Preço Médio"]) if pd.notna(r["Preço Médio"]) else None,
                float(r["Custos"]), float(r["Total"])
            ))
    conn.commit(); conn.close()

def _ymd_from_br(d: str) -> str | None:
    try:
        return f"{d[6:10]}-{d[3:5]}-{d[0:2]}"
    except Exception:
        return None

def db_positions_dataframe(as_of: str | None = None, filtro_ativo: str | None = None) -> pd.DataFrame:
    """Posição e PM até a data (inclusive)."""
    conn = db_connect()
    where = " WHERE ativo <> '' "
    params = []
    if as_of:
        ymd = _ymd_from_br(as_of)
        where += " AND date(substr(data_pregao,7,4)||'-'||substr(data_pregao,4,2)||'-'||substr(data_pregao,1,2)) <= date(?) "
        params.append(ymd)
    if filtro_ativo:
        where += " AND UPPER(ativo) LIKE UPPER(?) "
        params.append(f"%{filtro_ativo}%")
    df = pd.read_sql_query(f"""
        SELECT id, data_pregao, ativo, operacao, quantidade, valor
        FROM trades_raw
        {where}
        ORDER BY date(substr(data_pregao,7,4)||'-'||substr(data_pregao,4,2)||'-'||substr(data_pregao,1,2)) ASC, id ASC
    """, conn, params=params)
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["Ativo","Quantidade","PM","Custo Atual"])
    pos = {}
    for _, r in df.iterrows():
        t = (r["ativo"] or "").upper().strip()
        q = int(r["quantidade"] or 0)
        gross = float(r["valor"] or 0.0)
        side = "C" if (gross > 0 or str(r["operacao"]).lower()=="compra") else "V"
        d = pos.get(t, {"qty":0, "avg":0.0})
        if side == "C":
            new_qty = d["qty"] + q
            new_avg = (d["avg"]*d["qty"] + abs(gross)) / new_qty if new_qty else 0.0
            d["qty"], d["avg"] = new_qty, new_avg
        else:
            d["qty"] = d["qty"] - q
            if d["qty"] <= 0:
                d["qty"], d["avg"] = 0, 0.0
        pos[t] = d
    rows = []
    for t, d in sorted(pos.items()):
        rows.append({"Ativo": t, "Quantidade": d["qty"], "PM": d["avg"], "Custo Atual": d["qty"]*d["avg"]})
    return pd.DataFrame(rows).sort_values("Ativo")

def db_rollup_net_total_by_ticker(as_of: str | None = None, filtro_ativo: str | None = None) -> pd.DataFrame:
    """TOTAL líquido por ticker: compras somam, vendas subtraem."""
    conn = db_connect()
    where = " WHERE ativo <> '' "
    params = []
    if as_of:
        ymd = _ymd_from_br(as_of)
        where += " AND date(substr(data_pregao,7,4)||'-'||substr(data_pregao,4,2)||'-'||substr(data_pregao,1,2)) <= date(?) "
        params.append(ymd)
    if filtro_ativo:
        where += " AND UPPER(ativo) LIKE UPPER(?) "
        params.append(f"%{filtro_ativo}%")
    df = pd.read_sql_query(f"""
        SELECT ativo AS Ativo,
               COALESCE(SUM(CASE WHEN operacao='Compra' THEN total ELSE -total END),0.0) AS Total
        FROM trades_agg
        {where}
        GROUP BY ativo
    """, conn, params=params)
    conn.close()
    return df

def db_movements_for_ticker(ticker: str, as_of: str | None = None) -> pd.DataFrame:
    conn = db_connect()
    where = " WHERE ativo = ? "
    params = [ticker]
    if as_of:
        ymd = _ymd_from_br(as_of)
        where += " AND date(substr(data_pregao,7,4)||'-'||substr(data_pregao,4,2)||'-'||substr(data_pregao,1,2)) <= date(?) "
        params.append(ymd)
    df = pd.read_sql_query(f"""
        SELECT data_pregao AS "Data do Pregão",
               operacao    AS "Operação",
               quantidade  AS "Quantidade",
               valor       AS "Valor",
               preco_medio AS "Preço Médio",
               custos      AS "Custos",
               total       AS "Total"
        FROM trades_agg
        {where}
        ORDER BY date(substr(data_pregao,7,4)||'-'||substr(data_pregao,4,2)||'-'||substr(data_pregao,1,2)) ASC,
                 CASE WHEN operacao='Compra' THEN 0 ELSE 1 END
    """, conn, params=params)
    conn.close()
    return df

# ============================== GMAIL INTEGRAÇÃO ===============================
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def _gmail_client_config_from_secrets() -> dict:
    conf = st.secrets.get("gmail", {})
    if not conf or not conf.get("client_id") or not conf.get("client_secret") or not conf.get("redirect_uri"):
        return {}
    return {
        "web": {
            "client_id": conf["client_id"],
            "client_secret": conf["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [conf["web"]["redirect_uris"][0] if isinstance(conf.get("redirect_uris"), list) else conf["redirect_uri"]],
        }
    }

def gmail_load_creds_from_db() -> Optional[Credentials]:
    conn = db_connect()
    row = conn.execute("SELECT access_token, refresh_token, token_uri, client_id, client_secret, scopes, expiry FROM gmail_tokens WHERE id=1").fetchone()
    conn.close()
    if not row:
        return None
    access_token, refresh_token, token_uri, client_id, client_secret, scopes, expiry = row
    if not refresh_token:
        return None
    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri=token_uri or "https://oauth2.googleapis.com/token",
        client_id=client_id or st.secrets["gmail"]["client_id"],
        client_secret=client_secret or st.secrets["gmail"]["client_secret"],
        scopes=(scopes.split() if scopes else GMAIL_SCOPES),
    )
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            gmail_save_creds_to_db(creds)
    except Exception:
        pass
    return creds

def gmail_save_creds_to_db(creds: Credentials):
    conn = db_connect()
    conn.execute("""
        INSERT INTO gmail_tokens (id, access_token, refresh_token, token_uri, client_id, client_secret, scopes, expiry)
        VALUES (1,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            access_token=excluded.access_token,
            refresh_token=excluded.refresh_token,
            token_uri=excluded.token_uri,
            client_id=excluded.client_id,
            client_secret=excluded.client_secret,
            scopes=excluded.scopes,
            expiry=excluded.expiry
    """, (
        creds.token or "",
        getattr(creds, "refresh_token", "") or "",
        getattr(creds, "token_uri", "https://oauth2.googleapis.com/token"),
        st.secrets["gmail"]["client_id"],
        st.secrets["gmail"]["client_secret"],
        " ".join(GMAIL_SCOPES),
        getattr(creds, "expiry", None).isoformat() if getattr(creds, "expiry", None) else None,
    ))
    conn.commit()
    conn.close()

def gmail_clear_tokens():
    conn = db_connect()
    conn.execute("DELETE FROM gmail_tokens WHERE id=1")
    conn.commit()
    conn.close()

def gmail_auth_url() -> Optional[str]:
    cfg = _gmail_client_config_from_secrets()
    if not cfg:
        return None
    flow = Flow.from_client_config(cfg, scopes=GMAIL_SCOPES, redirect_uri=cfg["web"]["redirect_uris"][0])
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["gmail_oauth_state"] = state
    return auth_url

def gmail_handle_oauth_callback():
    params = st.query_params
    code = params.get("code")
    if not code:
        return
    if isinstance(code, list):
        code = code[0]
    cfg = _gmail_client_config_from_secrets()
    if not cfg:
        st.warning("Config do Gmail ausente em st.secrets.")
        return
    flow = Flow.from_client_config(cfg, scopes=GMAIL_SCOPES, redirect_uri=cfg["web"]["redirect_uris"][0])
    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        gmail_save_creds_to_db(creds)
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.success("Gmail conectado com sucesso.")
    except Exception as e:
        st.error(f"Falha na troca de token: {e}")

def gmail_list_messages(service, query: str, max_results: int = 20):
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    return resp.get("messages", [])

def gmail_fetch_pdf_attachments(service, message_id: str):
    """Retorna lista [(filename, bytes)] apenas de PDFs do email message_id."""
    import base64

    def _b64url_to_bytes(s: str) -> bytes:
        if not s:
            return b""
        pad = "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode(s + pad)

    out = []
    msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    payload = msg.get("payload", {}) or {}

    def _maybe_add(part, default_name: str):
        if not isinstance(part, dict):
            return
        filename = (part.get("filename") or default_name).strip()
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body", {}) or {}
        att_id = body.get("attachmentId")
        if att_id and (mime == "application/pdf" or filename.lower().endswith(".pdf")):
            att = service.users().messages().attachments().get(
                userId="me", messageId=message_id, id=att_id
            ).execute()
            data = att.get("data")
            if data:
                out.append((filename or default_name, _b64url_to_bytes(data)))

    _maybe_add(payload, default_name=f"{message_id}.pdf")

    def _walk(parts):
        for p in parts or []:
            _maybe_add(p, default_name=f"{message_id}.pdf")
            if p.get("parts"):
                _walk(p.get("parts"))

    _walk(payload.get("parts"))
    return out

def gmail_import_notes():
    XP_NOTA_QUERY = 'from:no-reply@xpi.com.br subject:"XP Investimentos | Nota de Negociação" has:attachment filename:pdf newer_than:2y'

    st.markdown("### 📧 Importar do Gmail")
    cfg_ok = bool(_gmail_client_config_from_secrets())
    if not cfg_ok:
        st.info("Configure os segredos `[gmail]` em *Secrets* para habilitar o login.")
        return

    gmail_handle_oauth_callback()
    creds = gmail_load_creds_from_db()

    if not creds:
        url = gmail_auth_url()
        if url:
            st.link_button("Conectar ao Gmail", url)
        return

    st.caption(f"Filtro Gmail aplicado: `{XP_NOTA_QUERY}`")
    query = XP_NOTA_QUERY
    maxr = st.number_input("Máx. e-mails a buscar", min_value=1, max_value=200, value=20, step=1)

    try:
        service = build("gmail", "v1", credentials=creds)
    except Exception as e:
        st.error(f"Não foi possível iniciar o cliente do Gmail: {e}")
        return

    if st.button("🔎 Buscar e-mails"):
        try:
            msgs = gmail_list_messages(service, query=query, max_results=int(maxr))
            if not msgs:
                st.info("Nenhum e-mail encontrado com esse filtro.")
            else:
                st.success(f"Encontrados {len(msgs)} e-mail(s).")
                for m in msgs:
                    mid = m.get("id")
                    pdfs = gmail_fetch_pdf_attachments(service, mid)
                    if not pdfs:
                        continue
                    for fname, pdfb in pdfs:
                        fh = sha1(pdfb)
                        exists = db_already_ingested(fh)
                        cols = st.columns([3,1,1])
                        with cols[0]:
                            st.write(f"📎 **{fname}** — hash `{fh[:10]}...`")
                        with cols[1]:
                            st.write("Status:", "🟡 já no banco" if exists else "🟢 novo")
                        with cols[2]:
                            if st.button("➕ Ingerir", key=f"ing_{fh}", disabled=exists):
                                res = process_one_pdf(pdfb, default_map, _pwd_sig=_PWD_SIG)
                                if not res.get("ok"):
                                    st.error(res.get("error", "Falha no processamento"))
                                else:
                                    db_save_ingestion(res, fh, fname)
                                    st.success("Nota ingerida no banco.")
        except Exception as e:
            st.error(f"Erro ao buscar/baixar e-mails: {e}")

    with st.expander("Desconectar Gmail"):
        if st.button("Remover tokens salvos"):
            gmail_clear_tokens()
            st.success("Tokens removidos. Clique em Conectar para vincular novamente.")

# =============================================================================
# Processamento PDF
# =============================================================================
def allocate_with_roundfix(amounts: pd.Series, total_costs: float) -> pd.Series:
    if amounts.sum() <= 0 or total_costs <= 0:
        return pd.Series([0.0]*len(amounts), index=amounts.index, dtype=float)
    raw = amounts/amounts.sum()*total_costs
    floored = (raw*100).apply(math.floor)/100.0
    residual = round(total_costs - floored.sum(), 2)
    if abs(residual)>0:
        frac = (raw*100) - (raw*100).apply(math.floor)
        order = frac.sort_values(ascending=(residual<0)).index
        step = 0.01 if residual>0 else -0.01
        i=0
        while round(residual,2)!=0 and i<len(order):
            floored.loc[order[i]] += step
            residual = round(residual - step, 2); i+=1
    return floored

@st.cache_data(show_spinner=False)
def process_one_pdf(pdf_bytes: bytes, map_dict: dict, _pwd_sig: str = ""):
    text, extractor = extract_text_from_pdf(pdf_bytes, passwords=_PWD_LIST)
    if not text:
        return {"ok": False, "error": "Não consegui extrair texto do PDF (talvez senha incorreta?).", "extractor": extractor}
    layout = detect_layout(text)
    data_pregao_str, liquido_para_data_str, liquido_para_valor_str = parse_header_dates_and_net(text)
    df_trades = parse_trades_any(text, map_dict)
    if df_trades.empty:
        return {"ok": False, "error": "Não encontrei linhas de negociação.", "extractor": extractor, "layout": layout}

    rows = []
    for _, r in df_trades.iterrows():
        tkr = (r.get("Ativo") or "").strip().upper()
        if not tkr:
            guess = derive_from_on_pn(r.get("Nome",""))
            if guess: tkr = guess
        rows.append({**r, "Ativo": tkr})
    df_trades = pd.DataFrame(rows)
    df_valid = df_trades[df_trades["Ativo"].str.len()>0].copy()
    if df_valid.empty:
        return {"ok": False, "error": "Não consegui identificar tickers nas negociações.", "extractor": extractor, "layout": layout}

    df_valid["AbsValor"] = df_valid["Valor"].abs()
    agg = (
        df_valid.groupby(["Ativo","Operação"], as_index=False)
        .agg(Quantidade=("Quantidade","sum"), Valor=("Valor","sum"), BaseRateio=("AbsValor","sum"))
    )
    fees = parse_cost_components(text)
    total_detected, used_detail = compute_rateable_total(fees)

    alloc = allocate_with_roundfix(agg.set_index(["Ativo","Operação"])["BaseRateio"], total_detected)
    alloc_df = alloc.reset_index(); alloc_df.columns = ["Ativo","Operação","Custos"]
    out = agg.merge(alloc_df, on=["Ativo","Operação"], how="left")
    out["Custos"] = out["Custos"].fillna(0.0)
    out["Total"] = out.apply(lambda r: abs(r["Valor"]) - r["Custos"] if r["Valor"]<0 else abs(r["Valor"])+r["Custos"], axis=1)
    out["Preço Médio"] = out.apply(lambda r: (abs(r["Valor"])/r["Quantidade"]) if r["Quantidade"] else None, axis=1)
    out.insert(0, "Data do Pregão", data_pregao_str or "—")
    out = out.sort_values(["Data do Pregão","Ativo","Operação"]).reset_index(drop=True)

    return {
        "ok": True, "extractor": extractor, "layout": layout,
        "data_pregao": data_pregao_str, "liquido_para_data": liquido_para_data_str,
        "liquido_para_valor": liquido_para_valor_str, "fees": fees,
        "total_rateio": total_detected, "used_detail": used_detail,
        "df_valid": df_valid, "out": out, "text": text,
    }

# =============================================================================
# UI helpers (estilo centralizado)
# =============================================================================
def style_result_df(df: pd.DataFrame) -> Any:
    cols_emphasis = [c for c in ["Data do Pregão","Quantidade","Valor","Total"] if c in df.columns]
    sty = df.style.format({
        "Quantidade": "{:.0f}",
        "Valor": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Preço Médio": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Custos": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Total": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
    })
    def op_style(v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "compra": return "color:#16a34a; font-weight:700; text-align:center"
            if s == "venda":  return "color:#dc2626; font-weight:700; text-align:center"
        return "text-align:center"
    if "Operação" in df.columns:
        try:
            sty = sty.map(op_style, subset=pd.IndexSlice[:, ["Operação"]])
        except Exception:
            sty = sty.applymap(op_style, subset=["Operação"])
    sty = sty.set_properties(**{"text-align":"center"})
    sty = sty.set_table_styles([{"selector":"th", "props":[("text-align","center")]}], overwrite=False)
    if cols_emphasis:
        sty = sty.set_properties(
            subset=cols_emphasis,
            **{"border-left":"1px solid #e5e7eb","border-right":"1px solid #e5e7eb"}
        )
    sty = sty.set_properties(**{"border":"1px solid #e5e7eb"})
    return sty

def render_table(df: pd.DataFrame):
    try:
        html = style_result_df(df).to_html()
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.dataframe(df, width="stretch", hide_index=True)

def render_portfolio_interactive(df: pd.DataFrame, as_of_str: str):
    widths = [1.4, 1.0, 1.0, 1.0, 1.0, 1.2]
    headers = ["Ativo","Quantidade","PM","Cotação","Total","Patrimônio"]

    cols = st.columns(widths, vertical_alignment="center")
    for c, h in zip(cols, headers):
        c.markdown(f"<div style='text-align:center;font-weight:700'>{h}</div>", unsafe_allow_html=True)

    for i, r in df.iterrows():
        cols = st.columns(widths, vertical_alignment="center")
        tkr = str(r["Ativo"])
        qtd = r.get("Quantidade")
        pm  = r.get("PM")
        cot = r.get("Cotação")
        tot = r.get("Total")
        pat = r.get("Patrimônio")

        def _num(x, money=False):
            if pd.isna(x): return ""
            return f"R$ {brl(float(x))}" if money else f"{int(x):d}"

        with cols[0]:
            left, mid, right = st.columns([1, 2, 1])
            with mid:
                if tkr == "TOTAL":
                    st.markdown("<div style='text-align:center;font-weight:700'>TOTAL</div>", unsafe_allow_html=True)
                else:
                    if hasattr(st, "popover"):
                        with st.popover(tkr, use_container_width=True):
                            df_mov = db_movements_for_ticker(tkr, as_of=as_of_str)
                            if df_mov.empty:
                                st.info("Sem movimentações para este ticker no período.")
                            else:
                                render_table(df_mov[["Data do Pregão","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
                    else:
                        if st.button(tkr, key=f"btn_{tkr}_{i}", use_container_width=True):
                            st.session_state["open_modal_tkr"] = tkr

        cols[1].markdown(f"<div style='text-align:center'>{_num(qtd)}</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='text-align:center'>{_num(pm, money=True)}</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='text-align:center'>{_num(cot, money=True)}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div style='text-align:center'>{_num(tot, money=True)}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div style='text-align:center'>{_num(pat, money=True)}</div>", unsafe_allow_html=True)

    if not hasattr(st, "popover") and st.session_state.get("open_modal_tkr"):
        tkr = st.session_state["open_modal_tkr"]
        if hasattr(st, "modal"):
            with st.modal(f"Movimentações — {tkr}"):
                df_mov = db_movements_for_ticker(tkr, as_of=as_of_str)
                if df_mov.empty:
                    st.info("Sem movimentações para este ticker no período.")
                else:
                    render_table(df_mov[["Data do Pregão","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
                if st.button("Fechar"):
                    st.session_state.pop("open_modal_tkr", None)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### Movimentações — {tkr}")
            df_mov = db_movements_for_ticker(tkr, as_of=as_of_str)
            if df_mov.empty:
                st.info("Sem movimentações para este ticker no período.")
            else:
                render_table(df_mov[["Data do Pregão","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
            if st.button("Fechar"):
                st.session_state.pop("open_modal_tkr", None)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Diagnóstico (opcional) ----------------------
def debug_probe_pdf_passwords(pdf_bytes: bytes) -> pd.DataFrame:
    cols = ["Engine", "Encrypted?", "Pwd idx", "Opened", "TextLen", "Note"]
    rows = []
    pwds = [None] + _collect_pdf_passwords_from_secrets()

    if fitz:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            enc = bool(doc.is_encrypted)
            opened = not enc
            used_idx = None
            note = ""
            if enc:
                for i, p in enumerate(pwds):
                    try:
                        if p and doc.authenticate(p):
                            opened = True
                            used_idx = i
                            break
                    except Exception:
                        continue
                if not opened:
                    note = "fitz não autenticou com nenhuma senha"
            txt = ""
            if opened:
                try:
                    txt = "\n".join(page.get_text("text") for page in doc)
                except Exception as e:
                    note = f"fitz get_text falhou: {e}"
            rows.append(["PyMuPDF", enc, used_idx, opened, len(txt), note])
        except Exception as e:
            rows.append(["PyMuPDF", "-", "-", False, 0, f"erro abrir: {e}"])

    if pdfplumber:
        ok = False
        for i, p in enumerate(pwds):
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes), password=p) as pdf:
                    txt = "".join(((page.extract_text() or "") + "\n") for page in pdf.pages)
                    if txt.strip():
                        rows.append(["pdfplumber", "?", i, True, len(txt), ""])
                        ok = True
                        break
            except Exception:
                continue
        if not ok:
            rows.append(["pdfplumber", "?", "-", False, 0, "não abriu ou sem texto"])

    if PyPDF2:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            enc = getattr(reader, "is_encrypted", False)
            ok = not enc
            used_idx = None
            if enc:
                for i, p in enumerate(pwds):
                    try:
                        res = reader.decrypt("" if p is None else p)
                        if res == 1:
                            ok = True
                            used_idx = i
                            break
                    except Exception:
                        continue
            txt = ""
            if ok:
                for pg in reader.pages:
                    txt += (pg.extract_text() or "") + "\n"
            rows.append(["PyPDF2", enc, used_idx, ok, len(txt), ""])
        except Exception as e:
            rows.append(["PyPDF2", "-", "-", False, 0, f"erro abrir: {e}"])

    return pd.DataFrame(rows, columns=cols)

# =============================================================================
# APP
# =============================================================================
st.set_page_config(page_title="Calc B3 - Nota de Corretagem", layout="wide")
db_init()

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
.subtitle { color:#6c757d; margin-bottom: 1rem; }
.card { padding:1rem 1.25rem; border:1px solid #e9ecef; border-radius:14px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.muted { color:#6c757d; font-size:.9rem; }
.section-title { font-weight:700; font-size:1.1rem; margin-bottom:.5rem; }
.badge { display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.85rem; font-weight:700; border:1px solid transparent; }
.badge-b3 { background:#E7F5FF; color:#095BC6; border-color:#B3D7FF; }
.badge-xp { background:#FFF3BF; color:#8B6A00; border-color:#FFD875; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Calc B3 – Nota de Corretagem</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Consolidado de notas B3/XP, carteira e movimentações com PM, patrimônio e cotações em tempo quase real.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Opções")
    if st.button("🔄 Limpar cache e recarregar", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.subheader("Banco de dados")
    try:
        st.caption(f"Arquivo DB: `{Path('carteira.db').resolve()}`")
    except Exception:
        st.caption("Arquivo DB: carteira.db")
    if Path("carteira.db").exists():
        st.download_button("⬇️ Exportar DB (carteira.db)", data=Path("carteira.db").read_bytes(),
                           file_name="carteira.db", mime="application/octet-stream", use_container_width=True)
    db_up = st.file_uploader("⬆️ Importar/Substituir DB (.db)", type=["db","sqlite"], key="db_upload")
    if db_up and st.button("Substituir banco atual", use_container_width=True):
        try:
            Path("carteira.db").write_bytes(db_up.read())
            st.success("Banco substituído com sucesso.")
            st.cache_data.clear(); st.rerun()
        except Exception as e:
            st.error(f"Falha ao substituir o banco: {e}")

    st.markdown("---")
    st.subheader("Mapeamento opcional Nome→Ticker")
    st.write("Forneça um CSV `Nome,Ticker` se sua nota vier sem ticker.")
    map_file = st.file_uploader("Upload CSV de mapeamento (opcional)", type=["csv"], key="map_csv")

    st.markdown("---")
    gmail_import_notes()

default_map = {"EVEN":"EVEN3","PETRORECSA":"RECV3","VULCABRAS":"VULC3"}  # opcional, continua valendo
if map_file is not None:
    try:
        mdf = pd.read_csv(map_file)
        for _, row in mdf.iterrows():
            if pd.notna(row.get("Nome")) and pd.notna(row.get("Ticker")):
                k = re.sub(r"[^A-Z]","", strip_accents(str(row["Nome"])).upper())
                default_map[k] = str(row["Ticker"]).strip().upper()
    except Exception as e:
        st.warning(f"Falha ao ler CSV: {e}")

# ========================= Uploads & processamento ============================
uploads = st.file_uploader("Carregue um ou mais PDFs da B3/XP", type=["pdf"], accept_multiple_files=True, key="pdfs")
results = []
if uploads:
    for f in uploads:
        try:
            pdf_bytes = f.read()
            res = process_one_pdf(pdf_bytes, default_map, _pwd_sig=_PWD_SIG)
            res["filename"] = f.name
            res["filehash"] = sha1(pdf_bytes)
        except Exception as e:
            res = {"ok": False, "error": str(e), "filename": f.name, "filehash": "NA"}
        results.append(res)

# ---------------- Diagnóstico PDF c/ senha (opcional) ----------------
with st.expander("🔐 Diagnóstico PDF com senha"):
    diag_file = st.file_uploader("Carregue um PDF só para teste", type=["pdf"], key="diag_pdf")
    st.caption(f"Senhas carregadas dos Secrets: {len(_PWD_LIST)} (não exibidas)")
    if diag_file and st.button("Testar PDF"):
        try:
            df_diag = debug_probe_pdf_passwords(diag_file.read())
            if df_diag.empty:
                st.info("Nada para mostrar.")
            else:
                st.dataframe(df_diag, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Falha no diagnóstico: {e}")

# ================================ Consolidado =================================
st.markdown("## 📚 Consolidado")
if not results:
    st.info("Envie um ou mais arquivos PDF para ver o consolidado (ou use o importador Gmail na barra lateral).")
else:
    valid_outs = [r["out"] for r in results if r.get("ok")]
    if not valid_outs:
        st.warning("Nenhuma nota válida processada.")
    else:
        all_out = pd.concat(valid_outs, ignore_index=True)
        colf1, colf2, colf3 = st.columns([1,1,2])
        with colf1:
            fil_compra = st.checkbox("Compras", True)
            fil_venda  = st.checkbox("Vendas", True)
        with colf2:
            apenas_fiis   = st.checkbox("Só FIIs", False)
            apenas_acoes  = st.checkbox("Só Ações", False)
        with colf3:
            filtro_ticker = st.text_input("Filtro por ticker (contém)", "")

        dfc = all_out.copy()
        if not fil_compra: dfc = dfc[dfc["Operação"] != "Compra"]
        if not fil_venda:  dfc = dfc[dfc["Operação"] != "Venda"]
        if apenas_fiis and not apenas_acoes:
            dfc = dfc[dfc["Ativo"].str.contains(r"11[A-Z]?$", regex=True, na=False)]
        if apenas_acoes and not apenas_fiis:
            dfc = dfc[dfc["Ativo"].str.contains(r"[3456][A-Z]?$", regex=True, na=False)]
        if filtro_ticker:
            dfc = dfc[dfc["Ativo"].str.contains(filtro_ticker, case=False, na=False)]
        dfc["Preço Médio"] = dfc.apply(lambda r: (abs(r["Valor"])/r["Quantidade"]) if r["Quantidade"] else None, axis=1)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_table(dfc[["Data do Pregão","Ativo","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
        csv_cons = dfc.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar consolidado", data=csv_cons, file_name="resultado_consolidado.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### ➕ Adicionar PDFs carregados ao banco")
        allow_dups = st.checkbox("Permitir redundância de notas", value=False)
        if st.button("➕ Adicionar todas as notas visíveis"):
            inserted = 0; duplicated = 0
            for r in results:
                if not r.get("ok"): 
                    continue
                fh = r["filehash"]
                if db_already_ingested(fh) and not allow_dups:
                    duplicated += 1
                    continue
                if db_already_ingested(fh) and allow_dups:
                    fh = f"{fh}:dup:{int(_now())}"
                db_save_ingestion(r, fh, r["filename"])
                inserted += 1
            if inserted:
                st.success(f"{inserted} nota(s) adicionada(s) ao banco.")
            if duplicated and not allow_dups:
                st.warning(f"{duplicated} nota(s) já existiam e foram ignoradas.")

# ================================ Banco / Carteira ============================
st.markdown("## 📦 Carteira")

colfL, colfR = st.columns([2,2])
with colfL:
    filtro_txt = st.text_input("Filtro por ativo (contém)", "")
with colfR:
    data_ate = st.date_input("Data de corte (até)", value=datetime.now(TZ).date())
data_ate_str = datetime.strftime(datetime.combine(data_ate, datetime.min.time()), "%d/%m/%Y")

df_pos = db_positions_dataframe(as_of=data_ate_str, filtro_ativo=filtro_txt)
df_tot = db_rollup_net_total_by_ticker(as_of=data_ate_str, filtro_ativo=filtro_txt)
df_master = pd.merge(df_pos, df_tot, on="Ativo", how="outer").fillna({"Quantidade":0,"PM":0.0,"Custo Atual":0.0,"Total":0.0})
df_master = df_master.sort_values("Ativo")

tickers_all = clean_b3_tickers(df_master["Ativo"].tolist())
salt = int(_now() // 30)
quotes_df = fetch_quotes_yahoo_for_tickers(tickers_all, _salt=int(salt)) if tickers_all else pd.DataFrame()
last_map = {r["Ticker"]: r["Último"] for _, r in quotes_df.iterrows()} if not quotes_df.empty else {}

df_master["Cotação"] = df_master["Ativo"].map(last_map)
df_master["Patrimônio"] = df_master.apply(lambda r: (r["Quantidade"] * r["Cotação"]) if (pd.notna(r["Cotação"]) and r["Quantidade"]>0) else 0.0, axis=1)

total_row = {
    "Ativo":"TOTAL",
    "Quantidade": df_master["Quantidade"].sum(skipna=True),
    "PM": None,
    "Custo Atual": df_master["Custo Atual"].sum(skipna=True),
    "Total": None,
    "Cotação": None,
    "Patrimônio": df_master["Patrimônio"].sum(skipna=True),
}
df_master_tot = pd.concat([df_master, pd.DataFrame([total_row])], ignore_index=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Ativos")
render_portfolio_interactive(df_master_tot[["Ativo","Quantidade","PM","Cotação","Total","Patrimônio"]], as_of_str=data_ate_str)
st.caption("• Total = compras − vendas (líquido).  • Patrimônio = Quantidade × Cotação (0 quando posição zerada).")
st.markdown('</div>', unsafe_allow_html=True)

# ================================ Gerenciar banco ============================
st.markdown("### ⚙️ Gerenciar carteira")
with st.expander("🗑️ Excluir/estornar uma ingestão (com undo)"):
    conn = db_connect()
    df_ing = pd.read_sql_query("SELECT filehash, filename, layout, data_pregao, created_at FROM ingestions ORDER BY created_at DESC", conn)
    conn.close()
    if df_ing.empty:
        st.info("Não há ingestões salvas.")
    else:
        st.dataframe(df_ing, width="stretch", hide_index=True)
        fh_sel = st.selectbox("Selecione o filehash para excluir", df_ing["filehash"].tolist())
        if st.button("Excluir ingestão selecionada"):
            snap = db_delete_ingestion_with_snapshot(fh_sel)
            if snap:
                st.session_state["last_deleted_snap"] = snap
                st.success("Ingestão excluída. Você pode desfazer abaixo.")
            else:
                st.warning("Ingestão não encontrada.")
        if st.button("↩️ Desfazer última exclusão"):
            snap = st.session_state.get("last_deleted_snap")
            if snap:
                ok = db_restore_ingestion_from_snapshot(snap)
                if ok:
                    st.success("Exclusão desfeita (ingestão restaurada).")
                else:
                    st.error("Falha ao restaurar.")
            else:
                st.info("Nada para desfazer.")
