# calcB3.py
# App Streamlit: múltiplos PDFs B3/XP, rateio por valor, PM, cotações (Yahoo),
# Banco/Carteira com posições (Ativos), Movimentações do ticker abaixo,
# TOTAL de Custos e Patrimônio, filtros, exclusão/undo e edição manual de lançamentos.
# As notas individuais ficam escondidas em um expander (clicáveis).

import io
import os
import re
import math
import unicodedata
import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from time import time as _now

import pandas as pd
import streamlit as st

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
try:
    import requests
except Exception:
    requests = None

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
# PDF → texto
# =============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    # 1) pdfplumber strict
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text(x_tolerance=1, y_tolerance=1) or "") + "\n"
                if text.strip():
                    return text, "pdfplumber (strict tol=1/1)"
        except Exception:
            pass
        # 1b) default
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
                if text.strip():
                    return text, "pdfplumber (default tol)"
        except Exception:
            pass
        # 1c) reconstructed words
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
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
                    return text, "pdfplumber (reconstructed words)"
        except Exception:
            pass
    # 2) PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            if text.strip():
                return text, "PyMuPDF (fitz)"
        except Exception:
            pass
    # 3) PyPDF2
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if getattr(reader, "is_encrypted", False):
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
FII_EXACT = re.compile(r"\b([A-Z]{4}11[A-Z]?)\b")
WITH_11   = re.compile(r"\b([A-Z]{3,6}11[A-Z]?)\b")
SHARES    = re.compile(r"\b([A-Z]{4}[3456][A-Z]?)\b")
BDR_2D    = re.compile(r"\b([A-Z]{4}\d{2}[A-Z]?)\b")

def extract_ticker_from_text(text: str) -> str | None:
    t = strip_accents(text).upper()
    for rx in (FII_EXACT, WITH_11, SHARES, BDR_2D):
        m = rx.search(t)
        if m:
            return m.group(1)
    return None

def derive_from_on_pn(text: str) -> str | None:
    t = strip_accents(text).upper()
    m = re.search(r"\b([A-Z]{3,6})\s+(ON|PN)\b", t)
    if not m:
        return None
    base, cls = m.group(1), m.group(2)
    return f"{base}3" if cls == "ON" else f"{base}4"

def clean_b3_tickers(lst) -> list:
    out = []
    for t in lst:
        if not isinstance(t, str): 
            continue
        t = strip_accents(t).upper().strip()
        if not t:
            continue
        if re.fullmatch(r"[A-Z]{3,6}\d{1,2}[A-Z]?", t):
            out.append(t)
    return sorted(set(out))

# =============================================================================
# Parsers & headers
# =============================================================================
def parse_trades_b3style(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    lines = text.splitlines()
    trade_lines = [l for l in lines if ("BOVESPA" in l and "VISTA" in l and "@" in l)]
    pat = re.compile(
        r"BOVESPA\s+(?P<cv>[CV])\s+VISTA\s+(?P<spec>.+?)@\s+(?P<qty>\d+)\s+(?P<price>\d+,\d+)\s+(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})\s+(?P<dc>[CD])"
    )
    recs = []
    for line in trade_lines:
        m = pat.search(line)
        if not m:
            continue
        cv = m.group("cv")
        spec = re.sub(r"\s+", " ", m.group("spec")).strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))
        dc = m.group("dc")
        ticker = extract_ticker_from_text(spec) or extract_ticker_from_text(line)
        if not ticker:
            key = re.sub(r"[^A-Z]", "", strip_accents(spec).upper())
            ticker = name_to_ticker_map.get(key)
        if not ticker:
            ticker = derive_from_on_pn(spec) or ""
        recs.append({
            "Ativo": ticker, "Nome": spec,
            "Operação": "Compra" if cv == "C" else "Venda",
            "Quantidade": qty, "Preço_Unitário": price,
            "Valor": value if cv == "C" else -value, "Sinal_DC": dc
        })
    return pd.DataFrame(recs)

def parse_trades_generic_table(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if "@" in l and re.search(r"\d{1,3}(?:\.\d{3})*,\d{2}", l)]
    pat = re.compile(
        r"(?P<cv>\b[CV]\b|\bCompra\b|\bVenda\b).*?(?P<spec>.+?)@\s+(?P<qty>\d+)\s+(?P<price>\d+,\d+)\s+(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})"
    )
    recs = []
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        cv_raw = m.group("cv").strip().upper()
        cv = "C" if cv_raw.startswith("C") else "V"
        spec = re.sub(r"\s+", " ", m.group("spec")).strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))
        ticker = extract_ticker_from_text(spec) or extract_ticker_from_text(line)
        if not ticker:
            key = re.sub(r"[^A-Z]", "", strip_accents(spec).upper())
            ticker = name_to_ticker_map.get(key)
        if not ticker:
            ticker = derive_from_on_pn(spec) or ""
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
        if any(re.search(p, norm) for p in [r"data\s*do\s*preg[aã]o", r"data\s*da\s*negocia[cç][aã]o", r"\bnegocia[cç][aã]o\b"]):
            md = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: data_pregao = md.group(1); break
    if not data_pregao:
        for raw, norm in pairs[:120]:
            if any(k in norm for k in ["consulta","referenc","liquido","l\u00edquido"]): continue
            if re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", raw): continue
            md = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: data_pregao = md.group(1); break
    liquido_para_data = None; liquido_para_valor = None
    for raw, norm in pairs[::-1]:
        if "liquido para" in norm or "l\u00edquido para" in norm:
            md = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if md: liquido_para_data = md.group(1)
            vals = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2})", raw)
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
        m = re.search(p, text, flags=re.IGNORECASE)
        if m: comp[k] = parse_brl_number(m.group(1))
    totals = {
        "total_bovespa_soma": r"Total\s*Bovespa\s*/\s*Soma\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "total_custos_despesas": r"Total\s*(?:Custos|Corretagem)\s*/\s*Despesas\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "taxas_b3": r"Taxas?\s*B3\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*,\d{2})",
    }
    for k, p in totals.items():
        m = re.search(p, text, flags=re.IGNORECASE)
        if m: comp[k] = parse_brl_number(m.group(1))
    m_irrf = re.search(r"I\.?R\.?R\.?F\.?.*?(\d{1,3}(?:\.\d{3})*,\d{2})", text, flags=re.IGNORECASE)
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
# Quotes (Yahoo + Google fallback)
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

@st.cache_data(show_spinner=False, ttl=60)
def fetch_quotes_yahoo_for_tickers(tickers: list, ref_date: datetime | None = None) -> pd.DataFrame:
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
    """Deleta uma ingestão e retorna snapshot para possível undo."""
    conn = db_connect(); cur = conn.cursor()
    # snapshot
    snap = {"ing": None, "raw": [], "agg": [], "fees": []}
    cur.execute("SELECT * FROM ingestions WHERE filehash=?", (filehash,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    snap["ing"] = row
    cur.execute("SELECT * FROM trades_raw WHERE filehash=?", (filehash,))
    snap["raw"] = cur.fetchall()
    cur.execute("SELECT * FROM trades_agg WHERE filehash=?", (filehash,))
    snap["agg"] = cur.fetchall()
    cur.execute("SELECT * FROM fees_components WHERE filehash=?", (filehash,))
    snap["fees"] = cur.fetchall()
    # delete (cascade on trades_*)
    cur.execute("DELETE FROM ingestions WHERE filehash=?", (filehash,))
    conn.commit(); conn.close()
    return snap

def db_restore_ingestion_from_snapshot(snap: dict):
    if not snap or not snap.get("ing"):
        return False
    conn = db_connect(); cur = conn.cursor()
    # columns layout as created above:
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

# ---- Views com data de corte e filtro de ativo
def _ymd_from_br(d: str) -> str:
    try:
        # dd/mm/yyyy -> yyyy-mm-dd
        return f"{d[6:10]}-{d[3:5]}-{d[0:2]}"
    except Exception:
        return None

def db_positions_dataframe(as_of: str | None = None, filtro_ativo: str | None = None) -> pd.DataFrame:
    """Calcula posição e PM até a data (inclusive). 'as_of' em dd/mm/aaaa."""
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
                # zera posição; PM zera (mantemos PM por movimento, não na posição zerada)
                d["qty"], d["avg"] = 0, 0.0
        pos[t] = d
    rows = []
    for t, d in sorted(pos.items()):
        rows.append({"Ativo": t, "Quantidade": d["qty"], "PM": d["avg"], "Custo Atual": d["qty"]*d["avg"]})
    return pd.DataFrame(rows).sort_values("Ativo")

def db_rollup_costs_total_by_ticker(as_of: str | None = None, filtro_ativo: str | None = None) -> pd.DataFrame:
    """Somatórios de Custos e Total até a data (inclusive)."""
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
               COALESCE(SUM(custos),0.0) AS Custos,
               COALESCE(SUM(total),0.0)  AS Total
        FROM trades_agg
        {where}
        GROUP BY ativo
    """, conn, params=params)
    conn.close()
    return df

def db_movements_for_ticker(ticker: str, as_of: str | None = None) -> pd.DataFrame:
    """Movimentações agregadas até data para um ticker; mantém PM também nas vendas."""
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

# =============================================================================
# Processamento de uma nota (uploads)
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
def process_one_pdf(pdf_bytes: bytes, map_dict: dict):
    text, extractor = extract_text_from_pdf(pdf_bytes)
    if not text:
        return {"ok": False, "error": "Não consegui extrair texto do PDF.", "extractor": extractor}
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
# UI helpers
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
            if s == "compra": return "color:#16a34a; font-weight:700"
            if s == "venda":  return "color:#dc2626; font-weight:700"
        return ""
    if "Operação" in df.columns:
        sty = sty.applymap(op_style, subset=["Operação"])
    if cols_emphasis:
        sty = sty.set_properties(
            subset=cols_emphasis,
            **{"border-left":"1px solid #e5e7eb","border-right":"1px solid #e5e7eb","background-color":"#f6f7f9","font-weight":"700"}
        )
        table_styles = []
        for c in cols_emphasis:
            idx = df.columns.get_loc(c)
            table_styles.append({
                "selector": f"th.col_heading.level0.col{idx}",
                "props": [("border","1px solid #e5e7eb"),("background-color","#f6f7f9"),("font-weight","700")]
            })
        sty = sty.set_table_styles(table_styles, overwrite=False)
    sty = sty.set_properties(**{"border":"1px solid #e5e7eb"})
    return sty

def render_result_table(df: pd.DataFrame):
    try:
        html = style_result_df(df).to_html()
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)

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
st.markdown('<div class="subtitle">Consolidado de notas B3/XP, carteira e movimentações com PM, custos, patrimônio e cotações.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Opções")
    if st.button("🔄 Recarregar/limpar cache", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.subheader("Banco de dados")
    try:
        st.caption(f"Arquivo DB: `{Path('carteira.db').resolve()}`")
    except Exception:
        st.caption("Arquivo DB: carteira.db")
    # Export
    if Path("carteira.db").exists():
        st.download_button("⬇️ Exportar DB (carteira.db)", data=Path("carteira.db").read_bytes(),
                           file_name="carteira.db", mime="application/octet-stream", use_container_width=True)
    # Import
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
    st.write("Se a nota usa **nome da empresa** em vez do ticker, forneça um CSV `Nome,Ticker`.")
    map_file = st.file_uploader("Upload CSV de mapeamento (opcional)", type=["csv"], key="map_csv")

default_map = {"EVEN":"EVEN3","PETRORECSA":"RECV3","VULCABRAS":"VULC3"}
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
            res = process_one_pdf(pdf_bytes, default_map)
            res["filename"] = f.name
            res["filehash"] = sha1(pdf_bytes)
        except Exception as e:
            res = {"ok": False, "error": str(e), "filename": f.name, "filehash": "NA"}
        results.append(res)

# ================================ Consolidado =================================
st.markdown("## 📚 Consolidado")
if not results:
    st.info("Envie um ou mais arquivos PDF para ver o consolidado.")
else:
    valid_outs = [r["out"] for r in results if r.get("ok")]
    if not valid_outs:
        st.warning("Nenhuma nota válida processada.")
    else:
        all_out = pd.concat(valid_outs, ignore_index=True)
        # Filtros rápidos: Compra/Venda, FIIs x Ações
        colf1, colf2, colf3 = st.columns([1,1,2])
        with colf1:
            fil_compra = st.checkbox("Somente Compras", True)
            fil_venda  = st.checkbox("Somente Vendas", True)
        with colf2:
            apenas_fiis   = st.checkbox("Somente FIIs", False)
            apenas_acoes  = st.checkbox("Somente Ações", False)
        with colf3:
            filtro_ticker = st.text_input("Filtro por ticker (contém)", "")

        dfc = all_out.copy()
        if not fil_compra:
            dfc = dfc[dfc["Operação"] != "Compra"]
        if not fil_venda:
            dfc = dfc[dfc["Operação"] != "Venda"]
        if apenas_fiis and not apenas_acoes:
            dfc = dfc[dfc["Ativo"].str.contains(r"11[A-Z]?$", regex=True, na=False)]
        if apenas_acoes and not apenas_fiis:
            dfc = dfc[dfc["Ativo"].str.contains(r"[3456][A-Z]?$", regex=True, na=False)]
        if filtro_ticker:
            dfc = dfc[dfc["Ativo"].str.contains(filtro_ticker, case=False, na=False)]
        # recomputa PM da linha filtrada (defensivo)
        dfc["Preço Médio"] = dfc.apply(lambda r: (abs(r["Valor"])/r["Quantidade"]) if r["Quantidade"] else None, axis=1)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_result_table(dfc[["Data do Pregão","Ativo","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
        csv_cons = dfc.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar consolidado", data=csv_cons, file_name="resultado_consolidado.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        # ➕ Adicionar ao banco (todas ou selecionar)
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

# Notas individuais escondidas (opcional)
if results:
    with st.expander("🗂️ Notas individuais (opcional)"):
        for r in results:
            st.markdown(f"**{r.get('filename','Arquivo')}** — hash `{r.get('filehash')}`")
            if not r.get("ok"):
                st.error(r.get("error","Erro ao processar")); 
                continue
            layout = r["layout"]; _badge_cls = "badge-xp" if layout=="XP" else "badge-b3"
            st.markdown(f'<span class="badge {_badge_cls}">{layout}</span> • Extrator: {r["extractor"]}', unsafe_allow_html=True)
            render_result_table(r["out"][["Data do Pregão","Ativo","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
            colb1, colb2 = st.columns([1,4])
            with colb1:
                if st.button(f"➕ Adicionar esta nota", key=f"add_{r['filehash']}"):
                    fh = r["filehash"]
                    if db_already_ingested(fh):
                        st.warning("Esta nota já existe no banco (mesmo filehash). Clique novamente para salvar duplicado.")
                        if st.button("✅ Salvar duplicado mesmo assim", key=f"dup_{r['filehash']}"):
                            db_save_ingestion(r, f"{fh}:dup:{int(_now())}", r["filename"])
                            st.success("Duplicado salvo.")
                    else:
                        db_save_ingestion(r, fh, r["filename"])
                        st.success("Salvo no banco.")

# ================================ Banco / Carteira ============================
st.markdown("## 📦 Carteira")

# Filtros (DB)
colfL, colfR = st.columns([2,2])
with colfL:
    filtro_txt = st.text_input("Filtro por ativo (contém)", "")
with colfR:
    data_ate = st.date_input("Data de corte (até)", value=datetime.now(TZ).date())
data_ate_str = datetime.strftime(datetime.combine(data_ate, datetime.min.time()), "%d/%m/%Y")

# Tabela Ativos (posição + cotação + patrimônio) + linha TOTAL
df_pos = db_positions_dataframe(as_of=data_ate_str, filtro_ativo=filtro_txt)
df_ct  = db_rollup_costs_total_by_ticker(as_of=data_ate_str, filtro_ativo=filtro_txt)
df_master = pd.merge(df_pos, df_ct, on="Ativo", how="outer").fillna({"Quantidade":0,"PM":0.0,"Custo Atual":0.0,"Custos":0.0,"Total":0.0})
df_master = df_master.sort_values("Ativo")

tickers_all = clean_b3_tickers(df_master["Ativo"].tolist())
quotes_df = fetch_quotes_yahoo_for_tickers(tickers_all) if tickers_all else pd.DataFrame()
last_map = {r["Ticker"]: r["Último"] for _, r in quotes_df.iterrows()} if not quotes_df.empty else {}

df_master["Cotação"] = df_master["Ativo"].map(last_map)
df_master["Patrimônio"] = df_master.apply(lambda r: (r["Quantidade"] * r["Cotação"]) if (pd.notna(r["Cotação"]) and r["Quantidade"]>0) else 0.0, axis=1)

# Linha TOTAL (Custos e Patrimônio)
total_row = {
    "Ativo":"TOTAL",
    "Quantidade": df_master["Quantidade"].sum(skipna=True),
    "PM": None,
    "Custo Atual": df_master["Custo Atual"].sum(skipna=True),
    "Custos": df_master["Custos"].sum(skipna=True),
    "Total": df_master["Total"].sum(skipna=True),
    "Cotação": None,
    "Patrimônio": df_master["Patrimônio"].sum(skipna=True),
}
df_master_tot = pd.concat([df_master, pd.DataFrame([total_row])], ignore_index=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Ativos")
st.dataframe(
    df_master_tot[["Ativo","Quantidade","PM","Cotação","Custos","Total","Patrimônio"]],
    use_container_width=True, hide_index=True,
    column_config={
        "Quantidade": st.column_config.NumberColumn(format="%.0f"),
        "PM": st.column_config.NumberColumn(format="R$ %.2f"),
        "Cotação": st.column_config.NumberColumn(format="R$ %.2f"),
        "Custos": st.column_config.NumberColumn(format="R$ %.2f"),
        "Total": st.column_config.NumberColumn(format="R$ %.2f"),
        "Patrimônio": st.column_config.NumberColumn(format="R$ %.2f"),
    }
)

# Movimentações do ticker (abaixo da tabela)
choices = sorted([t for t in df_master["Ativo"].dropna().astype(str).unique().tolist() if t != "TOTAL"])
pick = st.selectbox("🔎 Escolha o ativo para ver as movimentações", choices, index=0 if choices else None)
if pick:
    df_mov = db_movements_for_ticker(pick, as_of=data_ate_str)
    if df_mov.empty:
        st.info("Sem movimentações para este ticker no período.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### Movimentações – {pick}")
        # Mantém PM também nas vendas; adiciona uma coluna Patrimônio (0 em vendas; nas compras pode refletir o valor da operação)
        df_mov_view = df_mov.copy()
        df_mov_view["Patrimônio"] = df_mov_view.apply(lambda r: 0.0 if str(r["Operação"]).lower()=="venda" else abs(r["Valor"]), axis=1)
        render_result_table(df_mov_view[["Data do Pregão","Operação","Quantidade","Valor","Preço Médio","Custos","Total","Patrimônio"]])
        st.markdown('</div>', unsafe_allow_html=True)

# Gerenciamento de ingestões: excluir/undo e edição manual
st.markdown("### ⚙️ Gerenciar carteira")

with st.expander("🗑️ Excluir/estornar uma ingestão (com undo)"):
    # lista simples de ingestions
    conn = db_connect()
    df_ing = pd.read_sql_query("SELECT filehash, filename, layout, data_pregao, created_at FROM ingestions ORDER BY created_at DESC", conn)
    conn.close()
    if df_ing.empty:
        st.info("Não há ingestões salvas.")
    else:
        st.dataframe(df_ing, use_container_width=True, hide_index=True)
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

with st.expander("✏️ Edição manual de um lançamento (trades_raw)"):
    # escolha por ticker -> mostra registros e permite editar um
    conn = db_connect()
    df_raw_all = pd.read_sql_query("""
        SELECT id, data_pregao, ativo, operacao, quantidade, preco_unit, valor, filehash
        FROM trades_raw
        ORDER BY id DESC
    """, conn)
    conn.close()
    if df_raw_all.empty:
        st.info("Nenhum lançamento para editar.")
    else:
        st.dataframe(df_raw_all.head(200), use_container_width=True, hide_index=True)
        edit_id = st.number_input("ID para editar", min_value=1, step=1)
        if st.button("Carregar ID"):
            row = df_raw_all[df_raw_all["id"]==edit_id]
            if row.empty:
                st.warning("ID não encontrado.")
            else:
                r = row.iloc[0]
                with st.form("edit_form"):
                    data_p = st.text_input("Data do Pregão (dd/mm/aaaa)", value=str(r["data_pregao"]))
                    ativo  = st.text_input("Ativo", value=str(r["ativo"]))
                    oper   = st.selectbox("Operação", ["Compra","Venda"], index=0 if str(r["operacao"])=="Compra" else 1)
                    qtd    = st.number_input("Quantidade", value=int(r["quantidade"]), step=1)
                    punit  = st.number_input("Preço Unitário", value=float(r["preco_unit"] or 0.0), step=0.01, format="%.2f")
                    valor  = st.number_input("Valor (positivo compra, negativo venda)", value=float(r["valor"]), step=0.01, format="%.2f")
                    submitted = st.form_submit_button("Salvar alterações")
                    if submitted:
                        conn = db_connect(); cur = conn.cursor()
                        cur.execute("""
                            UPDATE trades_raw
                            SET data_pregao=?, ativo=?, operacao=?, quantidade=?, preco_unit=?, valor=?
                            WHERE id=?
                        """, (data_p, ativo.upper().strip(), oper, int(qtd), float(punit), float(valor), int(edit_id)))
                        conn.commit(); conn.close()
                        st.success("Lançamento atualizado. Recarregue para refletir posições.")

# Fim
