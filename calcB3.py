# calcB3.py
# App Streamlit: múltiplos PDFs B3/XP, rateio por valor, PM, cotações (Yahoo padrão),
# colunas destacadas (Data do Pregão, Quantidade, Valor, Total) e "Operação" colorida.

import io
import re
import math
import unicodedata
import hashlib
from typing import Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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


# =============================================================================
# PDF → texto
# =============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    # 1) pdfplumber (strict)
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
    if not m: return None
    base, cls = m.group(1), m.group(2)
    return f"{base}3" if cls == "ON" else f"{base}4"

def clean_b3_tickers(lst) -> list:
    out = []
    for t in lst:
        if not isinstance(t, str): continue
        t = strip_accents(t).upper().strip()
        if not t: continue
        if re.fullmatch(r"[A-Z]{3,6}\d{1,2}[A-Z]?", t):
            out.append(t)
    return sorted(set(out))


# =============================================================================
# Parsers
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
        if not m: continue
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
        if not m: continue
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
# Quotes (Yahoo padrão)
# =============================================================================
def guess_yf_symbols_for_b3(ticker: str) -> list:
    return [f"{ticker.upper()}.SA", ticker.upper()]

def _format_dt_local(dt) -> str:
    if dt is None or pd.isna(dt): return ""
    try:
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None: dt = dt.tz_localize(UTC)
            return dt.astimezone(TZ).strftime("%d/%m/%Y %H:%M")
        if getattr(dt, "tzinfo", None) is None: dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        try: return pd.to_datetime(dt, utc=True).tz_convert(TZ).strftime("%d/%m/%Y %H:%M")
        except Exception: return str(dt)

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
            "Último (quando)": (_format_dt_local(last_dt)+(" (fechamento)" if last_from_close and last_dt else "")) if last_dt else "",
            "Fechamento (pregão)": close_px,
            "Pregão (data)": close_dt.date().strftime("%d/%m/%Y") if close_dt else "",
            "Motivo": motivo,
        })
    return pd.DataFrame(rows, columns=cols)

def _parse_price_any(txt: str) -> float | None:
    if not txt: return None
    s = re.sub(r"[^\d,\.]", "", txt)
    if not s: return None
    if "," in s and "." in s: s = s.replace(".","").replace(",",".")
    elif "," in s: s = s.replace(",",".")
    try: return float(s)
    except Exception: return None

@st.cache_data(show_spinner=False, ttl=60)
def fetch_quotes_google_for_tickers(tickers: list) -> pd.DataFrame:
    cols = ["Ticker","Símbolo","Último","Último (quando)","Fechamento (pregão)","Pregão (data)","Motivo"]
    rows = []
    if requests is None or not tickers: return pd.DataFrame(columns=cols)
    now_brt = datetime.now(TZ).strftime("%d/%m/%Y %H:%M")
    hdr = {"User-Agent":"Mozilla/5.0"}
    for t in tickers:
        sym = f"{t}:BVMF"; ultimo = None; motivo = ""
        try:
            url = f"https://www.google.com/finance/quote/{t}:BVMF"
            resp = requests.get(url, headers=hdr, timeout=10)
            if resp.status_code == 200 and resp.text:
                html = resp.text
                cands = re.findall(r'YMlKec[^>]*>([^<]+)<', html)
                price = None
                for c in cands:
                    price = _parse_price_any(c)
                    if price is not None:
                        break
                if price is None:
                    m = re.search(r'data-last-price="([^"]+)"', html)
                    if m:
                        price = _parse_price_any(m.group(1))
                if price is not None:
                    ultimo = price
                else:
                    motivo = "preço não encontrado"
            else:
                motivo = f"HTTP {resp.status_code}"
        except Exception as e:
            motivo = f"erro: {e}"
        rows.append({
            "Ticker": t, "Símbolo": sym, "Último": ultimo,
            "Último (quando)": f"{now_brt} (Google)",
            "Fechamento (pregão)": None, "Pregão (data)": "", "Motivo": motivo
        })
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Processamento de uma nota
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
# UI helpers (estilo e render)
# =============================================================================
def style_result_df(df: pd.DataFrame) -> Any:
    """Destaque colunas: Data do Pregão, Quantidade, Valor, Total. 'Operação' colorida."""
    cols_emphasis = [c for c in ["Data do Pregão","Quantidade","Valor","Total"] if c in df.columns]

    sty = df.style.format({
        "Quantidade": "{:.0f}",
        "Valor": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Preço Médio": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Custos": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
        "Total": lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "",
    })

    # Cor na coluna Operação
    def op_style(v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "compra": return "color:#16a34a; font-weight:700"
            if s == "venda":  return "color:#dc2626; font-weight:700"
        return ""
    if "Operação" in df.columns:
        sty = sty.applymap(op_style, subset=["Operação"])

    # Destaque de colunas (borda esquerda/direita + fundo + negrito)
    if cols_emphasis:
        sty = sty.set_properties(
            subset=cols_emphasis,
            **{"border-left":"0px solid #000","border-right":"0px solid #000","background-color":"#f6f7f9","font-weight":"700"}
        )
        # Cabeçalhos dessas colunas
        table_styles = []
        for c in cols_emphasis:
            idx = df.columns.get_loc(c)
            table_styles.append({
                "selector": f"th.col_heading.level0.col{idx}",
                "props": [("border","0px solid #000"),("background-color","#f6f7f9"),("font-weight","700")]
            })
        sty = sty.set_table_styles(table_styles, overwrite=False)

    # Borda geral suave
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
st.markdown('<div class="subtitle">Upload múltiplo de notas B3/XP. Agregue, rateie por valor, calcule PM e veja cotações (Yahoo padrão).</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Opções")
    mostrar_cotacoes = st.checkbox("Mostrar cotações", value=True)
    fonte = st.radio("Fonte das cotações", ["Yahoo Finance","Google Finance"], index=0)
    if st.button("🔄 Atualizar cotações", use_container_width=True):
        st.cache_data.clear(); st.rerun()
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

uploads = st.file_uploader("Carregue um ou mais PDFs da B3/XP", type=["pdf"], accept_multiple_files=True, key="pdfs")
if not uploads:
    st.info("Envie um ou mais arquivos PDF para começar."); st.stop()

results = []
for f in uploads:
    try:
        pdf_bytes = f.read()
        res = process_one_pdf(pdf_bytes, default_map)
        res["filename"] = f.name; res["filehash"] = sha1(pdf_bytes)
    except Exception as e:
        res = {"ok": False, "error": str(e), "filename": f.name}
    results.append(res)

tab_titles = ["Consolidado"] + [r.get("filename", f"Arquivo {i+1}") for i, r in enumerate(results)]
tabs = st.tabs(tab_titles)

# Abas individuais
for idx, res in enumerate(results, start=1):
    with tabs[idx]:
        st.markdown(f"### 📄 {res.get('filename','Arquivo')}")
        if not res.get("ok"):
            st.error(res.get("error","Erro ao processar")); continue
        layout = res["layout"]; _badge_cls = "badge-xp" if layout=="XP" else "badge-b3"
        st.markdown(f'<div class="muted">Layout detectado: <span class="badge {_badge_cls}">{layout}</span> • Extrator: {res["extractor"]}</div>', unsafe_allow_html=True)

        colA, colB, colC, colD = st.columns(4)
        with colA: st.markdown(f'<div class="card"><div class="section-title">Data do Pregão</div><div class="muted">{res["data_pregao"] or "—"}</div></div>', unsafe_allow_html=True)
        with colB: st.markdown(f'<div class="card"><div class="section-title">Líquido para (Data)</div><div class="muted">{res["liquido_para_data"] or "—"}</div></div>', unsafe_allow_html=True)
        with colC: st.markdown(f'<div class="card"><div class="section-title">Líquido para (Valor)</div><div class="muted">R$ {res["liquido_para_valor"] or "—"}</div></div>', unsafe_allow_html=True)
        with colD:
            now_local = datetime.now(TZ).strftime("%d/%m/%Y %H:%M")
            st.markdown(f'<div class="card"><div class="section-title">Agora (BRT)</div><div class="muted">{now_local}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Resultado</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_result_table(res["out"][["Data do Pregão","Ativo","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
            csv_bytes = res["out"].to_csv(index=False).encode("utf-8-sig")
            st.download_button("Baixar CSV (nota)", data=csv_bytes, file_name=f"resultado_{res['filename']}.csv", mime="text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🛠️ Debug (mostrar/ocultar)"):
            text = res.get("text","")
            st.text_area("Texto extraído (primeiros 3000 chars)", value=text[:3000], height=240)
            st.write("Fees detectados:", res["fees"])
            st.write("Cálculo total (regra):", res["used_detail"])

# Aba Consolidado (por Data do Pregão + Ativo + Operação)
with tabs[0]:
    st.markdown("### 📚 Consolidado (Data do Pregão + Ativo + Operação)")
    valid_outs = [r["out"] for r in results if r.get("ok")]
    if not valid_outs:
        st.info("Nenhuma nota válida processada.")
    else:
        all_out = pd.concat(valid_outs, ignore_index=True)
        cons = (
            all_out.groupby(["Data do Pregão","Ativo","Operação"], as_index=False)
            .agg(Quantidade=("Quantidade","sum"),
                 Valor=("Valor","sum"),
                 Custos=("Custos","sum"),
                 Total=("Total","sum"))
            .sort_values(["Data do Pregão","Ativo","Operação"])
        )
        cons["Preço Médio"] = cons.apply(lambda r: (abs(r["Valor"])/r["Quantidade"]) if r["Quantidade"] else None, axis=1)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_result_table(cons[["Data do Pregão","Ativo","Operação","Quantidade","Valor","Preço Médio","Custos","Total"]])
        csv_cons = cons.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV (consolidado)", data=csv_cons, file_name="resultado_consolidado.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

# Cotações (padrão Yahoo)
if mostrar_cotacoes:
    st.markdown('<div class="section-title">💹 Cotações</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tickers = []
        for r in results:
            if r.get("ok"): tickers.extend(r["out"]["Ativo"].dropna().astype(str).tolist())
        tickers = clean_b3_tickers(tickers)
        colr1, colr2 = st.columns([4,1])
        with colr1: st.caption("Fonte padrão: Yahoo Finance • Cache 60s • Use o botão para atualizar.")
        with colr2:
            if st.button("🔄 Atualizar cotações (global)"):
                st.cache_data.clear(); st.rerun()
        if not tickers:
            st.info("Nenhum ticker válido detectado para consultar.")
        else:
            if fonte == "Yahoo Finance":
                dfq = fetch_quotes_yahoo_for_tickers(tickers, ref_date=None)
                if dfq.empty:
                    st.warning("Não foi possível obter cotações via Yahoo Finance.")
                else:
                    qfmt = dfq.copy()
                    for c in ["Último","Fechamento (pregão)"]:
                        qfmt[c] = qfmt[c].map(lambda x: brl(x) if pd.notna(x) else "")
                    st.dataframe(qfmt, use_container_width=True, hide_index=True)
            else:
                dfq = fetch_quotes_google_for_tickers(tickers)
                if dfq.empty:
                    st.warning("Não foi possível obter cotações via Google Finance.")
                else:
                    qfmt = dfq.copy(); qfmt["Último"] = qfmt["Último"].map(lambda x: brl(x) if pd.notna(x) else "")
                    st.dataframe(qfmt, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
