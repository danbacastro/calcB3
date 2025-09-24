# calcB3.py
# Streamlit app to parse B3/XP (Nota de Corretagem) PDFs, aggregate trades, allocate costs (by value),
# compute average price, and show quotes with timestamps.
# Usage: streamlit run calcB3.py

import io
import re
import math
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# Optional deps: pdfplumber (preferred), PyPDF2 (fallback), yfinance (quotes)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# yfinance for quotes
try:
    import yfinance as yf
except Exception:
    yf = None


# ---------------------------
# Helpers
# ---------------------------

TZ = ZoneInfo("America/Sao_Paulo")
UTC = timezone.utc

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def brl(v: float) -> str:
    """Format number as Brazilian currency-like string without R$ prefix."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def parse_brl_number(s: str) -> float:
    """Parse a string like '1.234,56' into float 1234.56."""
    return float(s.replace(".", "").replace(",", "."))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using pdfplumber (preferred) or PyPDF2."""
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    # Tight tolerances preserve column order better for B3 notes
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                    text += page_text + "\n"
        except Exception:
            text = ""
    if not text and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except Exception:
            text = ""
    return text


def detect_provider(text: str) -> str:
    """Very simple provider detection: 'xp' if XP keywords found; otherwise 'b3'."""
    t = strip_accents(text).lower()
    if "xp investimentos" in t or "xp inc" in t or "xp cctvm" in t or "xp investimentos cctvm" in t:
        return "xp"
    return "b3"


def parse_trades_b3style(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    """
    Parse trade lines from B3 'Negócios realizados' section.
    Expected lines resemble:
      '1-BOVESPA V VISTA EVEN          ON NM @ 300 7,76 2.328,00 C'
      '1-BOVESPA C VISTA PETRORECSA          ON NM @ 200 13,27 2.654,00 D'
    """
    lines = text.splitlines()
    trade_lines = [l for l in lines if ("BOVESPA" in l and "VISTA" in l and "@" in l)]
    pattern = re.compile(
        r"BOVESPA\s+(?P<cv>[CV])\s+VISTA\s+(?P<paper>[A-Z0-9ÇÃÕÉÊÍÓÚA-Z ]+?)\s+.*?@\s+(?P<qty>\d+)\s+(?P<price>\d+,\d+)\s+(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})\s+(?P<dc>[CD])"
    )

    records = []
    for line in trade_lines:
        m = pattern.search(line)
        if not m:
            continue
        cv = m.group("cv")
        paper_name = m.group("paper").strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))
        dc = m.group("dc")

        ticker = name_to_ticker_map.get(paper_name, paper_name)

        records.append(
            {
                "Ativo": ticker,
                "Nome": paper_name,
                "Operação": "Compra" if cv == "C" else "Venda",
                "Quantidade": qty,
                "Preço_Unitário": price,
                "Valor": value if cv == "C" else -value,  # vendas negativas
                "Sinal_DC": dc,
            }
        )
    return pd.DataFrame(records)


def parse_trades_generic_table(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    """
    Fallback very generic parser: look for lines that resemble '<NAME> ... @ <QTY> <PRICE> <VALUE>' even if "BOVESPA/VISTA" not present.
    """
    lines = [l for l in text.splitlines() if "@" in l and re.search(r"\d{1,3}(?:\.\d{3})*,\d{2}", l)]
    pattern = re.compile(
        r"(?P<cv>\b[CV]\b|\bCompra\b|\bVenda\b).*?(?P<paper>[A-Z0-9ÇÃÕÉÊÍÓÚA-Z ]{3,})\s+.*?@\s+(?P<qty>\d+)\s+(?P<price>\d+,\d+)\s+(?P<value>\d{1,3}(?:\.\d{3})*,\d{2})"
    )
    records = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        cv_raw = m.group("cv").strip().upper()
        cv = "C" if cv_raw.startswith("C") else "V"
        paper_name = m.group("paper").strip()
        qty = int(m.group("qty"))
        price = parse_brl_number(m.group("price"))
        value = parse_brl_number(m.group("value"))
        ticker = name_to_ticker_map.get(paper_name, paper_name)

        records.append(
            {
                "Ativo": ticker,
                "Nome": paper_name,
                "Operação": "Compra" if cv == "C" else "Venda",
                "Quantidade": qty,
                "Preço_Unitário": price,
                "Valor": value if cv == "C" else -value,
                "Sinal_DC": "",
            }
        )
    return pd.DataFrame(records)


def parse_trades_any(text: str, name_to_ticker_map: dict) -> pd.DataFrame:
    """Try B3-style first; if empty, try generic fallback (useful for some XP layouts)."""
    df = parse_trades_b3style(text, name_to_ticker_map)
    if df.empty:
        df = parse_trades_generic_table(text, name_to_ticker_map)
    return df


def parse_header_dates_and_net(text: str):
    """Extract 'Data do pregão' and 'Líquido para <data> <valor>' robustly (B3/XP)."""
    lines = text.splitlines()
    pairs = list(zip(lines, [strip_accents(l).lower() for l in lines]))

    date_labels = ("data do preg", "data preg", "data da negoc", "data negoc", "negociacao", "negociação", "pregao")
    data_pregao = None
    for raw, norm in pairs[:120]:
        if any(lbl in norm for lbl in date_labels):
            m = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if m:
                data_pregao = m.group(1)
                break
    if not data_pregao:
        for raw, norm in pairs[:120]:
            if re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", raw):
                continue
            m = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
            if m:
                data_pregao = m.group(1)
                break

    liquido_para_data = None
    liquido_para_valor = None
    for raw, norm in pairs[::-1]:
        if ("liquido para" in norm) or ("l\u00edquido para" in norm):
            md = re.search(r"(?:L[ií]quido para)\s+(\d{2}/\d{2}/\d{4})", raw)
            if md:
                liquido_para_data = md.group(1)
            vals = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2})", raw)
            if vals:
                liquido_para_valor = vals[-1]
            break

    if data_pregao and liquido_para_data and data_pregao == liquido_para_data:
        for raw, norm in pairs[:160]:
            m = re.findall(r"(\d{2}/\d{2}/\d{4})", raw)
            for d in m:
                if d != liquido_para_data:
                    data_pregao = d
                    break
            if data_pregao != liquido_para_data:
                break

    return data_pregao, liquido_para_data, liquido_para_valor


def parse_cost_components(text: str) -> dict:
    """
    Canonicalize cost components across B3 and XP.
    Returns a dict with atomic components (no aggregates), plus metadata keys:
      - _aggregates_found: list of aggregate labels found (not counted unless needed)
      - _used_aggregate_substitute: True if we had to use an aggregate (e.g., 'Taxas B3') due to missing atomics.
      - _irrf_detected: numeric IRRF for info (never included in rateio)
    """
    pats = {
        "liquidacao": r"Taxa\s*de\s*liquida[cç][aã]o\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "emolumentos": r"Emolumentos\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "registro": r"(?:Taxa\s*de\s*Registro|Taxa\s*de\s*Transf\.\s*de\s*Ativos)\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "corretagem": r"Corretag\w*\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "taxa_operacional": r"(?:Taxa|Tarifa)\s*Operacion\w+\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "iss": r"\bISS\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "impostos": r"\bImpostos\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "outros": r"\bOutros\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
    }
    components = {}
    for key, pat in pats.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            components[key] = parse_brl_number(m.group(1))

    aggregates = {}
    agg_pats = {
        "total_bovespa_soma": r"Total\s*Bovespa\s*/\s*Soma\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "taxas_b3": r"Taxas?\s*B3\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*,\d{2})",
        "total_custos_despesas": r"Total\s*Custos\s*/\s*Despesas\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
        "custos_despesas": r"\bCustos\s*/\s*Despesas\b\s+(\d{1,3}(?:\.\d{3})*,\d{2})",
    }
    for key, pat in agg_pats.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            aggregates[key] = parse_brl_number(m.group(1))

    irrf = None
    m_irrf = re.search(r"I\.?R\.?R\.?F\.?.*?(\d{1,3}(?:\.\d{3})*,\d{2})", text, flags=re.IGNORECASE)
    if m_irrf:
        irrf = parse_brl_number(m_irrf.group(1))

    atomic_total = sum(components.values()) if components else 0.0
    used_aggregate = False

    if atomic_total == 0.0:
        if "taxas_b3" in aggregates:
            components["taxas_b3_subst"] = aggregates["taxas_b3"]
            used_aggregate = True
        elif "total_bovespa_soma" in aggregates:
            components["total_bovespa_soma_subst"] = aggregates["total_bovespa_soma"]
            used_aggregate = True
        elif "total_custos_despesas" in aggregates:
            components["custos_despesas_subst"] = aggregates["total_custos_despesas"]
            used_aggregate = True

    components["_aggregates_found"] = list(aggregates.keys())
    components["_used_aggregate_substitute"] = used_aggregate
    components["_irrf_detected"] = irrf
    return components


def allocate_costs_proportional(amounts: pd.Series, total_costs: float) -> pd.Series:
    """Allocate total_costs across 'amounts' proportionally (by value), with rounding fix to match exactly."""
    if amounts.sum() <= 0 or total_costs <= 0:
        return pd.Series([0.0] * len(amounts), index=amounts.index, dtype=float)

    raw = amounts / amounts.sum() * total_costs
    floored = (raw * 100).apply(math.floor) / 100.0
    residual = round(total_costs - floored.sum(), 2)
    if abs(residual) > 0:
        frac = (raw * 100) - (raw * 100).apply(math.floor)
        order = frac.sort_values(ascending=(residual < 0)).index
        step = 0.01 if residual > 0 else -0.01
        i = 0
        while round(residual, 2) != 0 and i < len(order):
            floored.loc[order[i]] += step
            residual = round(residual - step, 2)
            i += 1
    return floored


def guess_yf_symbols_for_b3(ticker: str) -> list:
    """Return likely Yahoo symbols for a B3 ticker ('.SA' suffix)."""
    if not ticker:
        return []
    t = ticker.strip().upper()
    return [f"{t}.SA", t]


def _format_dt_local(dt) -> str:
    if dt is None or pd.isna(dt):
        return ""
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


@st.cache_data(show_spinner=False, ttl=60)
def fetch_quotes_for_tickers(tickers: list, ref_date: datetime | None = None) -> pd.DataFrame:
    """Intraday + fechamento do pregão."""
    cols = ["Ticker", "Símbolo", "Último", "Último (quando)", "Fechamento (pregão)", "Pregão (data)"]
    out_rows = []

    if yf is None or not tickers:
        return pd.DataFrame(columns=cols)

    now_tz = datetime.now(TZ)
    if ref_date is None:
        start_daily = (now_tz - timedelta(days=10)).strftime("%Y-%m-%d")
        end_daily = (now_tz + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_daily = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end_daily = (ref_date + timedelta(days=7)).strftime("%Y-%m-%d")

    for t in tickers:
        sym = guess_yf_symbols_for_b3(t)[0]
        last_px = None
        last_dt = None
        close_px = None
        close_dt = None

        try:
            h1m = yf.Ticker(sym).history(period="2d", interval="1m", auto_adjust=False)
            if not h1m.empty and "Close" in h1m:
                series = h1m["Close"].dropna()
                if not series.empty:
                    last_px = float(series.iloc[-1])
                    idx = series.index[-1]
                    if getattr(idx, "tzinfo", None) is None:
                        idx = pd.Timestamp(idx).tz_localize(UTC)
                    last_dt = idx.tz_convert(TZ)
        except Exception:
            pass

        try:
            hd = yf.Ticker(sym).history(start=start_daily, end=end_daily, auto_adjust=False)
            if not hd.empty and "Close" in hd:
                s = hd["Close"].dropna()
                if not s.empty:
                    if ref_date is None:
                        close_px = float(s.iloc[-1]); idx = s.index[-1]
                    else:
                        idxs = s.index[s.index.date <= ref_date.date()]
                        if len(idxs) > 0:
                            close_px = float(s.loc[idxs[-1]]); idx = idxs[-1]
                        else:
                            idx = s.index[-1]; close_px = float(s.iloc[-1])
                    if getattr(idx, "tzinfo", None) is None:
                        idx = pd.Timestamp(idx).tz_localize(UTC)
                    close_dt = idx.tz_convert(TZ)
        except Exception:
            pass

        out_rows.append({
            "Ticker": t,
            "Símbolo": sym,
            "Último": last_px,
            "Último (quando)": _format_dt_local(last_dt),
            "Fechamento (pregão)": close_px,
            "Pregão (data)": close_dt.date().strftime("%d/%m/%Y") if close_dt else "",
        })

    return pd.DataFrame(out_rows, columns=cols)


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Calc B3 - Nota de Corretagem", layout="wide")
st.markdown(
    """
    <style>
    .big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .subtitle { color: #6c757d; margin-bottom: 1rem; }
    .card { padding: 1rem 1.25rem; border: 1px solid #e9ecef; border-radius: 14px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    .muted { color: #6c757d; font-size: 0.9rem; }
    .section-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem; }
    .badge { display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.85rem; font-weight: 700; border: 1px solid transparent; }
    .badge-b3 { background: #E7F5FF; color: #095BC6; border-color: #B3D7FF; }
    .badge-xp { background: #FFF3BF; color: #8B6A00; border-color: #FFD875; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="big-title">Calc B3 – Nota de Corretagem</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extraia, agregue e rateie custos por ativo (proporcional ao valor financeiro), com preço médio e cotações.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Opções")
    incluir_bovespa_soma = st.checkbox("Incluir 'Total Bovespa / Soma' nos custos para rateio", value=False)
    mostrar_cotacoes = st.checkbox("Mostrar cotações (yfinance)", value=True)
    st.markdown("---")
    st.subheader("Mapeamento opcional Nome→Ticker")
    st.write("Se sua nota usa **nome da empresa** (ex.: VULCABRAS) ao invés do ticker (VULC3), forneça um CSV `Nome,Ticker`.")
    map_file = st.file_uploader("Upload CSV de mapeamento (opcional)", type=["csv"], key="map_csv")

# Mapeamento inicial (exemplos desta conversa)
default_map = {"EVEN": "EVEN3", "PETRORECSA": "RECV3", "VULCABRAS": "VULC3"}

# Carregar mapeamento adicional
if map_file is not None:
    try:
        mdf = pd.read_csv(map_file)
        for _, row in mdf.iterrows():
            if pd.notna(row.get("Nome")) and pd.notna(row.get("Ticker")):
                default_map[str(row["Nome"]).strip().upper()] = str(row["Ticker"]).strip().upper()
    except Exception as e:
        st.warning(f"Falha ao ler CSV de mapeamento: {e}")

uploaded = st.file_uploader("Carregue o PDF da B3/XP", type=["pdf"], key="pdf")

if uploaded is None:
    st.info("Envie um arquivo PDF de nota B3/XP para começar.")
    st.stop()

# Parse PDF text
pdf_bytes = uploaded.read()
text = extract_text_from_pdf(pdf_bytes)
if not text:
    st.error("Não consegui extrair texto do PDF. Tente enviar um PDF com texto (não imagem) ou exporte novamente.")
    st.stop()

# Detect provider and show badge (AGORA sim, depois do text)
provider = detect_provider(text)
_badge_cls = "badge-xp" if provider == "xp" else "badge-b3"
_badge_label = provider.upper()
st.markdown(f'<div class="muted">Provedor detectado: <span class="badge {_badge_cls}">{_badge_label}</span></div>', unsafe_allow_html=True)

# Header info
data_pregao_str, liquido_para_data_str, liquido_para_valor_str = parse_header_dates_and_net(text)

# Header cards
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown('<div class="card"><div class="section-title">Data do Pregão</div><div class="muted">{}</div></div>'.format(data_pregao_str or "—"), unsafe_allow_html=True)
with colB:
    st.markdown('<div class="card"><div class="section-title">Líquido para (Data)</div><div class="muted">{}</div></div>'.format(liquido_para_data_str or "—"), unsafe_allow_html=True)
with colC:
    st.markdown('<div class="card"><div class="section-title">Líquido para (Valor)</div><div class="muted">R$ {}</div></div>'.format(liquido_para_valor_str or "—"), unsafe_allow_html=True)
with colD:
    now_local = datetime.now(TZ).strftime("%d/%m/%Y %H:%M")
    st.markdown('<div class="card"><div class="section-title">Agora (BRT)</div><div class="muted">{}</div></div>'.format(now_local), unsafe_allow_html=True)

# Trades
df_trades = parse_trades_any(text, default_map)
if df_trades.empty:
    st.error("Não encontrei linhas de negociação. Tente: (i) subir o PDF original (não imagem), (ii) enviar um mapeamento Nome→Ticker ou (iii) compartilhar um PDF XP de exemplo para que eu adapte o parser.")
    st.stop()

# Agrupar por Ativo + Operação (base de rateio = valor financeiro absoluto)
df_trades["AbsValor"] = df_trades["Valor"].abs()
agg = (
    df_trades.groupby(["Ativo", "Operação"], as_index=False)
    .agg(Quantidade=("Quantidade", "sum"),
         Valor=("Valor", "sum"),
         BaseRateio=("AbsValor", "sum"))
)

# Custos padronizados B3/XP
fees = parse_cost_components(text)
atomic_keys = ["liquidacao", "emolumentos", "registro", "corretagem", "taxa_operacional", "iss", "impostos", "outros"]
atomics_sum = sum(fees.get(k, 0.0) for k in atomic_keys)
total_costs_detected = round(atomics_sum if atomics_sum > 0 else sum(v for k, v in fees.items() if k.endswith("_subst")), 2)

# Campo para confirmar/ajustar total de custos a ratear
st.subheader("Custos a ratear")
col1, col2 = st.columns([1, 3])
with col1:
    st.write("Componentes detectados:")
with col2:
    if fees:
        shown = {k: f"R$ {brl(v)}" for k, v in fees.items() if not k.startswith("_")}
        st.write(shown if shown else "(nenhum componente detectado)")
        if fees.get("_irrf_detected"):
            st.caption(f"IRRF detectado (não incluído no rateio): R$ {brl(fees['_irrf_detected'])}")
        if fees.get("_used_aggregate_substitute"):
            st.caption("Usando subtotal agregado (ex.: 'Taxas B3') como substituto por falta de componentes atômicos.")

total_costs_input = st.number_input(
    "Total de custos para **rateio** (proporcional ao valor financeiro)",
    min_value=0.0,
    value=round(total_costs_detected, 2),
    step=0.01,
    format="%.2f",
    help="Soma das taxas que deseja distribuir entre os ativos. Ajuste se necessário para casar com seus 'Custos' esperados."
)

# Alocar custos proporcionalmente por (Ativo, Operação)
alloc_series = allocate_costs_proportional(agg.set_index(["Ativo", "Operação"])["BaseRateio"], total_costs_input)
alloc_df = alloc_series.reset_index()
alloc_df.columns = ["Ativo", "Operação", "Custos"]

# Merge com agregação
out = agg.merge(alloc_df, on=["Ativo", "Operação"], how="left")
out["Custos"] = out["Custos"].fillna(0.0)

# Total (módulo): Venda => |Valor| - Custos ; Compra => |Valor| + Custos
out["Total"] = out.apply(lambda r: abs(r["Valor"]) - r["Custos"] if r["Valor"] < 0 else abs(r["Valor"]) + r["Custos"], axis=1)

# Preço médio = |Valor| / Quantidade (sem custos)
out["Preço Médio"] = out.apply(lambda r: (abs(r["Valor"]) / r["Quantidade"]) if r["Quantidade"] else None, axis=1)

# Ordenar por Ativo
out = out.sort_values(["Ativo", "Operação"]).reset_index(drop=True)

# ===== Resultado (CARD) =====
st.markdown('<div class="section-title">📊 Resultado</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(
        out[["Ativo", "Operação", "Quantidade", "Valor", "Preço Médio", "Custos", "Total"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quantidade": st.column_config.NumberColumn(format="%.0f"),
            "Valor": st.column_config.NumberColumn(format="R$ %.2f"),
            "Preço Médio": st.column_config.NumberColumn(format="R$ %.2f"),
            "Custos": st.column_config.NumberColumn(format="R$ %.2f"),
            "Total": st.column_config.NumberColumn(format="R$ %.2f"),
        }
    )
    csv_bytes = out.rename(columns={"BaseRateio": "Base_Rateio"}).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV (numérico)", data=csv_bytes, file_name="extrato_b3_agregado.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Cotações (CARD) =====
if mostrar_cotacoes:
    st.markdown('<div class="section-title">💹 Cotações</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        colr1, colr2 = st.columns([1, 4])
        with colr1:
            if st.button("🔄 Atualizar cotações"):
                st.cache_data.clear()
                st.rerun()
        with colr2:
            st.caption("Atualiza os dados intraday (1 minuto). Cache = 60s.")

        if yf is None:
            st.info("Pacote 'yfinance' não instalado. Para ver cotações, instale com: pip install yfinance")
        else:
            ref_date = None
            if data_pregao_str:
                try:
                    ref_date = datetime.strptime(data_pregao_str, "%d/%m/%Y")
                except Exception:
                    ref_date = None
            tickers = sorted(out["Ativo"].unique().tolist())
            quotes = fetch_quotes_for_tickers(tickers, ref_date=ref_date)
            if not quotes.empty:
                now_local = datetime.now(TZ).strftime("%d/%m/%Y %H:%M")
                st.caption(f"Agora (BRT): {now_local} • Fonte: Yahoo Finance (intraday 1m)")
                qfmt = quotes.copy()
                for c in ["Último", "Fechamento (pregão)"]:
                    qfmt[c] = qfmt[c].map(lambda x: brl(x) if pd.notna(x) else "")
                st.dataframe(qfmt, use_container_width=True, hide_index=True)
            else:
                st.info("Não foi possível obter cotações para os tickers detectados.")
        st.markdown('</div>', unsafe_allow_html=True)

# ===== Debug (Expander) =====
with st.expander("🛠️ Debug (mostrar/ocultar)"):
    st.text_area("Texto extraído (parcial)", value=text[:4000], height=200)
    st.write("Provedor detectado:", provider)
    st.write("Fees detectados:", fees)
    st.write("Agregado numérico:", out)
