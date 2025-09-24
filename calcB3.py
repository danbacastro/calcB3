# calcB3.py
# Streamlit app to parse B3/XP (Nota de Corretagem) PDFs, aggregate trades, allocate costs (by value),
# compute average price, and show quotes with timestamps (intraday 1m). Includes provider badge.
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

# PyMuPDF (fitz) for robust text extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


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
    """Parse a string like '1.234,56' into float 12
