"""
Stock Analysis Application - Komplexní nástroj pro analýzu akcií
Autor: Python FinTech Developer
Popis: Streamlit aplikace pro technickou, fundamentální a sentiment analýzu akcií
"""

import streamlit as st
from urllib.parse import quote_plus
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Any, Dict, List, Tuple, Optional

def extract_news_meta(item: Any) -> Dict[str, Any]:
    """Best-effort extraction of title/source/url from yfinance news item."""
    if item is None:
        return {"title": "Bez názvu", "publisher": None, "url": None, "provider": "Yahoo Finance"}
    if isinstance(item, str):
        return {"title": item.strip() or "Bez názvu", "publisher": None, "url": None, "provider": "Yahoo Finance"}

    if not isinstance(item, dict):
        return {"title": str(item), "publisher": None, "url": None, "provider": "Yahoo Finance"}

    title = (
        item.get("title")
        or item.get("headline")
        or item.get("text")
        or item.get("summary")
        or None
    )
    publisher = item.get("publisher") or item.get("source") or item.get("provider") or None
    url = item.get("link") or item.get("url") or None

    # Common nested structure in Yahoo payloads
    content = item.get("content")
    if isinstance(content, dict):
        title = title or content.get("title") or content.get("headline") or content.get("summary") or content.get("description")
        publisher = publisher or content.get("publisher") or content.get("provider") or content.get("source")
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = url or canonical.get("url")

    # Some payloads store url in item["content"]["clickThroughUrl"]["url"]
    if isinstance(content, dict):
        ctu = content.get("clickThroughUrl")
        if isinstance(ctu, dict):
            url = url or ctu.get("url")

    return {
        "title": (str(title).strip() if title else "Bez názvu"),
        "publisher": publisher,
        "url": url,
        "provider": "Yahoo Finance",
    }

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURACE STRÁNKY
# ============================================================================

st.set_page_config(
    page_title="📈 Stock Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .positive {
        color: #10b981;
        font-weight: bold;
    }
    .negative {
        color: #ef4444;
        font-weight: bold;
    }
    .neutral {
        color: #6b7280;
        font-weight: bold;
    }
    .sentiment-positive {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .sentiment-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .sentiment-neutral {
        background-color: #e5e7eb;
        color: #374151;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AI FUNKCE
# ============================================================================


def estimate_fair_value(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Return (fair_value_price, method). Best-effort.
    Priority:
      1) Yahoo analyst targetMeanPrice / targetMedianPrice
      2) Simple per-share DCF from freeCashflow (very rough)
    """
    try:
        tmean = info.get("targetMeanPrice")
        tmed = info.get("targetMedianPrice")
        if isinstance(tmean, (int, float)) and tmean and tmean > 0:
            return float(tmean), "Analyst target (mean)"
        if isinstance(tmed, (int, float)) and tmed and tmed > 0:
            return float(tmed), "Analyst target (median)"
    except Exception:
        pass

    # Rough DCF fallback (owner-earnings style) using FCF and shares outstanding
    try:
        fcf = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not (isinstance(fcf, (int, float)) and fcf and fcf > 0):
            return None, "N/A"
        if not (isinstance(shares, (int, float)) and shares and shares > 0):
            return None, "N/A"

        # Conservative defaults
        discount = 0.10
        growth = 0.06   # 6% for 5y
        terminal = 0.03 # 3% perpetual
        years = 5

        pv = 0.0
        f = float(fcf)
        for y in range(1, years + 1):
            f = f * (1 + growth)
            pv += f / ((1 + discount) ** y)

        terminal_value = (f * (1 + terminal)) / max(1e-6, (discount - terminal))
        pv += terminal_value / ((1 + discount) ** years)

        fair_total_equity = pv  # ignoring net debt for simplicity (since info can be missing)
        fair_per_share = fair_total_equity / float(shares)
        if fair_per_share > 0 and fair_per_share < 1e6:
            return float(fair_per_share), "Simple FCF DCF (rough)"
    except Exception:
        pass

    return None, "N/A"


def classify_insider_action(text_value: str, txn_value: str) -> str:
    """Classify insider transaction into BUY/SELL/GRANT/OTHER based on available text."""
    s = f"{txn_value or ''} {text_value or ''}".lower()
    if any(k in s for k in ["sale", "sell", "sold", "dispose", "disposed"]):
        return "SELL"
    if any(k in s for k in ["buy", "purchase", "purchased", "acquire", "acquired"]):
        return "BUY"
    if any(k in s for k in ["award", "grant", "stock award", "rsu", "option", "vesting", "vested"]):
        return "GRANT"
    return "OTHER"


def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None
def analyze_news_with_gemini(news_titles, api_key, ticker):
    """Analyzuje novinky pomocí Gemini.

    Ošetření:
    - limituje počet titulků (kvůli tokenům/kvótám),
    - model lze přepnout přes env GEMINI_MODEL (default: gemini-2.5-flash),
    - při 429/RESOURCE_EXHAUSTED (nebo jiné chybě) vrací fallback místo chybové hlášky.
    """
    import os
    import re

    def _is_quota_error(msg: str) -> bool:
        u = (msg or "").upper()
        return ("RESOURCE_EXHAUSTED" in u) or ("QUOTA" in u) or ("429" in u)

    def _fallback(headlines):
        pos = ["beats", "surge", "rally", "upgrade", "record", "strong", "growth", "profits", "wins", "raises"]
        neg = ["miss", "drop", "sell-off", "downgrade", "lawsuit", "weak", "cut", "loss", "decline", "slump"]
        text = " ".join(str(x).lower() for x in (headlines or []))
        score = sum(text.count(w) for w in pos) - sum(text.count(w) for w in neg)
        if score >= 2:
            return "Pozitivní", "AI shrnutí je dočasně nedostupné, sentiment je odhadnutý z titulků."
        if score <= -2:
            return "Negativní", "AI shrnutí je dočasně nedostupné, sentiment je odhadnutý z titulků."
        return "Neutrální", "AI shrnutí je dočasně nedostupné, sentiment je odhadnutý z titulků."

    # 1) Filtrace + limit počtu titulků
    valid_titles = []
    for t in (news_titles or []):
        s = str(t).strip()
        if s and s.lower() != "bez názvu" and len(s) > 3:
            valid_titles.append(re.sub(r"\s+", " ", s)[:200])

    if not valid_titles:
        return "Neutrální", "Pro tento ticker nejsou dostupné čitelné titulky zpráv."

    valid_titles = valid_titles[:8]

    try:
        from google import genai  # google-genai
        client = genai.Client(api_key=api_key)

        news_text = "\n".join([f"- {t}" for t in valid_titles])
        prompt = f"""Jsi finanční analytik. Analyzuj sentiment těchto zpráv pro akcii {ticker}:

{news_text}

Odpověz POUZE v tomto formátu:
SENTIMENT: [Pozitivní/Negativní/Neutrální]
SHRNUTÍ: [Stručné shrnutí v češtině, max 2 věty]"""

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(model=model_name, contents=prompt)
        result = getattr(response, "text", "") or ""

        sentiment = "Neutrální"
        summary = result.strip() or "(bez odpovědi)"

        for line in result.split("\n"):
            u = line.upper()
            if "SENTIMENT" in u and ":" in line:
                sentiment = line.split(":", 1)[-1].strip().replace("[", "").replace("]", "")
            elif ("SHRNUTÍ" in u or "SUMMARY" in u) and ":" in line:
                summary = line.split(":", 1)[-1].strip()

        return sentiment, summary

    except Exception as e:
        msg = str(e)
        # pro kvóty/429 i jiné chyby vrať fallback, ať UI nezůstane "červené"
        sent, summ = _fallback(valid_titles)
        if _is_quota_error(msg):
            return sent, summ
        return sent, summ



    
def analyze_news_with_openai(news_titles: List[str], api_key: str, ticker: str) -> Tuple[str, str]:
    """
    Analyzuje novinky pomocí OpenAI GPT
    
    Args:
        news_titles: Seznam titulků zpráv
        api_key: OpenAI API klíč
        ticker: Symbol akcie
    
    Returns:
        Tuple (sentiment, summary)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        news_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(news_titles)])
        
        prompt = f"""Analyzuj následující novinky o akci {ticker} a poskytni:
1. Celkový sentiment (odpověz POUZE: Pozitivní NEBO Negativní NEBO Neutrální)
2. Krátké shrnutí (maximálně 2-3 věty)

Novinky:
{news_text}

Formát odpovědi:
SENTIMENT: [Pozitivní/Negativní/Neutrální]
SHRNUTÍ: [tvé shrnutí zde]
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Jsi finanční analytik specializující se na analýzu tržních zpráv."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        
        # Parsování odpovědi
        sentiment = "Neutrální"
        summary = "Analýza nedostupná"
        
        lines = result.split('\n')
        for line in lines:
            line_upper = line.upper()
            if 'SENTIMENT' in line_upper:
                if 'POZITIVNÍ' in line_upper or 'POSITIVE' in line_upper:
                    sentiment = "Pozitivní"
                elif 'NEGATIVNÍ' in line_upper or 'NEGATIVE' in line_upper:
                    sentiment = "Negativní"
                else:
                    sentiment = "Neutrální"
            elif 'SHRNUTÍ' in line_upper or 'SUMMARY' in line_upper:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    summary = parts[1].strip()
        
        if summary == "Analýza nedostupná" and len(result) > 20:
            summary_lines = [l for l in lines if 'SENTIMENT' not in l.upper() and l.strip()]
            if summary_lines:
                summary = " ".join(summary_lines[:3])
        
        return sentiment, summary
        
    except ImportError:
        return "Neutrální", "⚠️ Knihovna openai není nainstalována. Spusť: pip install openai"
    except Exception as e:
        return "Neutrální", f"⚠️ Chyba při AI analýze: {str(e)}"

# ============================================================================
# UTILITY FUNKCE
# ============================================================================

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Stáhne historická data akcie z Yahoo Finance
    
    Args:
        ticker: Symbol akcie (např. AAPL, TSLA)
        period: Časové období (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    
    Returns:
        DataFrame s historickými daty
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Chyba při stahování dat: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(ticker: str) -> Dict:
    """
    Získá fundamentální informace o akcii
    
    Args:
        ticker: Symbol akcie
    
    Returns:
        Slovník s informacemi o akcii
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        st.error(f"Chyba při získávání informací: {str(e)}")
        return {}

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """
    Vypočítá Relative Strength Index (RSI)
    
    Args:
        data: Cenová data
        periods: Počet period pro výpočet
    
    Returns:
        Series s RSI hodnotami
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Vypočítá Simple Moving Average (SMA)
    
    Args:
        data: Cenová data
        window: Velikost okna
    
    Returns:
        Series se SMA hodnotami
    """
    return data.rolling(window=window).mean()

def calculate_macd(data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vypočítá MACD indikátor
    
    Args:
        data: Cenová data
    
    Returns:
        Tuple (MACD line, Signal line, Histogram)
    """
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return macd, signal, histogram

def get_recommendation_color(recommendation: str) -> str:
    """Vrátí barvu podle doporučení"""
    rec_lower = recommendation.lower()
    if 'buy' in rec_lower or 'strong buy' in rec_lower:
        return '🟢'
    elif 'hold' in rec_lower:
        return '🟡'
    elif 'sell' in rec_lower:
        return '🔴'
    else:
        return '⚪'


def map_analyst_grade(grade: str) -> str:
    """Mapuje různé texty doporučení analytiků do 5 bucketů."""
    if not grade:
        return "unknown"
    g = str(grade).strip().lower()

    # Strong Buy / Strong Sell
    if "strong buy" in g:
        return "strong_buy"
    if "strong sell" in g:
        return "strong_sell"

    # Buy-ish
    buy_terms = [" buy", "buy", "overweight", "outperform", "positive", "accumulate", "add"]
    if any(t in g for t in buy_terms):
        return "buy"

    # Sell-ish
    sell_terms = [" sell", "sell", "underweight", "underperform", "negative", "reduce", "trim"]
    if any(t in g for t in sell_terms):
        return "sell"

    # Hold-ish
    hold_terms = ["hold", "neutral", "market perform", "equal-weight", "equal weight", "in-line", "inline"]
    if any(t in g for t in hold_terms):
        return "hold"

    return "unknown"


def get_analyst_consensus_counts(ticker_obj: yf.Ticker, lookback_days: int = 180) -> Dict[str, int]:
    """Vrátí počty doporučení analytiků podle posledních změn ratingů (yfinance recommendations)."""
    counts = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0}

    try:
        recs = getattr(ticker_obj, "recommendations", None)
        if recs is None or len(recs) == 0:
            return counts

        recs = recs.copy()
        # index bývá datetime
        if hasattr(recs.index, "to_pydatetime"):
            cutoff = datetime.now() - timedelta(days=lookback_days)
            recs = recs[recs.index >= cutoff]

        # yfinance obvykle: columns = ['Firm','To Grade','From Grade','Action']
        col_to = None
        for c in recs.columns:
            if str(c).lower() in ["to grade", "to_grade", "to"]:
                col_to = c
                break

        if col_to is None:
            return counts

        for grade in recs[col_to].dropna().astype(str).tolist():
            bucket = map_analyst_grade(grade)
            if bucket in counts:
                counts[bucket] += 1

        return counts
    except Exception:
        return counts
def calculate_financial_health_score(info: Dict) -> Tuple[str, str, int]:
    """
    Vypočítá skóre finanční zdraví společnosti
    
    Args:
        info: Slovník s informacemi o akcii
    
    Returns:
        Tuple (status, popis, skóre)
    """
    score = 0
    max_score = 5
    
    # 1. Debt to Equity ratio
    debt_to_equity = info.get('debtToEquity', None)
    if debt_to_equity is not None:
        if debt_to_equity < 50:
            score += 1
        elif debt_to_equity > 150:
            score -= 0.5
    
    # 2. Current Ratio
    current_ratio = info.get('currentRatio', None)
    if current_ratio is not None:
        if current_ratio > 1.5:
            score += 1
        elif current_ratio < 1:
            score -= 0.5
    
    # 3. Free Cash Flow
    free_cash_flow = info.get('freeCashflow', None)
    if free_cash_flow is not None and free_cash_flow > 0:
        score += 1
    
    # 4. Profit Margins
    profit_margin = info.get('profitMargins', None)
    if profit_margin is not None:
        if profit_margin > 0.15:
            score += 1
        elif profit_margin < 0:
            score -= 1
    
    # 5. ROE
    roe = info.get('returnOnEquity', None)
    if roe is not None:
        if roe > 0.15:
            score += 1
        elif roe < 0:
            score -= 0.5
    
    # Normalizace skóre
    score = max(0, min(score, max_score))
    percentage = (score / max_score) * 100
    
    if percentage >= 70:
        return "🟢 Silná", "Společnost má výborné finanční zdraví", int(percentage)
    elif percentage >= 40:
        return "🟡 Střední", "Společnost má průměrné finanční zdraví", int(percentage)
    else:
        return "🔴 Slabá", "Společnost má slabé finanční zdraví", int(percentage)

def get_fear_greed_index() -> Dict:
    """
    Získá Fear & Greed Index (simulovaný - v produkci použij CNN API)
    
    Returns:
        Slovník s indexem a popisem
    """
    import random
    value = random.randint(0, 100)
    
    if value >= 75:
        classification = "Extreme Greed"
        color = "🔴"
    elif value >= 55:
        classification = "Greed"
        color = "🟠"
    elif value >= 45:
        classification = "Neutral"
        color = "🟡"
    elif value >= 25:
        classification = "Fear"
        color = "🔵"
    else:
        classification = "Extreme Fear"
        color = "🟢"
    
    return {
        "value": value,
        "classification": classification,
        "color": color
    }

def analyze_valuation(info: Dict) -> Tuple[str, str]:
    """
    Analyzuje, zda je akcie podhodnocená, nadhodnocená nebo férově oceněná
    
    Args:
        info: Informace o akcii
    
    Returns:
        Tuple (status, důvod)
    """
    signals = []
    
    # P/E ratio analýza
    pe_ratio = info.get('trailingPE', None)
    forward_pe = info.get('forwardPE', None)
    industry_pe = 20  # Průměrné P/E pro trh
    
    if pe_ratio:
        if pe_ratio < industry_pe * 0.8:
            signals.append(("undervalued", f"P/E ratio ({pe_ratio:.2f}) je pod průměrem trhu"))
        elif pe_ratio > industry_pe * 1.3:
            signals.append(("overvalued", f"P/E ratio ({pe_ratio:.2f}) je nad průměrem trhu"))
    
    # PEG ratio
    peg_ratio = info.get('pegRatio', None)
    if peg_ratio:
        if peg_ratio < 1:
            signals.append(("undervalued", f"PEG ratio ({peg_ratio:.2f}) < 1 indikuje podhodnocení"))
        elif peg_ratio > 2:
            signals.append(("overvalued", f"PEG ratio ({peg_ratio:.2f}) > 2 indikuje nadhodnocení"))
    
    # Price to Book
    price_to_book = info.get('priceToBook', None)
    if price_to_book:
        if price_to_book < 1:
            signals.append(("undervalued", f"Price-to-Book ({price_to_book:.2f}) < 1"))
        elif price_to_book > 5:
            signals.append(("overvalued", f"Price-to-Book ({price_to_book:.2f}) je velmi vysoký"))
    
    # Vyhodnocení
    undervalued_count = sum(1 for s in signals if s[0] == "undervalued")
    overvalued_count = sum(1 for s in signals if s[0] == "overvalued")
    
    if undervalued_count > overvalued_count:
        status = "🟢 PODHODNOCENÁ"
        reasons = "\n".join([f"• {s[1]}" for s in signals if s[0] == "undervalued"])
        return status, reasons
    elif overvalued_count > undervalued_count:
        status = "🔴 NADHODNOCENÁ"
        reasons = "\n".join([f"• {s[1]}" for s in signals if s[0] == "overvalued"])
        return status, reasons
    else:
        status = "🟡 FÉROVĚ OCENĚNÁ"
        return status, "Valuační metriky jsou v průměru"

def get_news_sentiment_simple(ticker: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Jednoduchá sentiment analýza novinek (bez AI).

    Vrací seznam položek s titulkem a (pokud je dostupné) i URL/publisherem,
    aby šlo zprávy rovnou prokliknout.

    Returns:
        Tuple (sentiment, news_items)
    """
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news if hasattr(stock, "news") else []
        raw_news = raw_news[:10] if raw_news else []

        if not raw_news:
            return "Neutrální", []

        news_items = [extract_news_meta(item) for item in raw_news]
        titles = [ni.get("title", "") for ni in news_items]

        # Jednoduchá keyword analýza (fallback)
        positive_words = ["up", "gain", "profit", "growth", "success", "beat", "surge", "rally", "rise", "record", "upgrade"]
        negative_words = ["down", "loss", "fall", "decline", "miss", "drop", "plunge", "crash", "fear", "downgrade", "lawsuit"]

        positive_count = sum(1 for title in titles for word in positive_words if word in str(title).lower())
        negative_count = sum(1 for title in titles for word in negative_words if word in str(title).lower())

        if positive_count > negative_count:
            sentiment = "Pozitivní"
        elif negative_count > positive_count:
            sentiment = "Negativní"
        else:
            sentiment = "Neutrální"

        return sentiment, news_items
    except Exception:
        return "Neutrální", []


def get_news_sentiment_ai(ticker: str, ai_provider: str, api_key: str) -> Tuple[str, str, List[str]]:
    try:
        stock = yf.Ticker(ticker)
        # Získání novinek s pojistkou
        raw_news = stock.news if hasattr(stock, 'news') and stock.news else []
        
        if not raw_news:
            return "Neutrální", "Žádné novinky k dispozici na Yahoo Finance.", []
        
        # OPRAVA: Yahoo/yfinance mění strukturu - bereme titulky i z vnořených polí
        def _extract_title(it):
            if isinstance(it, str):
                return it.strip() if it.strip() else None
            if not isinstance(it, dict):
                return None

            t = it.get("title") or it.get("headline") or it.get("text") or it.get("summary")
            if t and str(t).strip():
                return str(t).strip()

            content = it.get("content")
            if isinstance(content, dict):
                t2 = content.get("title") or content.get("headline") or content.get("summary") or content.get("description")
                if t2 and str(t2).strip():
                    return str(t2).strip()

            return None

        news_titles = []
        for item in raw_news:
            t = _extract_title(item)
            if t and t.lower() != "bez názvu":
                news_titles.append(t)

        if not news_titles:
            return "Neutrální", "Novinky nalezeny, ale nepodařilo se extrahovat jejich titulky.", []
        
        # Omezení na 5 zpráv pro AI
        news_to_analyze = news_titles[:5]
        
        if ai_provider == "Google Gemini":
            sentiment, summary = analyze_news_with_gemini(news_to_analyze, api_key, ticker)
        elif ai_provider == "OpenAI":
            sentiment, summary = analyze_news_with_openai(news_to_analyze, api_key, ticker)
        else:
            sentiment, summary = "Neutrální", "Nepodporovaný AI poskytovatel", []
        
        return sentiment, summary, news_titles
    
    except Exception as e:
        return "Neutrální", f"Chyba při přípravě dat: {str(e)}", []

# ============================================================================
# GRAFY
# ============================================================================

def create_price_chart(df: pd.DataFrame, ticker: str, show_sma: bool, show_rsi: bool, show_volume: bool):
    """
    Vytvoří interaktivní graf ceny s indikátory
    
    Args:
        df: DataFrame s historickými daty
        ticker: Symbol akcie
        show_sma: Zobrazit klouzavé průměry
        show_rsi: Zobrazit RSI
        show_volume: Zobrazit objem
    """
    if df.empty:
        st.warning("Žádná data k zobrazení")
        return
    
    # Vytvoření subplotů
    rows = 1
    row_heights = [0.7]
    
    if show_rsi:
        rows += 1
        row_heights.append(0.15)
    if show_volume:
        rows += 1
        row_heights.append(0.15)
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=['Cena', 'RSI' if show_rsi else '', 'Objem' if show_volume else '']
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Cena'
        ),
        row=1, col=1
    )
    
    # SMA
    if show_sma:
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['SMA_200'] = calculate_sma(df['Close'], 200)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    current_row = 1
    
    # RSI
    if show_rsi:
        current_row += 1
        df['RSI'] = calculate_rsi(df['Close'])
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
            row=current_row, col=1
        )
        
        # Přidání pásem překoupenosti/přeprodanosti
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
    
    # Volume
    if show_volume:
        current_row += 1
        colors = ['red' if df['Close'][i] < df['Open'][i] else 'green' for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Objem', marker_color=colors, opacity=0.5),
            row=current_row, col=1
        )
        
        fig.update_yaxes(title_text="Objem", row=current_row, col=1)
    
    # Layout
    fig.update_layout(
        title=f'{ticker} - Technická analýza',
        yaxis_title='Cena (USD)',
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# HLAVNÍ APLIKACE
# ============================================================================


# =============================
# Investor metrics helpers
# =============================

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def _latest_col(df: pd.DataFrame):
    if df is None or getattr(df, "empty", True):
        return None
    return df.columns[0]

def get_income_statement_metrics(stock):
    """Return key income statement metrics (latest annual if available)."""
    try:
        inc = getattr(stock, "financials", None)
        if inc is None or inc.empty:
            return {}
        col = _latest_col(inc)
        series = inc[col]
        rev = _safe_float(series.get("Total Revenue")) if "Total Revenue" in inc.index else None
        gross = _safe_float(series.get("Gross Profit")) if "Gross Profit" in inc.index else None
        op = _safe_float(series.get("Operating Income")) if "Operating Income" in inc.index else None
        net = _safe_float(series.get("Net Income")) if "Net Income" in inc.index else None
        out = {"revenue": rev, "gross_profit": gross, "operating_income": op, "net_income": net}
        if rev and rev != 0:
            out["gross_margin"] = (gross / rev) if gross is not None else None
            out["operating_margin"] = (op / rev) if op is not None else None
            out["net_margin"] = (net / rev) if net is not None else None
        return out
    except Exception:
        return {}

def get_cashflow_metrics(stock):
    """Return latest annual CFO/CapEx/FCF if available."""
    try:
        cf = getattr(stock, "cashflow", None)
        if cf is None or cf.empty:
            return {}
        col = _latest_col(cf)
        series = cf[col]
        cfo = _safe_float(series.get("Total Cash From Operating Activities")) if "Total Cash From Operating Activities" in cf.index else None
        capex = _safe_float(series.get("Capital Expenditures")) if "Capital Expenditures" in cf.index else None
        fcf = _safe_float(series.get("Free Cash Flow")) if "Free Cash Flow" in cf.index else None
        if fcf is None and cfo is not None and capex is not None:
            fcf = cfo + capex  # capex is typically negative in yfinance
        return {"cfo": cfo, "capex": capex, "fcf": fcf}
    except Exception:
        return {}

def get_balance_sheet_metrics(stock):
    """Return cash, debt, net debt and simple liquidity ratios if possible."""
    try:
        bs = getattr(stock, "balance_sheet", None)
        if bs is None or bs.empty:
            return {}
        col = _latest_col(bs)
        series = bs[col]
        cash = _safe_float(series.get("Cash And Cash Equivalents")) if "Cash And Cash Equivalents" in bs.index else None
        debt = _safe_float(series.get("Total Debt")) if "Total Debt" in bs.index else None
        cur_assets = _safe_float(series.get("Total Current Assets")) if "Total Current Assets" in bs.index else None
        cur_liab = _safe_float(series.get("Total Current Liabilities")) if "Total Current Liabilities" in bs.index else None
        out = {"cash": cash, "debt": debt}
        if cash is not None and debt is not None:
            out["net_debt"] = debt - cash
        if cur_assets is not None and cur_liab not in (None, 0):
            out["current_ratio"] = cur_assets / cur_liab
        return out
    except Exception:
        return {}

def calc_volatility_and_drawdown(df: pd.DataFrame):
    """Compute annualized vol and max drawdown from close prices."""
    try:
        if df is None or df.empty or "Close" not in df.columns:
            return {}
        rets = df["Close"].pct_change().dropna()
        if rets.empty:
            return {}
        vol = float(rets.std() * np.sqrt(252))
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum / peak) - 1.0
        max_dd = float(dd.min())
        return {"volatility": vol, "max_drawdown": max_dd}
    except Exception:
        return {}

def simple_scorecard(info: dict, income: dict, cf: dict, bs: dict, risk: dict, sentiment_label: str | None):
    """Heuristic 0-100 scorecard to guide analysis (not financial advice)."""
    scores = {"Valuation": 50, "Quality": 50, "Growth": 50, "Health": 50, "Risk": 50, "Sentiment": 50}
    gm = income.get("gross_margin")
    om = income.get("operating_margin")
    if gm is not None:
        scores["Quality"] += 10 if gm > 0.4 else (5 if gm > 0.25 else -5)
    if om is not None:
        scores["Quality"] += 10 if om > 0.2 else (5 if om > 0.1 else -5)

    net_debt = bs.get("net_debt")
    if net_debt is not None:
        scores["Health"] += 8 if net_debt < 0 else (-8 if net_debt > 0 else 0)
    cr = bs.get("current_ratio")
    if cr is not None:
        scores["Health"] += 6 if cr >= 1.5 else (0 if cr >= 1.0 else -6)

    if income.get("revenue") is not None:
        scores["Growth"] += 4
    if cf.get("fcf") is not None:
        scores["Growth"] += 8 if cf["fcf"] > 0 else -8

    mc = _safe_float(info.get("marketCap"))
    fcf = cf.get("fcf")
    if mc and fcf:
        fcf_yield = fcf / mc
        scores["Valuation"] += 12 if fcf_yield >= 0.05 else (4 if fcf_yield >= 0.03 else -8)

    vol = risk.get("volatility")
    if vol is not None:
        scores["Risk"] += 6 if vol < 0.25 else (-6 if vol > 0.45 else 0)
    mdd = risk.get("max_drawdown")
    if mdd is not None:
        scores["Risk"] += 6 if mdd > -0.35 else (-6 if mdd < -0.6 else 0)

    if sentiment_label:
        if sentiment_label.upper().startswith("POZ"):
            scores["Sentiment"] += 10
        elif sentiment_label.upper().startswith("NEG"):
            scores["Sentiment"] -= 10

    for k in list(scores.keys()):
        scores[k] = int(max(0, min(100, scores[k])))
    total = int(round(sum(scores.values()) / len(scores)))
    return scores, total

def scenario_fair_value(info: dict, income: dict, years: int, growth: float, fcf_margin: float, exit_fcf_multiple: float, discount: float):
    """Simple scenario model (terminal FCF multiple)."""
    mc = _safe_float(info.get("marketCap"))
    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    shares = _safe_float(info.get("sharesOutstanding"))
    rev = income.get("revenue")
    if rev is None:
        return None
    rev_t = rev * ((1 + growth) ** years)
    fcf_t = rev_t * fcf_margin
    terminal_value = fcf_t * exit_fcf_multiple
    pv = terminal_value / ((1 + discount) ** years)
    if shares and shares > 0:
        fair_price = pv / shares
    elif mc and price:
        est_shares = mc / price
        fair_price = pv / est_shares if est_shares else None
    else:
        fair_price = None
    return {"fair_price": fair_price, "pv_terminal": pv, "rev_t": rev_t, "fcf_t": fcf_t}


def main():
    """Hlavní funkce aplikace"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
        st.title("⚙️ Nastavení")
        
        ticker = st.text_input(
            "📊 Ticker symbol",
            value="AAPL",
            help="Zadej symbol akcie (např. AAPL, TSLA, MSFT, CEZ.PR)"
        ).upper()
        
        period = st.selectbox(
            "📅 Časové období",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
        
        st.markdown("---")
        st.subheader("📊 Indikátory")
        
        show_sma = st.checkbox("Klouzavé průměry (SMA)", value=True)
        show_rsi = st.checkbox("RSI indikátor", value=True)
        show_volume = st.checkbox("Objem obchodů", value=True)
        show_macd = st.checkbox("MACD", value=False)
        
        st.markdown("---")
        st.subheader("🤖 AI Analýza")
        
        enable_ai = st.checkbox("Povolit AI analýzu novinek", value=False)
        
        api_key = None
        ai_provider = None
        
        if enable_ai:
            ai_provider = st.radio("Zvolte poskytovatele AI:", ["Google Gemini", "OpenAI"])
            api_key = st.text_input("API klíč", type="password", help="Zadej svůj API klíč")
            
            if ai_provider == "Google Gemini":
                st.info("💡 Získej API klíč zdarma na: https://makersuite.google.com/app/apikey")
            else:
                st.info("💡 Získej API klíč na: https://platform.openai.com/api-keys")
        
        st.markdown("---")
        st.info("💡 **Tip:** Klikněte na graf pro detailní pohled")
        
        analyze_button = st.button("🔍 ANALYZOVAT AKCII", type="primary", use_container_width=True)
    
    # Main content
    if analyze_button or ticker:
        with st.spinner(f"Načítám data pro {ticker}..."):
            # Získání dat
            df = get_stock_data(ticker, period)
            info = get_stock_info(ticker)
            
            if df.empty or not info:
                st.error(f"❌ Nepodařilo se načíst data pro ticker {ticker}. Zkontrolujte, zda je symbol správný.")
                return
            # yfinance objekt pro funkce, které potřebují `.recommendations` / `.news` apod.
            try:
                stock = yf.Ticker(ticker)
            except Exception:
                stock = None

            
            # Company info
            company_name = info.get('longName', ticker)
            current_price = info.get('currentPrice', df['Close'].iloc[-1] if not df.empty else 0)
            previous_close = info.get('previousClose', 0)
            
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0
            
            # Header metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="🏢 Společnost",
                    value=company_name
                )
            
            with col2:
                st.metric(
                    label="💰 Aktuální cena",
                    value=f"${current_price:.2f}",
                    delta=f"{change_percent:+.2f}%"
                )
            
            with col3:
                fair_value, fv_method = estimate_fair_value(info)
                if fair_value is not None:
                    upside = ((fair_value / current_price) - 1) * 100 if current_price else 0
                    st.metric(
                        label="🎯 Férová cena",
                        value=f"${fair_value:.2f}",
                        delta=f"{upside:+.1f}% vs cena"
                    )
                    st.caption(f"Metoda: {fv_method}")
                else:
                    st.metric(label="🎯 Férová cena", value="N/A")
                    st.caption("Není k dispozici (chybí target price nebo FCF/shares).")

            with col4:
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    market_cap_formatted = f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_formatted = "N/A"
                st.metric(
                    label="📊 Market Cap",
                    value=market_cap_formatted
                )

            with col5:
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', 1)
                volume_ratio = (volume / avg_volume * 100) if avg_volume != 0 else 0
                st.metric(
                    label="📈 Objem",
                    value=f"{volume/1e6:.2f}M",
                    delta=f"{volume_ratio-100:+.1f}% vs průměr"
                )
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📈 Technická analýza",
                "📊 Fundamentální data",
                "🎯 Doporučení analytiků",
                "📰 Novinky & Sentiment",
                "💼 Insider Trading",
                "🏥 Finanční zdraví",
                "🧾 Investor Dashboard"
            ])
            
            # TAB 1: Technická analýza
            with tab1:
                st.subheader("📈 Cenový graf s indikátory")
                create_price_chart(df, ticker, show_sma, show_rsi, show_volume)
                
                # MACD
                if show_macd and not df.empty:
                    st.subheader("📊 MACD Indikátor")
                    macd, signal, histogram = calculate_macd(df['Close'])
                    
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red')))
                    fig_macd.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color='gray', opacity=0.5))
                    
                    fig_macd.update_layout(
                        title='MACD',
                        height=300,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Technické signály
                st.subheader("🎯 Technické signály")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if not df.empty and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                        current_price_val = df['Close'].iloc[-1]
                        sma50 = df['SMA_50'].iloc[-1]
                        sma200 = df['SMA_200'].iloc[-1]
                        
                        if pd.notna(sma50) and pd.notna(sma200):
                            if sma50 > sma200:
                                signal = "🟢 Bullish (Golden Cross)"
                            else:
                                signal = "🔴 Bearish (Death Cross)"
                        else:
                            signal = "⚪ Nedostatek dat"
                    else:
                        signal = "⚪ Nedostatek dat"
                    
                    st.info(f"**SMA Cross:**\n\n{signal}")
                
                with col2:
                    if not df.empty and 'RSI' in df.columns:
                        rsi_current = df['RSI'].iloc[-1]
                        if pd.notna(rsi_current):
                            if rsi_current > 70:
                                rsi_signal = f"🔴 Překoupeno ({rsi_current:.1f})"
                            elif rsi_current < 30:
                                rsi_signal = f"🟢 Přeprodáno ({rsi_current:.1f})"
                            else:
                                rsi_signal = f"🟡 Neutrální ({rsi_current:.1f})"
                        else:
                            rsi_signal = "⚪ Nedostatek dat"
                    else:
                        rsi_signal = "⚪ Nedostatek dat"
                    
                    st.info(f"**RSI:**\n\n{rsi_signal}")
                
                with col3:
                    if not df.empty:
                        price_change_5d = ((df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100) if len(df) >= 5 else 0
                        if price_change_5d > 5:
                            trend = f"🟢 Silný vzestup (+{price_change_5d:.1f}%)"
                        elif price_change_5d < -5:
                            trend = f"🔴 Silný pokles ({price_change_5d:.1f}%)"
                        else:
                            trend = f"🟡 Konsolidace ({price_change_5d:+.1f}%)"
                    else:
                        trend = "⚪ Nedostatek dat"
                    
                    st.info(f"**Trend (5D):**\n\n{trend}")
            
            # TAB 2: Fundamentální data
            with tab2:
                st.subheader("📊 Klíčové finanční ukazatele")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💹 Valuační metriky")
                    
                    metrics_data = {
                        "P/E Ratio (Trailing)": info.get('trailingPE', 'N/A'),
                        "Forward P/E": info.get('forwardPE', 'N/A'),
                        "PEG Ratio": info.get('pegRatio', 'N/A'),
                        "Price-to-Book": info.get('priceToBook', 'N/A'),
                        "Price-to-Sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                        "Enterprise Value": f"${info.get('enterpriseValue', 0)/1e9:.2f}B" if info.get('enterpriseValue') else 'N/A',
                    }
                    
                    df_metrics = pd.DataFrame(list(metrics_data.items()), columns=['Metrika', 'Hodnota'])
                    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("### 💰 Ziskovost & Výnosy")
                    
                    profitability_data = {
                        "EPS (Trailing)": f"${info.get('trailingEps', 'N/A')}",
                        "Forward EPS": f"${info.get('forwardEps', 'N/A')}",
                        "Profit Margin": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
                        "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A',
                        "Return on Equity": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                        "Return on Assets": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                    }
                    
                    df_profitability = pd.DataFrame(list(profitability_data.items()), columns=['Metrika', 'Hodnota'])
                    st.dataframe(df_profitability, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("### 📈 Dividendy")
                    
                    dividend_yield = info.get('dividendYield', 0)
                    dividend_rate = info.get('dividendRate', 0)
                    payout_ratio = info.get('payoutRatio', 0)
                    
                    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "Žádné dividendy")
                    st.metric("Annual Dividend", f"${dividend_rate:.2f}" if dividend_rate else "N/A")
                    st.metric("Payout Ratio", f"{payout_ratio*100:.2f}%" if payout_ratio else "N/A")
                
                with col4:
                    st.markdown("### 🏦 Zadluženost")
                    
                    debt_to_equity = info.get('debtToEquity', 0)
                    total_debt = info.get('totalDebt', 0)
                    total_cash = info.get('totalCash', 0)
                    
                    st.metric("Debt-to-Equity", f"{debt_to_equity:.2f}" if debt_to_equity else "N/A")
                    st.metric("Total Debt", f"${total_debt/1e9:.2f}B" if total_debt else "N/A")
                    st.metric("Total Cash", f"${total_cash/1e9:.2f}B" if total_cash else "N/A")
                
                # Analýza ocenění
                st.markdown("---")
                st.subheader("🎯 Analýza ocenění")
                
                valuation_status, valuation_reason = analyze_valuation(info)
                
                if "PODHODNOCENÁ" in valuation_status:
                    st.success(f"## {valuation_status}")
                    st.markdown(f"**Důvody:**\n{valuation_reason}")
                    st.info("💡 **Indikace:** Akcie může představovat nákupní příležitost, ale vždy proveďte další analýzu.")
                elif "NADHODNOCENÁ" in valuation_status:
                    st.error(f"## {valuation_status}")
                    st.markdown(f"**Důvody:**\n{valuation_reason}")
                    st.warning("⚠️ **Indikace:** Akcie může být drahá. Zvažte čekání na lepší cenu.")
                else:
                    st.info(f"## {valuation_status}")
                    st.markdown(f"**Analýza:** {valuation_reason}")
            
            # TAB 3: Doporučení analytiků
            with tab3:
                st.subheader("🎯 Doporučení od analytiků")
                
                target_price = info.get('targetMeanPrice', None)
                current_price_val = info.get('currentPrice', current_price)
                recommendation = info.get('recommendationKey', 'N/A')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if target_price:
                        upside = ((target_price - current_price_val) / current_price_val * 100)
                        st.metric(
                            "🎯 Cílová cena (průměr)",
                            f"${target_price:.2f}",
                            delta=f"{upside:+.2f}% potenciál"
                        )
                    else:
                        st.metric("🎯 Cílová cena", "N/A")
                
                with col2:
                    target_high = info.get('targetHighPrice', None)
                    target_low = info.get('targetLowPrice', None)
                    
                    if target_high and target_low:
                        st.metric("📊 Cílový rozsah", f"${target_low:.2f} - ${target_high:.2f}")
                    else:
                        st.metric("📊 Cílový rozsah", "N/A")
                
                with col3:
                    recommendation_display = recommendation.upper() if recommendation != 'N/A' else 'N/A'
                    color = get_recommendation_color(recommendation_display)
                    st.metric("📋 Doporučení", f"{color} {recommendation_display}")
                
                # Počet analytiků
                st.markdown("---")
                st.markdown("### 👥 Konsenzus analytiků")

                # numberOfAnalystOpinions = počet analytiků započítaných do cílové ceny / konsenzu v Yahoo datech
                num_analysts = int(info.get('numberOfAnalystOpinions') or 0)

                # Rozpad doporučení (počítáno z posledních změn ratingů v yfinance `ticker.recommendations`)
                consensus_counts = get_analyst_consensus_counts(stock, lookback_days=180) if stock is not None else {}
                total_actions = sum(consensus_counts.values())

                if num_analysts > 0:
                    st.info(f"**Počet analytiků (k cílové ceně):** {num_analysts}")
                else:
                    st.info("**Počet analytiků (k cílové ceně):** N/A")

                # Souhrn doporučení z `info` (Yahoo)
                rec_key = info.get("recommendationKey") or "N/A"
                rec_mean = info.get("recommendationMean", None)

                colA, colB = st.columns(2)
                with colA:
                    st.metric("📌 Doporučení (průměr)", str(rec_key).upper() if rec_key != "N/A" else "N/A")
                with colB:
                    if rec_mean is not None:
                        try:
                            st.metric("📏 Doporučení (mean)", f"{float(rec_mean):.2f}")
                        except Exception:
                            st.metric("📏 Doporučení (mean)", str(rec_mean))
                    else:
                        st.metric("📏 Doporučení (mean)", "N/A")

                # Poslední změny ratingů (yfinance recommendations) – nemusí se rovnat počtu analytiků v konsenzu
                if total_actions > 0:
                    st.caption("Níže je přehled posledních změn ratingů (cca posledních 180 dní) z yfinance. Nejde o plný rozpad konsenzu (Yahoo ten často neposkytuje přes API).")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Strong Buy", consensus_counts["strong_buy"])
                    with col2:
                        st.metric("Buy", consensus_counts["buy"])
                    with col3:
                        st.metric("Hold", consensus_counts["hold"])
                    with col4:
                        st.metric("Sell", consensus_counts["sell"])
                    with col5:
                        st.metric("Strong Sell", consensus_counts["strong_sell"])
                else:
                    st.info("Rozpad doporučení (Strong Buy/Buy/Hold/Sell) se přes yfinance často nevrací. Zobrazuji alespoň průměrné doporučení z Yahoo (recommendationKey/Mean).")

            
            # TAB 4: Novinky & Sentiment
            with tab4:
                st.subheader("📰 Poslední novinky a sentiment analýza")

                # AI nebo jednoduchá analýza
                if enable_ai and api_key:
                    st.success("🤖 AI analýza novinek je AKTIVNÍ")

                    with st.spinner("Analyzuji novinky pomocí AI..."):
                        sentiment, summary, news_list = get_news_sentiment_ai(ticker, ai_provider, api_key)
                        st.session_state['news_sentiment_label'] = sentiment

                    # Zobrazení AI shrnutí
                    st.markdown("### 🤖 AI Shrnutí")
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if sentiment == "Pozitivní":
                            st.markdown('<div class="sentiment-positive">😊 POZITIVNÍ</div>', unsafe_allow_html=True)
                        elif sentiment == "Negativní":
                            st.markdown('<div class="sentiment-negative">😟 NEGATIVNÍ</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="sentiment-neutral">😐 NEUTRÁLNÍ</div>', unsafe_allow_html=True)

                    with col2:
                        st.write(summary)

                else:
                    if enable_ai and not api_key:
                        st.warning("⚠️ Pro AI analýzu zadej API klíč v levém panelu")

                    # get_news_sentiment_simple vrací jen (sentiment, news_list)
                    sentiment, news_list = get_news_sentiment_simple(ticker)
                    summary = "AI analýza není aktivní."

                # SPOLEČNÁ ČÁST PRO VÝPIS ZPRÁV
                st.markdown("---")
                st.markdown("### 📋 Poslední zprávy")

                if not news_list:
                    st.info("Pro tento ticker nejsou momentálně dostupné žádné čitelné zprávy.")
                else:
                    for i, item in enumerate(news_list, 1):
                        # news_list může být List[Dict] (preferované) nebo List[str] (legacy)
                        meta = extract_news_meta(item)
                        title = meta.get("title") or f"Zpráva č. {i}"
                        publisher = meta.get("publisher") or "Neznámý zdroj"
                        url = meta.get("url")

                        with st.expander(f"📰 {title}"):
                            st.write(f"**Zdroj:** {publisher}")
                            if url:
                                st.markdown(f"🔗 [Otevřít článek]({url})")
                            else:
                                q = quote_plus(f"{title} {publisher or ''}")
                                st.markdown(f"[🔎 Vyhledat článek na webu](https://www.google.com/search?q={q})")
                                st.caption("Přímý odkaz Yahoo někdy neposílá – hledání použije nadpis + zdroj.")
                            st.caption(f"Pořadí: #{i}")

                # Market context
                st.markdown("---")
                st.subheader("🌍 Kontext trhu")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📊 S&P 500")
                    spy_data = get_stock_data("^GSPC", "5d")
                    if not spy_data.empty:
                        spy_change = ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100)
                        st.metric("5denní změna", f"{spy_change:+.2f}%")
                    else:
                        st.warning("Data S&P 500 nejsou k dispozici")

                with col2:
                    st.markdown("### 😱 Fear & Greed Index")
                    fg_index = get_fear_greed_index()
                    st.metric(
                        fg_index['classification'],
                        f"{fg_index['color']} {fg_index['value']}/100"
                    )
            # TAB 5: Insider Trading
            with tab5:
                st.subheader("💼 Insider Trading - Transakce vedení společnosti")

                st.caption(
                    "Pozn.: Yahoo/yfinance často vrací kombinaci prodejů, nákupů i akciových grantů (RSU/Stock Award). "
                    "Granty nejsou nákup na trhu – proto je oddělujeme."
                )

                try:
                    stock = yf.Ticker(ticker)
                    insider_trades = stock.insider_transactions

                    if insider_trades is None or insider_trades.empty:
                        st.info("📊 Data o insider trading nejsou k dispozici pro tento ticker.")
                    else:
                        df_it = insider_trades.copy()

                        # Normalize columns
                        for c in ["Insider", "Position", "Text", "Transaction", "URL", "Start Date", "Value", "Shares", "Ownership"]:
                            if c not in df_it.columns:
                                df_it[c] = None

                        # Classify action using both Transaction and Text
                        df_it["Action"] = df_it.apply(
                            lambda r: classify_insider_action(
                                str(r.get("Text", "") or ""),
                                str(r.get("Transaction", "") or "")
                            ),
                            axis=1
                        )

                        # Ensure numeric
                        df_it["Shares_num"] = pd.to_numeric(df_it.get("Shares", 0), errors="coerce").fillna(0)
                        df_it["Value_num"] = pd.to_numeric(df_it.get("Value", 0), errors="coerce").fillna(0)

                        # Who buys / sells (top insiders)
                        buys = df_it[df_it["Action"] == "BUY"]
                        sells = df_it[df_it["Action"] == "SELL"]
                        grants = df_it[df_it["Action"] == "GRANT"]

                        colA, colB, colC, colD = st.columns(4)
                        with colA:
                            st.metric("🟢 Nákupy (počet transakcí)", int(len(buys)))
                        with colB:
                            st.metric("🔴 Prodeje (počet transakcí)", int(len(sells)))
                        with colC:
                            st.metric("🟣 Granty/RSU (počet)", int(len(grants)))
                        with colD:
                            # simple sentiment: buys vs sells (ignore grants)
                            if len(buys) > len(sells):
                                sentiment_insider = "🟢 Pozitivní"
                            elif len(sells) > len(buys):
                                sentiment_insider = "🔴 Negativní"
                            else:
                                sentiment_insider = "🟡 Neutrální"
                            st.metric("Sentiment (buy vs sell)", sentiment_insider)

                        st.markdown("### 👤 Kdo kupuje / prodává")
                        col1, col2 = st.columns(2)

                        with col1:
                            if not buys.empty:
                                top_buy = (buys.groupby("Insider")["Shares_num"].sum().sort_values(ascending=False).head(10))
                                st.write("**Top nákupci (akcie):**")
                                st.dataframe(top_buy.reset_index().rename(columns={"Shares_num": "Shares"}), use_container_width=True)
                            else:
                                st.info("Žádné nákupy v dostupných datech (může jít jen o granty/prodeje).")

                        with col2:
                            if not sells.empty:
                                top_sell = (sells.groupby("Insider")["Shares_num"].sum().sort_values(ascending=False).head(10))
                                st.write("**Top prodejci (akcie):**")
                                st.dataframe(top_sell.reset_index().rename(columns={"Shares_num": "Shares"}), use_container_width=True)
                            else:
                                st.info("Žádné prodeje v dostupných datech.")

                        st.markdown("### 📊 Celkový rozpis")
                        summary = (
                            df_it.groupby("Action")
                            .agg(
                                Transactions=("Action", "count"),
                                Shares=("Shares_num", "sum"),
                                Value=("Value_num", "sum"),
                            )
                            .reset_index()
                            .sort_values("Transactions", ascending=False)
                        )
                        st.dataframe(summary, use_container_width=True)
                        st.caption("Value bývá u části záznamů 0/None, protože Yahoo někdy neposílá hodnotu transakce. "
                                   "V takovém případě ji nelze spolehlivě dopočítat bez dalších dat (např. cena z formuláře SEC).")

                        st.markdown("### 🧾 Detailní transakce")
                        # Prettier ordering + keep key cols
                        show_cols = ["Start Date", "Insider", "Position", "Action", "Shares", "Value", "Text", "URL", "Ownership"]
                        show_cols = [c for c in show_cols if c in df_it.columns]
                        st.dataframe(df_it[show_cols].head(50), use_container_width=True)

                except Exception as e:
                    st.warning(f"⚠️ Nepodařilo se načíst data o insider trading: {str(e)}")

            # TAB 6: Finanční zdraví
            with tab6:
                st.subheader("🏥 Analýza finančního zdraví společnosti")
                
                health_status, health_desc, health_score = calculate_financial_health_score(info)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"## {health_status}")
                    st.progress(health_score / 100)
                    st.markdown(f"**Skóre:** {health_score}/100")
                
                with col2:
                    st.markdown(f"### {health_desc}")
                    st.markdown("""
                    **Hodnocení vychází z:**
                    - 📊 Debt-to-Equity ratio
                    - 💰 Current Ratio (likvidita)
                    - 💵 Free Cash Flow
                    - 📈 Profit Margins
                    - 🎯 Return on Equity
                    """)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📊 Zadluženost")
                    debt_to_equity = info.get('debtToEquity', 0)
                    
                    if debt_to_equity:
                        if debt_to_equity < 50:
                            st.success(f"🟢 Nízká: {debt_to_equity:.2f}")
                        elif debt_to_equity < 150:
                            st.warning(f"🟡 Střední: {debt_to_equity:.2f}")
                        else:
                            st.error(f"🔴 Vysoká: {debt_to_equity:.2f}")
                    else:
                        st.info("N/A")
                
                with col2:
                    st.markdown("### 💰 Likvidita")
                    current_ratio = info.get('currentRatio', 0)
                    
                    if current_ratio:
                        if current_ratio > 1.5:
                            st.success(f"🟢 Dobrá: {current_ratio:.2f}")
                        elif current_ratio > 1:
                            st.warning(f"🟡 Průměrná: {current_ratio:.2f}")
                        else:
                            st.error(f"🔴 Slabá: {current_ratio:.2f}")
                    else:
                        st.info("N/A")
                
                with col3:
                    st.markdown("### 📈 Ziskovost")
                    profit_margin = info.get('profitMargins', 0)
                    
                    if profit_margin:
                        profit_margin_pct = profit_margin * 100
                        if profit_margin_pct > 15:
                            st.success(f"🟢 Vysoká: {profit_margin_pct:.2f}%")
                        elif profit_margin_pct > 5:
                            st.warning(f"🟡 Střední: {profit_margin_pct:.2f}%")
                        else:
                            st.error(f"🔴 Nízká: {profit_margin_pct:.2f}%")
                    else:
                        st.info("N/A")
                
                st.markdown("---")
                st.markdown("### 💵 Cash Flow analýza")
                
                free_cash_flow = info.get('freeCashflow', 0)
                operating_cash_flow = info.get('operatingCashflow', 0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if free_cash_flow:
                        st.metric(
                            "Free Cash Flow",
                            f"${free_cash_flow/1e9:.2f}B",
                            delta="Pozitivní" if free_cash_flow > 0 else "Negativní"
                        )
                    else:
                        st.metric("Free Cash Flow", "N/A")
                
                with col2:
                    if operating_cash_flow:
                        st.metric(
                            "Operating Cash Flow",
                            f"${operating_cash_flow/1e9:.2f}B"
                        )
                    else:
                        st.metric("Operating Cash Flow", "N/A")
            

            # TAB 7: Investor Dashboard
            with tab7:
                st.subheader("🧾 Investor Dashboard")
                st.caption("Souhrn metrik, které investoři běžně sledují (valuace, kvalita, růst, zdraví firmy, riziko). "
                           "Nejde o investiční doporučení – ber to jako strukturovaný checklist.")

                # Základní data z yfinance (odolné proti výpadkům)
                try:
                    income_m = get_income_statement_metrics(stock) if stock is not None else {}
                    cf_m = get_cashflow_metrics(stock) if stock is not None else {}
                    bs_m = get_balance_sheet_metrics(stock) if stock is not None else {}
                except Exception:
                    income_m, cf_m, bs_m = {}, {}, {}

                risk_m = calc_volatility_and_drawdown(df)
                sentiment_label = st.session_state.get("news_sentiment_label")

                # --- Valuation ---
                st.markdown("### 💰 Valuace")
                v1, v2, v3 = st.columns(3)
                market_cap = _safe_float(info.get("marketCap"))
                current_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
                fcf = cf_m.get("fcf")
                revenue = income_m.get("revenue")

                with v1:
                    st.metric("Market Cap", f"{market_cap/1e9:.2f}B" if market_cap else "N/A")
                    st.caption("Tržní kapitalizace = cena firmy na burze (počet akcií × cena).")
                with v2:
                    if market_cap and fcf:
                        fcf_yield = fcf / market_cap
                        st.metric("FCF Yield", f"{fcf_yield*100:.2f}%")
                        st.caption("FCF yield = volný cashflow / market cap. Vyšší často znamená levnější ocenění (v kontextu kvality/růstu).")
                    else:
                        st.metric("FCF Yield", "N/A")
                        st.caption("Nelze spočítat bez market cap a FCF.")
                with v3:
                    pe = _safe_float(info.get("trailingPE"))
                    fpe = _safe_float(info.get("forwardPE"))
                    st.metric("P/E (TTM / Forward)", f"{pe:.1f} / {fpe:.1f}" if pe and fpe else (f"{pe:.1f}" if pe else "N/A"))
                    st.caption("P/E je citlivé na jednorázové položky. U růstových firem často dává větší smysl EV/FCF nebo EV/Sales.")

                # --- Growth ---
                st.markdown("### 📈 Růst")
                g1, g2, g3 = st.columns(3)
                rev_growth = None
                try:
                    inc = getattr(stock, "financials", None)
                    if inc is not None and not inc.empty and "Total Revenue" in inc.index and inc.shape[1] >= 2:
                        r0 = _safe_float(inc.iloc[inc.index.get_loc("Total Revenue"), 0])
                        r1 = _safe_float(inc.iloc[inc.index.get_loc("Total Revenue"), 1])
                        if r0 is not None and r1 not in (None, 0):
                            rev_growth = (r0 / r1) - 1.0
                except Exception:
                    rev_growth = None

                with g1:
                    st.metric("Tržby (posl. rok)", f"{revenue/1e9:.2f}B" if revenue else "N/A")
                    st.caption("Základní velikost firmy. U menších firem bývá růst volatilnější.")
                with g2:
                    st.metric("YoY růst tržeb", f"{rev_growth*100:.1f}%" if rev_growth is not None else "N/A")
                    st.caption("Meziroční růst dle posledních 2 ročních období z yfinance (pokud jsou k dispozici).")
                with g3:
                    if revenue and fcf is not None:
                        fcf_margin = fcf / revenue if revenue else None
                        st.metric("FCF marže", f"{fcf_margin*100:.1f}%" if fcf_margin is not None else "N/A")
                        st.caption("Kolik % z tržeb se reálně promění ve volné peníze (po investicích).")
                    else:
                        st.metric("FCF marže", "N/A")
                        st.caption("Nelze spočítat bez tržeb a FCF.")

                # --- Quality ---
                st.markdown("### 🏰 Kvalita byznysu")
                q1, q2, q3 = st.columns(3)
                gm = income_m.get("gross_margin")
                om = income_m.get("operating_margin")
                nm = income_m.get("net_margin")
                with q1:
                    st.metric("Gross margin", f"{gm*100:.1f}%" if gm is not None else "N/A")
                    st.caption("Hrubá marže – síla produktu/pricing a nákladová struktura.")
                with q2:
                    st.metric("Operating margin", f"{om*100:.1f}%" if om is not None else "N/A")
                    st.caption("Provozní marže – efektivita řízení firmy po provozních nákladech.")
                with q3:
                    st.metric("Net margin", f"{nm*100:.1f}%" if nm is not None else "N/A")
                    st.caption("Čistá marže – kolik z tržeb zůstane akcionářům po všem (daně, úroky).")

                # --- Financial health ---
                st.markdown("### 🏥 Finanční zdraví")
                h1, h2, h3 = st.columns(3)
                cash = bs_m.get("cash")
                debt = bs_m.get("debt")
                net_debt = bs_m.get("net_debt")
                cr = bs_m.get("current_ratio")
                with h1:
                    st.metric("Cash", f"{cash/1e9:.2f}B" if cash else "N/A")
                    st.caption("Hotovost a ekvivalenty. Důležitá pro flexibilitu a přežití v krizi.")
                with h2:
                    st.metric("Net debt", f"{net_debt/1e9:.2f}B" if net_debt is not None else "N/A")
                    st.caption("Dluh minus cash. Záporné = firma má více cash než dluhu.")
                with h3:
                    st.metric("Current ratio", f"{cr:.2f}" if cr is not None else "N/A")
                    st.caption("Likvidita krátkodobě. <1 může být varování (záleží na sektoru).")

                # --- Risk ---
                st.markdown("### ⚠️ Riziko")
                r1, r2, r3 = st.columns(3)
                beta = _safe_float(info.get("beta"))
                vol = risk_m.get("volatility")
                mdd = risk_m.get("max_drawdown")
                with r1:
                    st.metric("Beta", f"{beta:.2f}" if beta is not None else "N/A")
                    st.caption("Citlivost vůči trhu. >1 = obvykle více kolísá než trh.")
                with r2:
                    st.metric("Volatilita (ann.)", f"{vol*100:.1f}%" if vol is not None else "N/A")
                    st.caption("Roční volatilita z denních výnosů (historická).")
                with r3:
                    st.metric("Max drawdown", f"{mdd*100:.1f}%" if mdd is not None else "N/A")
                    st.caption("Největší historický propad z lokálního maxima (v zobrazeném období).")

                # --- Scorecard ---
                st.markdown("### 🧠 Scorecard (heuristika)")
                scores, total = simple_scorecard(info, income_m, cf_m, bs_m, risk_m, sentiment_label)
                st.metric("Celkové skóre", f"{total}/100")
                st.caption("Skóre je orientační – cílem je rychle odhalit slabá místa a kde se ptát dál.")
                sc_cols = st.columns(6)
                for i, (k, v) in enumerate(scores.items()):
                    with sc_cols[i]:
                        st.metric(k, f"{v}/100")

                # --- Scenario model ---
                st.markdown("### 🧮 Scénáře férové ceny (jednoduchý model)")
                st.caption("Rychlý model: projekce tržeb → FCF → terminální hodnota přes FCF multiple. "
                           "Je to zjednodušení, ale dobré pro 'sanity check' očekávání.")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    years = st.slider("Horizont (roky)", 2, 10, 5)
                with s2:
                    growth = st.slider("Růst tržeb p.a.", 0.0, 0.5, 0.10, 0.01)
                with s3:
                    default_margin = float((fcf / revenue) if (fcf is not None and revenue) else 0.15)
                    fcf_margin = st.slider("FCF marže", 0.0, 0.6, min(max(default_margin, 0.0), 0.6), 0.01)
                with s4:
                    exit_mult = st.slider("Exit FCF multiple", 5.0, 40.0, 20.0, 1.0)

                discount = st.slider("Diskontní sazba", 0.05, 0.20, 0.10, 0.01)
                scen = scenario_fair_value(info, income_m, years, growth, fcf_margin, exit_mult, discount)
                if scen and scen.get("fair_price") is not None and current_price:
                    fair = scen["fair_price"]
                    upside = (fair / current_price) - 1.0 if current_price else None
                    cA, cB, cC = st.columns(3)
                    with cA:
                        st.metric("Férová cena (scénář)", f"${fair:.2f}")
                        st.caption("Odhad na základě zvolených vstupů.")
                    with cB:
                        st.metric("Aktuální cena", f"${current_price:.2f}")
                        st.caption("Cena z Yahoo / yfinance.")
                    with cC:
                        st.metric("Potenciál vs aktuální", f"{upside*100:.1f}%" if upside is not None else "N/A")
                        st.caption("Pozitivní = scénář říká, že je prostor nahoru. Neřeší to rizika a konkurenci.")
                else:
                    st.info("Scénář nejde spočítat (chybí tržby/market data). Zkus jiný ticker nebo zkontroluj zdroj dat.")

                # --- Red flags ---
                st.markdown("### 🚩 Red flags (automatická varování)")
                flags = []
                if cf_m.get("fcf") is not None and cf_m["fcf"] < 0:
                    flags.append("Negativní FCF (firma pálí hotovost).")
                if bs_m.get("current_ratio") is not None and bs_m["current_ratio"] < 1.0:
                    flags.append("Current ratio < 1 (krátkodobá likvidita může být napjatá).")
                if bs_m.get("net_debt") is not None and bs_m["net_debt"] > 0 and cash and cash > 0 and (bs_m["net_debt"] / cash) > 2:
                    flags.append("Net debt je výrazně vyšší než hotovost (potenciální tlak v krizi).")
                if om is not None and om < 0:
                    flags.append("Negativní operating margin (provozní ztráta).")
                if rev_growth is not None and rev_growth < 0:
                    flags.append("Meziroční pokles tržeb (zkontroluj, jestli jde o cyklus nebo strukturální problém).")

                if flags:
                    for f in flags:
                        st.warning(f)
                else:
                    st.success("Bez zjevných red flags z dostupných dat (stále platí: ověř si kontext).")

            # Footer s doporučením
            st.markdown("---")
            with st.expander("🎯 FINÁLNÍ VYHODNOCENÍ (souhrn)", expanded=False):
            
                final_signals = []
                # Další signály: doporučení analytiků (Yahoo) + sentiment zpráv (bez AI)
                try:
                    rk = (info.get("recommendationKey") or "").lower()
                    if rk in ["strong_buy", "buy"]:
                        final_signals.append(("buy", f"Analytici (Yahoo): {rk.replace('_',' ').title()}"))
                    elif rk in ["sell", "strong_sell"]:
                        final_signals.append(("sell", f"Analytici (Yahoo): {rk.replace('_',' ').title()}"))
                except Exception:
                    pass

                try:
                    news_sent, _news_items = get_news_sentiment_simple(ticker)
                    if news_sent == "Pozitivní":
                        final_signals.append(("buy", "Novinky: pozitivní sentiment (fallback)"))
                    elif news_sent == "Negativní":
                        final_signals.append(("sell", "Novinky: negativní sentiment (fallback)"))
                except Exception:
                    pass
            
                # Technické signály
                if not df.empty and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                    sma50 = df['SMA_50'].iloc[-1]
                    sma200 = df['SMA_200'].iloc[-1]
                    if pd.notna(sma50) and pd.notna(sma200):
                        if sma50 > sma200:
                            final_signals.append(("buy", "Technická analýza: Golden Cross"))
                        else:
                            final_signals.append(("sell", "Technická analýza: Death Cross"))
            
                # Valuace
                if "PODHODNOCENÁ" in valuation_status:
                    final_signals.append(("buy", "Fundamentální analýza: Podhodnocená"))
                elif "NADHODNOCENÁ" in valuation_status:
                    final_signals.append(("sell", "Fundamentální analýza: Nadhodnocená"))
            
                # Finanční zdraví
                if health_score >= 70:
                    final_signals.append(("buy", "Finanční zdraví: Silná společnost"))
                elif health_score < 40:
                    final_signals.append(("sell", "Finanční zdraví: Slabá společnost"))
            
                # Vyhodnocení
                st.caption("Pozn.: Tyto signály jsou souhrn pravidel této aplikace (technické/fundamentální), nejsou to hlasy analytiků.")
                buy_signals = sum(1 for s in final_signals if s[0] == "buy")
                sell_signals = sum(1 for s in final_signals if s[0] == "sell")
            
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.metric("🟢 Nákupní signály (interní)", buy_signals)
            
                with col2:
                    st.metric("🔴 Prodejní signály (interní)", sell_signals)
            
                with col3:
                    if buy_signals > sell_signals:
                        recommendation_final = "🟢 KOUPIT"
                        st.success(recommendation_final)
                    elif sell_signals > buy_signals:
                        recommendation_final = "🔴 NEPORUČENO"
                        st.error(recommendation_final)
                    else:
                        recommendation_final = "🟡 DRŽET"
                        st.warning(recommendation_final)
            
                # Seznam signálů
                st.markdown("### 📋 Detaily signálů:")
                for signal_type, description in final_signals:
                    if signal_type == "buy":
                        st.success(f"✅ {description}")
                    else:
                        st.error(f"❌ {description}")
            
            
# Disclaimer
            st.markdown("---")
            st.warning("""
            ⚠️ **DŮLEŽITÉ UPOZORNĚNÍ:**
            
            Tato aplikace slouží pouze pro vzdělávací a informativní účely. Není finančním poradcem
            a neposkytuje investiční doporučení. Všechna investiční rozhodnutí činíte na vlastní riziko.
            Vždy proveďte důkladný výzkum a konzultujte s profesionálním finančním poradcem před
            investováním.
            """)
    
    else:
        # Úvodní obrazovka
        st.info("👈 Začněte zadáním tickeru v levém panelu a klikněte na tlačítko 'ANALYZOVAT AKCII'")
        
        st.markdown("### 🎯 Co tato aplikace nabízí:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📈 Technická analýza
            - Interaktivní grafy s candlesticky
            - SMA 50 & 200
            - RSI indikátor
            - MACD
            - Automatické signály
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Fundamentální data
            - P/E, PEG ratio
            - Market Cap
            - Dividendy
            - EPS a ziskovost
            - Analýza ocenění
            """)
        
        with col3:
            st.markdown("""
            #### 🤖 Pokročilé funkce
            - Doporučení analytiků
            - **AI sentiment analýza** 🆕
            - Insider trading
            - Finanční zdraví
            - Fear & Greed Index
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Příklady tickerů:")
        st.code("AAPL, MSFT, GOOGL, TSLA, NVDA, META, AMZN, CEZ.PR, BTC-USD")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Analýza novinek")
        st.info("""
        Aplikace nyní podporuje AI analýzu novinek! 
        
        **Jak aktivovat:**
        1. V levém panelu zaškrtni "Povolit AI analýzu novinek"
        2. Vyber Google Gemini (zdarma) nebo OpenAI
        3. Zadej svůj API klíč
        4. Analýza se spustí automaticky
        
        **Google Gemini zdarma:** https://makersuite.google.com/app/apikey
        """)

if __name__ == "__main__":
    main()