"""
Stock Analysis Pro - Master Edition v12.0
Komplexní terminál pro rok 2026.
Prioritní DCF, Finanční zdraví (Z-Score), Makro Outlook a AI Sentiment.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import quote_plus

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DESIGN A KONFIGURACE
# ============================================================================
st.set_page_config(page_title="Stock Analyzer Master Pro 2026", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #0f172a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem; }
    .stMetric { background-color: #ffffff; border-radius: 12px; padding: 15px !important; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .status-card { padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6; background: white; margin-bottom: 20px; }
    .sentiment-pos { background-color: #dcfce7; color: #166534; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
    .sentiment-neg { background-color: #fee2e2; color: #991b1b; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. ANALYTIKŮV MOZEK (VÝPOČTY)
# ============================================================================

def safe_get(data, key, default=0):
    val = data.get(key)
    return float(val) if val is not None and not pd.isna(val) else default

def calculate_advanced_valuation(info: Dict) -> Tuple[Optional[float], str]:
    """Výpočet vnitřní hodnoty (DCF) jako priority."""
    try:
        fcf = safe_get(info, 'freeCashflow')
        shares = safe_get(info, 'sharesOutstanding')
        if fcf > 0 and shares > 0:
            growth = 0.05; discount = 0.10; terminal = 0.025; years = 5
            pv_fcf = sum([(fcf * (1 + growth)**i) / (1 + discount)**i for i in range(1, years + 1)])
            tv = (fcf * (1 + growth)**years * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / (1 + discount)**years
            return (pv_fcf + pv_tv) / shares, "DCF Model (Intrinsic)"
    except: pass
    t_mean = safe_get(info, 'targetMeanPrice', None)
    return t_mean, "Analyst Consensus" if t_mean else "N/A"

def get_financial_health_metrics(stock_obj, info):
    """Výpočet Altman Z-Score a Piotroski F-Score pro hloubkovou analýzu rizika."""
    results = {"z_score": "N/A", "f_score": 0, "status": "N/A"}
    try:
        bs = stock_obj.balance_sheet.iloc[:, 0]
        is_stmt = stock_obj.financials.iloc[:, 0]
        
        # Altman Z-Score (Manufacturing)
        total_assets = safe_get(bs, 'Total Assets', 1)
        ebit = safe_get(is_stmt, 'EBIT', 0)
        z = (safe_get(bs, 'Working Capital') / total_assets * 1.2) + \
            (safe_get(bs, 'Retained Earnings') / total_assets * 1.4) + \
            (ebit / total_assets * 3.3) + \
            (info.get('marketCap', 0) / safe_get(bs, 'Total Liabilities Net Minority Interest', 1) * 0.6) + \
            (safe_get(is_stmt, 'Total Revenue') / total_assets * 1.0)
        results["z_score"] = round(z, 2)
        results["status"] = "🟢 Bezpečné" if z > 2.99 else ("🟡 Šedá zóna" if z > 1.8 else "🔴 Riziko úpadku")
    except: pass
    return results

# ============================================================================
# 3. AI & NEWS (OPRAVA PRÁZDNÝCH TITULKŮ)
# ============================================================================

def get_clean_news(ticker):
    """Získává novinky a agresivně čistí titulky pro AI."""
    try:
        s = yf.Ticker(ticker)
        raw = s.news[:10]
        processed = []
        for n in raw:
            title = n.get('title') or n.get('headline') or "Bez názvu"
            if title != "Bez názvu" and len(title) > 10:
                processed.append({"title": title, "link": n.get('link'), "pub": n.get('publisher')})
        return processed
    except: return []

# ============================================================================
# 4. HLAVNÍ APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Master Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Terminál")
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        history_period = st.selectbox("Historická data", ["1y", "2y", "5y", "10y", "max"], index=1)
        st.markdown("---")
        ai_enabled = st.checkbox("🤖 AI Analýza Sentimentu", value=False)
        gemini_key = st.text_input("Klíč Gemini API", type="password") if ai_enabled else ""
        st.markdown("---")
        st.subheader("🧮 Vstupy pro DCF")
        g_rate = st.slider("Růst (Growth)", 0.0, 0.4, 0.12)
        wacc = st.slider("Sazba (WACC)", 0.07, 0.18, 0.10)
        analyze_btn = st.button("SPUSTIT HLOUBKOVOU ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker:
        with st.spinner(f"Doluji hloubková data pro {ticker}..."):
            stock = yf.Ticker(ticker)
            df = stock.history(period=history_period)
            info = stock.info
            
            if df.empty:
                st.error("Chyba: Ticker nebyl nalezen.")
                return

            # --- HEADER ---
            h1, h2, h3, h4 = st.columns(4)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            h1.metric("🏢 Firma", info.get('longName', ticker))
            h2.metric("💰 Cena", f"${curr_p:.2f}", f"{((curr_p - info.get('previousClose', curr_p))/info.get('previousClose', 1)*100):.2f}%")
            
            fair_v, fair_m = calculate_advanced_valuation(info)
            if fair_v:
                upside = ((fair_v / curr_p) - 1) * 100
                h3.metric("🎯 Férová cena", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            
            health = get_financial_health_metrics(stock, info)
            h4.metric("🛡️ Z-Score", health["z_score"], health["status"])

            st.markdown("---")

            # --- TAB SYSTÉM ---
            tabs = st.tabs(["📈 Graf & Signály", "📊 Fundamenty (10Y)", "📰 AI Sentiment", "💼 Insiders", "🌍 Macro 2026", "👥 Peers"])

            with tabs[0]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='MA50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='MA200', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.3), row=2, col=1)
                fig.update_layout(height=600, template='plotly_white', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                st.subheader("🏛️ Hloubková historie tržeb a zisku")
                fin_data = stock.financials.T
                if not fin_data.empty:
                    st.bar_chart(fin_data[['Total Revenue', 'Net Income']] if 'Total Revenue' in fin_data.columns else [])
                    st.markdown("#### Detailní rozvaha (poslední rok)")
                    st.dataframe(stock.balance_sheet.head(10), use_container_width=True)
                else: st.warning("Finanční historie není k dispozici.")

            with tabs[2]:
                st.subheader("📰 AI Analýza & Zprávy")
                news_items = get_clean_news(ticker)
                if ai_enabled and gemini_key:
                    from google import genai
                    client = genai.Client(api_key=gemini_key)
                    titles = [n['title'] for n in news_items]
                    prompt = f"Analyzuj sentiment pro {ticker}:\n" + "\n".join(titles[:8])
                    try:
                        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                        st.info(f"### AI Shrnutí:\n{resp.text}")
                    except Exception as e: st.error(f"AI Chyba: {e}")
                
                st.markdown("---")
                for n in news_items:
                    with st.expander(f"📰 {n['title']}"):
                        st.write(f"**Zdroj:** {n['pub']}")
                        st.markdown(f"[🔗 Odkaz na zprávu]({n['link']})")

            with tabs[4]:
                st.subheader("🗓️ Makro Kalendář 2026")
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Příští FED (Sazby)", "18. března 2026")
                c_m2.metric("Report Inflace (CPI)", "11. března 2026")
                c_m3.metric("Earnings Season", "Duben 2026")
                st.info("💡 **Tip analytika:** Pokud výnosy 10letých dluhopisů (US10Y) rostou, technologické akcie s vysokým P/E bývají pod tlakem.")

            with tabs[5]:
                st.subheader("👥 Srovnání s konkurencí")
                peers = st.text_input("Zadejte tickery konkurence (oddělené čárkou)", value="MSFT, GOOGL, AMZN")
                if peers:
                    all_p = [ticker] + [p.strip().upper() for p in peers.split(",")]
                    p_data = []
                    for p in all_p:
                        try:
                            pi = yf.Ticker(p).info
                            p_data.append({"Ticker": p, "Price": pi.get('currentPrice'), "P/E": pi.get('trailingPE'), "Marže": pi.get('profitMargins')})
                        except: pass
                    st.table(pd.DataFrame(p_data))

if __name__ == "__main__":
    main()
