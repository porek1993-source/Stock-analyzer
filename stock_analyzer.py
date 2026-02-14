"""
Stock Analysis Pro - Masterpiece Edition v15.0
Hloubkový terminál: DCF (Priorita), Altman Z-Score, 10Y Historie, Macro 2026.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURACE A DESIGN
# ============================================================================
st.set_page_config(page_title="Stock Analyzer Master Pro 2026", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 10px; }
    .stMetric { background-color: #ffffff; border-radius: 12px; padding: 15px !important; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .status-card { padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6; background: #f8fafc; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANALYTIKŮV MOZEK (MATEMATICKÉ MODELY)
# ============================================================================

def safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None and not pd.isna(x) else None
    except: return None

def estimate_fair_value_pro(info: Dict) -> Tuple[Optional[float], str]:
    """Hloubková vnitřní hodnota přes Discounted Cash Flow (DCF)."""
    try:
        fcf = safe_float(info.get("freeCashflow"))
        shares = safe_float(info.get("sharesOutstanding"))
        if fcf and shares and fcf > 0:
            # Model: 5 let růst 5%, diskont 10%, terminální růst 2%
            discount = 0.10; growth = 0.05; terminal = 0.02; years = 5
            pv_fcf = 0; temp_fcf = fcf
            for i in range(1, years + 1):
                temp_fcf *= (1 + growth)
                pv_fcf += temp_fcf / ((1 + discount) ** i)
            tv = (temp_fcf * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / ((1 + discount) ** years)
            return (pv_fcf + pv_tv) / shares, "DCF Model (Intrinsic Value)"
    except: pass
    
    t_mean = safe_float(info.get("targetMeanPrice"))
    return t_mean, "Analyst Target Mean" if t_mean else "N/A"

def get_altman_z_score(stock_obj, info):
    """Predikce finančního zdraví (Altman Z-Score). Nad 3.0 = Safe."""
    try:
        bs = stock_obj.balance_sheet.iloc[:, 0]
        is_stmt = stock_obj.financials.iloc[:, 0]
        assets = safe_float(bs.get('Total Assets', 1))
        z = (safe_float(bs.get('Working Capital', 0)) / assets * 1.2) + \
            (safe_float(bs.get('Retained Earnings', 0)) / assets * 1.4) + \
            (safe_float(is_stmt.get('EBIT', 0)) / assets * 3.3) + \
            (safe_float(info.get('marketCap', 0)) / safe_float(bs.get('Total Liabilities Net Minority Interest', 1)) * 0.6) + \
            (safe_float(is_stmt.get('Total Revenue', 0)) / assets * 1.0)
        return round(z, 2)
    except: return None

# ============================================================================
# HLAVNÍ APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Master Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Terminál")
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        history_period = st.selectbox("Období historie", ["2y", "5y", "10y", "max"], index=2)
        st.markdown("---")
        ai_enabled = st.checkbox("🤖 AI Analýza novinek", value=False)
        gemini_key = os.environ.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
        
        st.subheader("🧮 DCF Parametry")
        user_g = st.slider("Očekávaný růst (5y)", 0.0, 0.4, 0.12)
        user_d = st.slider("Diskontní sazba", 0.07, 0.18, 0.10)
        
        analyze_btn = st.button("SPUSTIT HLOUBKOVOU ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker:
        with st.spinner(f"Provádím analýzu {ticker}..."):
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            info = stock.info
            
            if df.empty:
                st.error("Data nebyla nalezena.")
                return

            # --- HEADER ---
            h1, h2, h3, h4 = st.columns(4)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            h1.metric("🏢 Společnost", info.get('shortName', ticker))
            h2.metric("💰 Cena", f"${curr_p:.2f}")
            
            fair_v, fair_m = estimate_fair_value_pro(info)
            if fair_v:
                upside = ((fair_v / curr_p) - 1) * 100
                h3.metric("🎯 Férová cena (DCF)", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            
            z_score = get_altman_z_score(stock, info)
            h4.metric("🛡️ Altman Z-Score", z_score if z_score else "N/A", "Safe" if z_score and z_score > 3 else "Risk")

            # --- TABS ---
            tabs = st.tabs(["📈 Technická analýza", "📊 10Y Fundamenty", "📰 AI Sentiment", "💼 Insider Trading", "🌍 Macro & Peers"])

            with tabs[0]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='MA50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='MA200', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.3), row=2, col=1)
                fig.update_layout(height=600, template='plotly_white', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                st.subheader("🏛️ Historický vývoj (Tržby, Zisk, FCF)")
                fin = stock.financials.T
                cf = stock.cashflow.T
                if not fin.empty and not cf.empty:
                    hist_data = pd.DataFrame({
                        "Tržby": fin['Total Revenue'] if 'Total Revenue' in fin.columns else 0,
                        "Zisk": fin['Net Income'] if 'Net Income' in fin.columns else 0,
                        "FCF": cf['Free Cash Flow'] if 'Free Cash Flow' in cf.columns else 0
                    }).sort_index()
                    st.bar_chart(hist_data)
                    st.dataframe(hist_data.T, use_container_width=True)
                else: st.warning("Hloubková historická data nejsou pro tento ticker dostupná.")

            with tabs[2]:
                st.subheader("📰 Novinky a AI Sentiment")
                news = stock.news[:10]
                for n in news:
                    with st.expander(f"📰 {n.get('title', 'Zpráva')}"):
                        st.write(f"**Zdroj:** {n.get('publisher')}")
                        st.markdown(f"[Odkaz na článek]({n.get('link')})")

            with tabs[3]:
                st.subheader("💼 Insider Trading")
                ins = stock.insider_transactions
                if ins is not None and not ins.empty:
                    st.dataframe(ins.head(30), use_container_width=True)
                else: st.info("Data o insider trading nejsou dostupná.")

            with tabs[4]:
                st.subheader("🌍 Makro Kontext & Konkurence")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.info("**Příští události 2026:**\n- FED Meeting: 18. března\n- CPI Report: 11. března")
                with col_m2:
                    peers = st.text_input("Srovnání s konkurencí (Tickery)", value="MSFT, GOOGL, AMZN")
                    if peers:
                        p_list = [ticker] + [p.strip().upper() for p in peers.split(",")]
                        p_data = []
                        for p in p_list:
                            try:
                                pi = yf.Ticker(p).info
                                p_data.append({"Ticker": p, "P/E": pi.get('trailingPE'), "Marže": pi.get('profitMargins')})
                            except: pass
                        st.table(pd.DataFrame(p_data))

if __name__ == "__main__":
    main()
