"""
Stock Analysis Pro - Analyst Edition v11.0
Komplexní terminál: DCF, Altman Z-Score, Makro Outlook, Peer Comparison.
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

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURACE A DESIGN
# ============================================================================

st.set_page_config(page_title="Ultimate Stock Analyst 2026", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main-header { font-size: 3rem; font-weight: 800; color: #1e3a8a; text-align: center; margin-bottom: 20px; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .card { background: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 5px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MATEMATICKÉ MODELY (DCF & RISK)
# ============================================================================

def calculate_dcf(fcf, shares, growth=0.05, discount=0.10, terminal=0.02, years=5):
    """Výpočet vnitřní hodnoty pomocí DCF: $PV = \sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^n}$"""
    if not fcf or not shares or fcf <= 0: return None
    
    cash_flows = [fcf * (1 + growth)**i for i in range(1, years + 1)]
    pv_cf = sum([cf / (1 + discount)**i for i, cf in enumerate(cash_flows, 1)])
    
    # Gordonův růstový model pro terminální hodnotu
    tv = (cash_flows[-1] * (1 + terminal)) / (discount - terminal)
    pv_tv = tv / (1 + discount)**years
    
    return (pv_cf + pv_tv) / shares

def get_altman_z_score(info, ticker_obj):
    """Výpočet Altman Z-Score pro predikci finančního zdraví"""
    try:
        bs = ticker_obj.balance_sheet.iloc[:, 0]
        is_stmt = ticker_obj.financials.iloc[:, 0]
        
        working_capital = bs.get('Working Capital', bs.get('Total Current Assets', 0) - bs.get('Total Current Liabilities', 0))
        total_assets = bs.get('Total Assets', 1)
        retained_earnings = bs.get('Retained Earnings', 0)
        ebit = is_stmt.get('EBIT', 0)
        market_cap = info.get('marketCap', 0)
        total_liabilities = bs.get('Total Liabilities Net Minority Interest', bs.get('Total Assets', 1) - bs.get('Total Equity Gross Minority Interest', 0))
        sales = is_stmt.get('Total Revenue', 0)

        # Tradiční váhy pro výrobní firmy (Z-Score)
        A = (working_capital / total_assets) * 1.2
        B = (retained_earnings / total_assets) * 1.4
        C = (ebit / total_assets) * 3.3
        D = (market_cap / total_liabilities) * 0.6
        E = (sales / total_assets) * 1.0
        
        z_score = A + B + C + D + E
        return round(z_score, 2)
    except: return None

# ============================================================================
# UI MODULY
# ============================================================================

def macro_outlook_section():
    """Výhled pro rok 2026"""
    st.markdown("### 🗓️ Macro Outlook 2026")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Příští zasedání FEDu**")
        st.code("18. března 2026")
        st.caption("Očekávání: Snížení o 25 bps")
    with col2:
        st.write("**Report Inflace (CPI)**")
        st.code("11. března 2026")
        st.caption("Cíl: Stabilizace na 2.1%")
    with col3:
        st.write("**Earnings Season**")
        st.code("Duben 2026")
        st.caption("Sledujte marže v AI sektoru")

# ============================================================================
# HLAVNÍ APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyst Ultimate Pro v11.0</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Nastavení Analýzy")
        ticker = st.text_input("Zadejte Ticker", value="NVDA").upper()
        ai_enabled = st.checkbox("🤖 Aktivovat AI Sentiment", value=False)
        api_key = st.text_input("Gemini API Key", type="password") if ai_enabled else ""
        st.markdown("---")
        st.subheader("🧮 DCF Parametry")
        g_rate = st.slider("Růst (Growth)", 0.0, 0.5, 0.15)
        d_rate = st.slider("Diskont (WACC)", 0.07, 0.20, 0.10)
        analyze_btn = st.button("SPUSTIT HLOUBKOVOU ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker:
        with st.spinner(f"Analyzuji data pro {ticker}..."):
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            info = stock.info

            if df.empty:
                st.error("Chyba při stahování dat. Zkontrolujte ticker.")
                return

            # --- TOP METRIKY ---
            m1, m2, m3, m4 = st.columns(4)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            m1.metric("🏢 Společnost", info.get('shortName', ticker))
            m2.metric("💰 Aktuální cena", f"${curr_p:.2f}")
            
            # Dynamický DCF výpočet
            fair_v = calculate_dcf(info.get('freeCashflow'), info.get('sharesOutstanding'), g_rate, d_rate)
            if fair_v:
                upside = ((fair_v / curr_p) - 1) * 100
                m3.metric("🎯 Vnitřní hodnota (DCF)", f"${fair_v:.2f}", f"{upside:+.1f}%")
            
            z_score = get_altman_z_score(info, stock)
            m4.metric("🛡️ Altman Z-Score", z_score if z_score else "N/A", delta="Safe" if z_score and z_score > 3 else "Risk")

            st.markdown("---")

            # --- TABS ---
            tabs = st.tabs(["📊 Technika", "🏛️ Fundamenty", "📰 AI Sentiment", "💼 Insider Trading", "🌍 Macro Outlook", "👥 Peers"])

            with tabs[0]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='MA50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='MA200', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.3), row=2, col=1)
                fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                st.subheader("📋 Klíčové finanční ukazatele")
                f1, f2, f3 = st.columns(3)
                f1.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                f1.write(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
                f2.write(f"**Gross Margin:** {info.get('grossMargins', 0)*100:.2f}%")
                f2.write(f"**Profit Margin:** {info.get('profitMargins', 0)*100:.2f}%")
                f3.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
                f3.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%")
                
                st.markdown("#### 📈 Historický růst (10 let)")
                st.bar_chart(stock.financials.T[['Total Revenue', 'Net Income']] if 'Total Revenue' in stock.financials.T.columns else [])

            with tabs[4]:
                macro_outlook_section()
                st.markdown("---")
                st.subheader("💡 Strategické shrnutí")
                st.write(f"Analýza akcie {ticker} pro rok 2026 naznačuje, že klíčovým faktorem bude {info.get('sector', 'daný sektor')}. "
                         f"S beta faktorem {info.get('beta', 'N/A')} je akcie " + 
                         ("více" if info.get('beta', 0) > 1 else "méně") + " volatilní než trh.")

            with tabs[5]:
                st.subheader("👥 Srovnání s konkurencí (Peer Analysis)")
                peer_input = st.text_input("Tickery konkurentů", value="AMD, INTC, MSFT")
                if peer_input:
                    p_list = [ticker] + [x.strip().upper() for x in peer_input.split(",")]
                    comp_data = []
                    for p in p_list:
                        try:
                            pi = yf.Ticker(p).info
                            comp_data.append({"Ticker": p, "Price": pi.get('currentPrice'), "P/E": pi.get('trailingPE'), "Marže": pi.get('profitMargins')})
                        except: pass
                    st.table(pd.DataFrame(comp_data))

            st.markdown('<div class="footer">Stock Analyzer Pro v11.0 | Data: Yahoo Finance | © 2026 Financial Terminal</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
