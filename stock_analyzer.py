"""
Stock Analysis Application - PRO EDITION v9.0
Vše v jednom: DCF, AI Sentiment, Insider Trading, Makro, Finanční zdraví.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import quote_plus
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURACE A CSS (Rozšířený styling)
# ============================================================================

st.set_page_config(
    page_title="📈 Stock Analyzer Pro - Ultimate Edition",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem; }
    .sentiment-positive { background-color: #d1fae5; color: #065f46; padding: 0.8rem; border-radius: 10px; font-weight: bold; text-align: center; border: 1px solid #059669; }
    .sentiment-negative { background-color: #fee2e2; color: #991b1b; padding: 0.8rem; border-radius: 10px; font-weight: bold; text-align: center; border: 1px solid #dc2626; }
    .sentiment-neutral { background-color: #f1f5f9; color: #475569; padding: 0.8rem; border-radius: 10px; font-weight: bold; text-align: center; border: 1px solid #94a3b8; }
    .metric-card { background-color: #ffffff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border-top: 4px solid #3b82f6; }
    .status-strong { color: #059669; font-weight: bold; }
    .status-weak { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# POMOCNÉ FUNKCE (LOGIKA A VÝPOČTY)
# ============================================================================

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return float(x)
    except: return None

def format_large_num(num):
    if num is None: return "N/A"
    if abs(num) >= 1e12: return f"${num/1e12:.2f}T"
    if abs(num) >= 1e9: return f"${num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"${num/1e6:.2f}M"
    return f"${num:.2f}"

def estimate_fair_value(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """DCF prioritně, Analyst Mean jako záloha."""
    try:
        fcf = safe_float(info.get("freeCashflow"))
        shares = safe_float(info.get("sharesOutstanding"))
        if fcf and shares and fcf > 0:
            discount, growth, terminal, years = 0.10, 0.05, 0.02, 5
            pv_fcf = sum([(fcf * (1 + growth)**i) / (1 + discount)**i for i in range(1, years + 1)])
            tv = (fcf * (1 + growth)**years * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / (1 + discount)**years
            return (pv_fcf + pv_tv) / shares, "DCF Model (Intrinsic)"
        tmean = safe_float(info.get("targetMeanPrice"))
        if tmean: return tmean, "Analyst Consensus"
    except: pass
    return None, "N/A"

def calculate_financial_score(info: Dict) -> Tuple[int, str]:
    """Vypočítá skóre zdraví 0-100."""
    score = 0
    metrics = {
        'debtToEquity': lambda x: 20 if x < 80 else (10 if x < 150 else 0),
        'currentRatio': lambda x: 20 if x > 1.5 else (10 if x > 1.0 else 0),
        'profitMargins': lambda x: 20 if x > 0.15 else 10,
        'returnOnEquity': lambda x: 20 if x > 0.15 else 10,
        'freeCashflow': lambda x: 20 if x > 0 else 0
    }
    for k, func in metrics.items():
        val = safe_float(info.get(k))
        if val is not None: score += func(val)
    
    status = "🔴 Slabé" if score < 40 else ("🟡 Průměrné" if score < 70 else "🟢 Silné")
    return score, status

def get_macro_data():
    tickers = {"S&P 500": "^GSPC", "US 10Y Yield": "^TNX", "VIX Index": "^VIX", "Gold": "GC=F", "Crude Oil": "CL=F"}
    res = {}
    for name, sym in tickers.items():
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                c, p = h["Close"].iloc[-1], h["Close"].iloc[-2]
                res[name] = {"val": c, "chg": ((c-p)/p)*100}
        except: pass
    return res

# ============================================================================
# AI A ZPRÁVY
# ============================================================================

def analyze_news_ai(news_titles, api_key, ticker):
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        valid = [t for t in news_titles if len(t) > 15]
        if not valid: return "Neutrální", "Nedostatek dat pro AI."
        
        prompt = f"Jsi finanční analytik. Analyzuj sentiment zpráv pro {ticker}:\n" + "\n".join(valid[:8]) + \
                 "\nOdpověz ve formátu:\nSENTIMENT: [Pozitivní/Negativní/Neutrální]\nSHRNUTÍ: [2 věty]"
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text
        sent = "Neutrální"
        summ = text
        for line in text.split('\n'):
            if 'SENTIMENT:' in line.upper(): sent = line.split(':')[-1].strip().replace('[','').replace(']','')
            elif 'SHRNUTÍ:' in line.upper(): summ = line.split(':')[-1].strip()
        return sent, summ
    except Exception as e:
        return "Neutrální", f"Chyba AI: {str(e)}"

# ============================================================================
# HLAVNÍ APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Ultimate Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
        st.header("⚙️ Ovládací panel")
        ticker_symbol = st.text_input("Zadejte Ticker (např. TSLA, NVDA)", value="AAPL").upper()
        period = st.selectbox("Historické období", ["1y", "2y", "5y", "10y", "max"], index=1)
        st.markdown("---")
        ai_toggle = st.checkbox("🤖 Aktivovat AI Analýzu", value=False)
        api_key = st.text_input("Gemini API Key", type="password") if ai_toggle else ""
        st.markdown("---")
        st.info("Klíč zdarma: [Google AI Studio](https://aistudio.google.com/)")
        analyze_btn = st.button("🚀 SPUSTIT HLOUBKOVOU ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker_symbol:
        with st.spinner(f"Provádím komplexní analýzu {ticker_symbol}..."):
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period=period)
            info = stock.info
            
            if df.empty:
                st.error("Data nebyla nalezena. Zkontrolujte ticker.")
                return

            # --- TOP BAR METRIKY ---
            m1, m2, m3, m4, m5 = st.columns(5)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            prev_p = info.get('previousClose', curr_p)
            m1.metric("🏢 Firma", info.get('shortName', ticker_symbol))
            m2.metric("💰 Cena", f"${curr_p:.2f}", f"{((curr_p-prev_p)/prev_p)*100:.2f}%")
            
            fair_v, fair_m = estimate_fair_value(info)
            if fair_v:
                upside = ((fair_v/curr_p)-1)*100
                m3.metric("🎯 Férová cena", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            else: m3.metric("🎯 Férová cena", "N/A")
            
            m4.metric("📊 Market Cap", format_large_num(info.get('marketCap')))
            m5.metric("📈 P/E Ratio", info.get('trailingPE', 'N/A'))

            # --- TABS (Všechny taby jsou zpět!) ---
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "📈 Technická analýza", "🏛️ Fundamenty", "📰 Novinky & AI", 
                "💼 Insider Trading", "🏥 Zdraví & Skóre", "🌍 Makro & Peers", "🧮 DCF Simulátor"
            ])

            # TAB 1: Technická analýza
            with t1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='SMA 50', line=dict(color='orange', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='SMA 200', line=dict(color='red', width=1.5)), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.3), row=2, col=1)
                fig.update_layout(height=600, template='plotly_white', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # TAB 2: Fundamenty
            with t2:
                st.subheader("📊 Klíčové metriky")
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.write("**Valuace**")
                    st.write(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
                    st.write(f"PEG Ratio: {info.get('pegRatio', 'N/A')}")
                    st.write(f"Price/Book: {info.get('priceToBook', 'N/A')}")
                with f2:
                    st.write("**Ziskovost**")
                    st.write(f"Profit Margin: {info.get('profitMargins', 0)*100:.2f}%")
                    st.write(f"ROE: {info.get('returnOnEquity', 0)*100:.2f}%")
                    st.write(f"EPS (TTM): ${info.get('trailingEps', 'N/A')}")
                with f3:
                    st.write("**Dividendy**")
                    st.write(f"Výnos: {info.get('dividendYield', 0)*100:.2f}%")
                    st.write(f"Payout Ratio: {info.get('payoutRatio', 0)*100:.2f}%")
                
                st.markdown("---")
                st.subheader("📊 Roční tržby a zisk")
                fin = stock.financials.T
                if not fin.empty and 'Total Revenue' in fin.columns:
                    st.bar_chart(fin[['Total Revenue', 'Net Income']])

            # TAB 3: Novinky & AI
            with t3:
                raw_news = stock.news[:10]
                titles = [n.get('title') for n in raw_news if n.get('title')]
                if ai_toggle and api_key:
                    with st.spinner("AI mozek analyzuje zprávy..."):
                        sent, summ = analyze_news_ai(titles, api_key, ticker_symbol)
                        st.markdown(f"### AI Sentiment: {sent}")
                        st.info(summ)
                
                st.markdown("---")
                for n in raw_news:
                    with st.expander(f"📰 {n.get('title')}"):
                        st.write(f"Zdroj: {n.get('publisher')}")
                        st.markdown(f"[Přečíst článek]({n.get('link')})")

            # TAB 4: Insider Trading (Vracíme v plné verzi!)
            with t4:
                st.subheader("💼 Transakce vedení společnosti (Insiders)")
                insider = stock.insider_transactions
                if insider is not None and not insider.empty:
                    st.dataframe(insider.head(20), use_container_width=True)
                else:
                    st.info("Data o insider trading nejsou pro tento ticker momentálně dostupná.")

            # TAB 5: Zdraví & Skóre
            with t5:
                score, status = calculate_financial_score(info)
                c_s1, c_s2 = st.columns([1, 2])
                with c_s1:
                    st.metric("Finanční skóre", f"{score}/100")
                    st.subheader(f"Status: {status}")
                with c_s2:
                    st.write("Skóre je složeno z: Likvidity, Zadluženosti, Rentability a Cash Flow.")
                    st.progress(score / 100)

            # TAB 6: Makro & Peers
            with t6:
                st.subheader("🌍 Globální trh")
                m_data = get_macro_data()
                m_cols = st.columns(len(m_data))
                for i, (name, d) in enumerate(m_data.items()):
                    m_cols[i].metric(name, f"{d['val']:.2f}", f"{d['chg']:.2f}%")
                
                st.markdown("---")
                st.subheader("👥 Srovnání s konkurencí")
                p_in = st.text_input("Zadejte tickery konkurence (oddělené čárkou)", value="MSFT, GOOGL, AMZN")
                if p_in:
                    all_p = [ticker_symbol] + [x.strip().upper() for x in p_in.split(",")]
                    peer_res = []
                    for p in all_p:
                        try:
                            pi = yf.Ticker(p).info
                            peer_res.append({"Ticker": p, "P/E": pi.get('trailingPE'), "Marže": pi.get('profitMargins', 0)*100, "EV/Sales": pi.get('enterpriseToRevenue')})
                        except: pass
                    st.table(pd.DataFrame(peer_res))

            # TAB 7: DCF Simulátor
            with t7:
                st.subheader("🧮 Vlastní DCF model (Discounted Cash Flow)")
                col_d1, col_d2 = st.columns([1, 2])
                with col_d1:
                    g_rate = st.slider("Růst FCF (příštích 5 let)", 0.0, 0.40, 0.15, 0.01)
                    d_rate = st.slider("Diskontní sazba (WACC)", 0.07, 0.15, 0.10, 0.01)
                with col_d2:
                    fcf = safe_float(info.get("freeCashflow"))
                    sh = safe_float(info.get("sharesOutstanding"))
                    if fcf and sh:
                        # Zjednodušený výpočet pro UI
                        val = (fcf*(1+g_rate)**5 / (d_rate-0.025)) / (1+d_rate)**5 / sh
                        st.metric("Tvoje férová cena", f"${val:.2f}", f"{((val/curr_p)-1)*100:+.1f}% vs trh")
                    else: st.warning("Data pro DCF nejsou dostupná.")

            st.markdown("---")
            st.caption(f"Poslední aktualizace: {datetime.now().strftime('%d.%m.%Y %H:%M')}. Zdroj dat: Yahoo Finance.")

if __name__ == "__main__":
    main()
