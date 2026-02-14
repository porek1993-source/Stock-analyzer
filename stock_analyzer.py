"""
Stock Analysis Pro - Masterpiece Edition v10.0
Komplexní nástroj pro hloubkovou fundamentální a technickou analýzu.
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
# 1. KONFIGURACE A STYLING
# ============================================================================

st.set_page_config(
    page_title="📈 Stock Analyzer Ultimate Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesionální UI Styling
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem; }
    .stMetric { background-color: #f8fafc; border-radius: 10px; padding: 15px !important; border-left: 5px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sentiment-positive { background-color: #dcfce7; color: #166534; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .sentiment-negative { background-color: #fee2e2; color: #991b1b; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .sentiment-neutral { background-color: #f1f5f9; color: #475569; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .footer { text-align: center; color: #64748b; padding: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. VÝPOČETNÍ MODULY (DCF, ZDRAVÍ, MAKRO)
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

def estimate_fair_value_dcf(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Hloubkový DCF Model - Priorita aplikace."""
    try:
        fcf = safe_float(info.get("freeCashflow"))
        shares = safe_float(info.get("sharesOutstanding"))
        if fcf and shares and fcf > 0:
            # Parametry: 5 let růst 5%, diskont 10%, terminální růst 2.5%
            pv_fcf = 0; growth = 0.05; discount = 0.10; terminal = 0.025
            temp_fcf = fcf
            for i in range(1, 6):
                temp_fcf *= (1 + growth)
                pv_fcf += temp_fcf / ((1 + discount) ** i)
            tv = (temp_fcf * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / ((1 + discount) ** 5)
            return (pv_fcf + pv_tv) / shares, "DCF Model (Intrinsic)"
        
        tmean = safe_float(info.get("targetMeanPrice"))
        if tmean: return tmean, "Analyst Consensus"
    except: pass
    return None, "N/A"

@st.cache_data(ttl=3600)
def get_macro_dashboard():
    """Získává makro data pro kontext investora"""
    tickers = {"S&P 500": "^GSPC", "US 10Y Yield": "^TNX", "VIX Index": "^VIX", "Gold": "GC=F", "USD Index": "DX-Y.NYB"}
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
# 3. AI A NEWS (GEMINI 2.0 FLASH)
# ============================================================================

def analyze_sentiment_ai(news_titles, api_key, ticker):
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        valid = [t for t in news_titles if len(t) > 15 and t != "Bez názvu"]
        if not valid: return "Neutrální", "Nedostatek čitelných zpráv pro AI."
        
        prompt = f"Jsi expert na akciový trh. Analyzuj dopad těchto zpráv na {ticker}:\n" + "\n".join(valid[:8]) + \
                 "\nOdpověz ve formátu:\nSENTIMENT: [Pozitivní/Negativní/Neutrální]\nSHRNUTÍ: [Stručné shrnutí, 3 věty]"
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text
        sent = "Neutrální"
        summ = text
        for line in text.split('\n'):
            if 'SENTIMENT:' in line.upper(): sent = line.split(':')[-1].strip().replace('[','').replace(']','')
            elif 'SHRNUTÍ:' in line.upper() or 'SUMMARY:' in line.upper(): summ = line.split(':')[-1].strip()
        return sent, summ
    except Exception as e:
        return "Neutrální", f"AI Chyba: {str(e)}"

# ============================================================================
# 4. HLAVNÍ UI APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Ultimate Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Analýza")
        ticker = st.text_input("Zadejte Ticker (např. AAPL, NVDA)", value="AAPL").upper()
        period = st.selectbox("Historie", ["1y", "2y", "5y", "10y", "max"], index=1)
        st.markdown("---")
        ai_on = st.checkbox("🤖 Aktivovat AI Sentiment", value=False)
        api_key = st.text_input("Gemini API Key", type="password") if ai_on else ""
        st.markdown("---")
        analyze_btn = st.button("🚀 SPUSTIT PROFESIONÁLNÍ ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker:
        with st.spinner(f"Provádím hloubkovou analýzu {ticker}..."):
            stock_obj = yf.Ticker(ticker)
            df = stock_obj.history(period=period)
            info = stock_obj.info
            
            if df.empty:
                st.error("Data nebyla nalezena. Zkontrolujte ticker.")
                return

            # --- TOP METRIKY ---
            m1, m2, m3, m4, m5 = st.columns(5)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            prev_p = info.get('previousClose', curr_p)
            m1.metric("🏢 Firma", info.get('shortName', ticker))
            m2.metric("💰 Cena", f"${curr_p:.2f}", f"{((curr_p-prev_p)/prev_p)*100:.2f}%")
            
            fair_v, fair_m = estimate_fair_value_dcf(info)
            if fair_v:
                upside = ((fair_v/curr_p)-1)*100
                m3.metric("🎯 Férová cena (DCF)", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            else: m3.metric("🎯 Férová cena", "N/A")
            
            m4.metric("📊 Market Cap", format_large_num(info.get('marketCap')))
            m5.metric("📈 P/E Ratio", info.get('trailingPE', 'N/A'))

            # --- TABS (ROZŠÍŘENÁ SADA) ---
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "📈 Tech. Graf", "🏛️ Historické Fundamenty", "📰 AI & Novinky", 
                "💼 Insider Trading", "🏥 Zdraví & Skóre", "🌍 Makro & Peers", "🧮 DCF Kalkulačka"
            ])

            with t1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='SMA 50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='SMA 200', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.4), row=2, col=1)
                fig.update_layout(height=650, template='plotly_white', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.subheader("📊 Hloubková Historie (Tržby, Zisk, FCF)")
                fin = stock_obj.financials.T
                cf = stock_obj.cashflow.T
                if not fin.empty and not cf.empty:
                    hist_df = pd.DataFrame({
                        "Tržby": fin['Total Revenue'] if 'Total Revenue' in fin.columns else 0,
                        "Čistý zisk": fin['Net Income'] if 'Net Income' in fin.columns else 0,
                        "FCF": cf['Free Cash Flow'] if 'Free Cash Flow' in cf.columns else 0
                    }).sort_index()
                    st.bar_chart(hist_df)
                    st.dataframe(hist_df.T, use_container_width=True)
                else:
                    st.warning("Historická data nejsou dostupná přes bezplatné API.")

            with t3:
                st.subheader("📰 AI Analýza & Aktuální Zprávy")
                news = stock_obj.news[:10]
                titles = [n.get('title') for n in news]
                if ai_on and api_key:
                    with st.spinner("AI studuje zprávy..."):
                        sent, summ = analyze_sentiment_ai(titles, api_key, ticker)
                        st.markdown(f"### AI Sentiment: {sent}")
                        st.info(summ)
                st.markdown("---")
                for n in news:
                    with st.expander(f"📰 {n.get('title')}"):
                        st.write(f"Zdroj: {n.get('publisher')}")
                        st.markdown(f"[🔗 Přečíst článek]({n.get('link')})")

            with t4:
                st.subheader("💼 Insider Trading (Transakce vedení)")
                ins = stock_obj.insider_transactions
                if ins is not None and not ins.empty:
                    st.dataframe(ins.head(30), use_container_width=True)
                else:
                    st.info("Data o insider trading nejsou momentálně dostupná.")

            with t5:
                st.subheader("🏥 Skóre finančního zdraví")
                # Jednoduché skóre založené na likviditě a dluhu
                de = safe_float(info.get('debtToEquity', 0))
                cr = safe_float(info.get('currentRatio', 0))
                score = 0
                if de and de < 100: score += 50
                if cr and cr > 1.2: score += 50
                st.metric("Finanční zdraví", f"{score}/100")
                st.progress(score / 100)

            with t6:
                st.subheader("🌍 Globální Makro Dashboard")
                macro = get_macro_dashboard()
                mcols = st.columns(len(macro))
                for i, (name, d) in enumerate(macro.items()):
                    mcols[i].metric(name, f"{d['val']:.2f}", f"{d['chg']:.2f}%")
                
                st.markdown("---")
                st.subheader("👥 Srovnání s peers (konkurencí)")
                peer_list = st.text_input("Zadejte tickery konkurentů (oddělené čárkou)", value="MSFT, GOOGL, AMZN")
                if peer_list:
                    peers = [ticker] + [x.strip().upper() for x in peer_list.split(",")]
                    p_res = []
                    for p in peers:
                        try:
                            pi = yf.Ticker(p).info
                            p_res.append({"Ticker": p, "P/E": pi.get('trailingPE'), "Marže": pi.get('profitMargins', 0)*100, "ROE": pi.get('returnOnEquity', 0)*100})
                        except: pass
                    st.table(pd.DataFrame(p_res))

            with t7:
                st.subheader("🧮 Interaktivní DCF Kalkulačka")
                st.caption("Namodelujte si férovou cenu podle vlastních očekávání růstu.")
                d1, d2 = st.columns([1, 2])
                with d1:
                    g_rate = st.slider("Roční růst FCF (5 let)", 0.0, 0.4, 0.15, 0.01)
                    d_rate = st.slider("Diskontní sazba (WACC)", 0.07, 0.20, 0.10, 0.01)
                with d2:
                    fcf_val = safe_float(info.get("freeCashflow"))
                    sh_val = safe_float(info.get("sharesOutstanding"))
                    if fcf_val and sh_val:
                        # Terminální hodnota přes Gordonův model
                        tv = (fcf_val * (1+g_rate)**5 * 1.025) / (d_rate - 0.025)
                        pv = (fcf_val * 5) + (tv / (1+d_rate)**5)
                        fair = pv / sh_val
                        st.metric("Vaše Férová Cena", f"${fair:.2f}", f"{((fair/curr_p)-1)*100:+.1f}% vs Trh")
                    else: st.warning("Chybí data pro výpočet DCF.")

            st.markdown('<div class="footer">Stock Analyzer Pro v10.0 | Data by Yahoo Finance & Google Gemini</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
