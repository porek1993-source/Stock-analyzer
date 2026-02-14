"""
Stock Analysis Pro - Masterpiece Edition v9.5
Komplexní nástroj pro hloubkovou analýzu trhu
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

# Profesionální CSS Dark Mode Styling
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem; }
    .stMetric { background-color: #1e293b; border-radius: 10px; padding: 15px !important; border-left: 5px solid #3b82f6; }
    .sentiment-positive { background-color: #065f46; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .sentiment-negative { background-color: #991b1b; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .sentiment-neutral { background-color: #475569; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px; }
    .footer { text-align: center; color: #64748b; padding: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. LOGICKÉ MODULY (DATA & VÝPOČTY)
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

@st.cache_data(ttl=3600)
def get_macro_data():
    """Získává globální makro indikátory (včetně dluhopisů a VIX)"""
    tickers = {
        "S&P 500": "^GSPC",
        "US 10Y Výnosy": "^TNX",
        "VIX (Index Strachu)": "^VIX",
        "Dolar (DXY)": "DX-Y.NYB",
        "Zlato": "GC=F",
        "Ropa (Brent)": "BZ=F"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                curr, prev = h["Close"].iloc[-1], h["Close"].iloc[-2]
                data[name] = {"val": curr, "chg": ((curr-prev)/prev)*100}
        except: pass
    return data

def estimate_fair_value_dcf(info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Hloubkový DCF Model - Vnitřní hodnota na základě Cash Flow"""
    try:
        fcf = safe_float(info.get("freeCashflow"))
        shares = safe_float(info.get("sharesOutstanding"))
        if fcf and shares and fcf > 0:
            # Parametry: 5 let růst 5%, diskont 10%, terminální růst 2%
            pv_fcf = 0; growth = 0.05; discount = 0.10; terminal = 0.02
            temp_fcf = fcf
            for i in range(1, 6):
                temp_fcf *= (1 + growth)
                pv_fcf += temp_fcf / ((1 + discount) ** i)
            tv = (temp_fcf * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / ((1 + discount) ** 5)
            return (pv_fcf + pv_tv) / shares, "DCF Model (Vnitřní hodnota)"
        
        tmean = safe_float(info.get("targetMeanPrice"))
        if tmean: return tmean, "Analyst Target (Mean)"
    except: pass
    return None, "N/A"

def calculate_health_score(info: Dict) -> Tuple[int, str, str]:
    """Komplexní skóring finančního zdraví (0-100)"""
    score = 0
    checks = []
    
    # 1. Zadluženost
    de = safe_float(info.get('debtToEquity'))
    if de is not None:
        if de < 80: score += 20; checks.append("✅ Nízký dluh")
        elif de < 150: score += 10; checks.append("⚠️ Mírný dluh")
        else: checks.append("❌ Vysoký dluh")
    
    # 2. Likvidita
    cr = safe_float(info.get('currentRatio'))
    if cr is not None:
        if cr > 1.5: score += 20; checks.append("✅ Dobrá likvidita")
        else: checks.append("❌ Slabá likvidita")
        
    # 3. Ziskovost (ROE)
    roe = safe_float(info.get('returnOnEquity'))
    if roe and roe > 0.15: score += 20; checks.append("✅ Vysoká rentabilita")
    
    # 4. Cash Flow
    fcf = safe_float(info.get('freeCashflow'))
    if fcf and fcf > 0: score += 20; checks.append("✅ Pozitivní Cash Flow")
    
    # 5. Marže
    margin = safe_float(info.get('profitMargins'))
    if margin and margin > 0.15: score += 20; checks.append("✅ Zdravé marže")

    status = "🟢 EXCELENTNÍ" if score >= 80 else ("🟡 STABILNÍ" if score >= 50 else "🔴 RIZIKOVÉ")
    return score, status, "\n".join(checks)

# ============================================================================
# 3. AI & NEWS MODUL (OPRAVENÝ)
# ============================================================================

def get_robust_news(ticker):
    """Robustní extrakce titulků pro zamezení chybám 'Bez názvu'"""
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news[:10] if hasattr(stock, 'news') else []
        refined = []
        for item in raw_news:
            title = item.get('title') or item.get('headline') or "Zpráva bez názvu"
            pub = item.get('publisher') or "Yahoo Finance"
            link = item.get('link') or item.get('url')
            if title != "Zpráva bez názvu":
                refined.append({"title": title, "publisher": pub, "link": link})
        return refined
    except: return []

def analyze_sentiment_gemini(news_titles, api_key, ticker):
    """AI Analýza pomocí Gemini 2.0 Pro/Flash"""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        valid = [t for t in news_titles if len(t) > 10]
        if not valid: return "Neutrální", "Nedostatek dat pro AI analýzu."
        
        prompt = f"Jsi seniorní analytik na Wall Street. Analyzuj dopad těchto zpráv na {ticker}:\n" + "\n".join(valid[:8]) + \
                 "\n\nOdpověz ve formátu:\nSENTIMENT: [Pozitivní/Negativní/Neutrální]\nSHRNUTÍ: [Max 3 věty v češtině]"
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text
        sent = "Neutrální"
        summ = text
        for line in text.split('\n'):
            if 'SENTIMENT:' in line.upper(): sent = line.split(':')[-1].strip().replace('[','').replace(']','')
            elif 'SHRNUTÍ:' in line.upper(): summ = line.split(':')[-1].strip()
        return sent, summ
    except Exception as e:
        return "Neutrální", f"AI nedostupná: {str(e)}"

# ============================================================================
# 4. HLAVNÍ UI APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Ultimate Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Vyhledávání")
        ticker_symbol = st.text_input("Zadejte Ticker (např. AAPL, TSLA, NVDA)", value="AAPL").upper()
        time_period = st.selectbox("Časový horizont", ["1y", "2y", "5y", "10y", "max"], index=1)
        st.markdown("---")
        ai_on = st.checkbox("🤖 Povolit AI (Gemini 2.0)", value=False)
        gemini_api = st.text_input("Vložte API Klíč", type="password") if ai_on else ""
        st.markdown("---")
        st.info("💡 Tip: Sledujte 'Makro Dashboard' pro pochopení trendů trhu.")
        analyze_btn = st.button("🚀 SPUSTIT KOMPLETNÍ ANALÝZU", type="primary", use_container_width=True)

    if analyze_btn or ticker_symbol:
        with st.spinner(f"Doluji data pro {ticker_symbol}..."):
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period=time_period)
            info = stock.info
            
            if df.empty:
                st.error("❌ Ticker nebyl nalezen nebo Yahoo Finance neposkytuje data.")
                return

            # --- TOP HEADER METRIKY ---
            col1, col2, col3, col4, col5 = st.columns(5)
            price = info.get('currentPrice', df['Close'].iloc[-1])
            prev = info.get('previousClose', price)
            
            col1.metric("🏢 Společnost", info.get('shortName', ticker_symbol))
            col2.metric("💰 Cena", f"${price:.2f}", f"{((price-prev)/prev)*100:.2f}%")
            
            fair_v, fair_m = estimate_fair_value_dcf(info)
            if fair_v:
                upside = ((fair_v/price)-1)*100
                col3.metric("🎯 Férová cena", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            else: col3.metric("🎯 Férová cena", "N/A")
            
            col4.metric("📊 Market Cap", format_large_num(info.get('marketCap')))
            col5.metric("📈 P/E Ratio", info.get('trailingPE', 'N/A'))

            # --- TAB SYSTÉM (Kompletní sada) ---
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "📈 Technický Graf", "🏛️ Fundamenty", "📰 AI & Novinky", 
                "💼 Insider Trading", "🏥 Zdraví & Skóre", "🌍 Makro & Peers", "🧮 DCF Simulátor"
            ])

            # TAB 1: Graf
            with t1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='SMA 50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), name='SMA 200', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.4), row=2, col=1)
                fig.update_layout(height=650, template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # TAB 2: Fundamenty
            with t2:
                st.subheader("📊 Klíčové Finanční Ukazatele")
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.write("**Valuace**")
                    st.write(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
                    st.write(f"PEG Ratio: {info.get('pegRatio', 'N/A')}")
                with f2:
                    st.write("**Ziskovost**")
                    st.write(f"Marže: {info.get('profitMargins', 0)*100:.2f}%")
                    st.write(f"ROE: {info.get('returnOnEquity', 0)*100:.2f}%")
                with f3:
                    st.write("**Dividendy**")
                    st.write(f"Výnos: {info.get('dividendYield', 0)*100:.2f}%")
                    st.write(f"Výplatní poměr: {info.get('payoutRatio', 0)*100:.2f}%")
                
                st.markdown("---")
                st.subheader("📜 Historie Tržeb (Financials)")
                fin = stock.financials.T
                if not fin.empty:
                    st.bar_chart(fin[['Total Revenue', 'Net Income']] if 'Total Revenue' in fin.columns else fin)

            # TAB 3: AI & Novinky
            with t3:
                st.subheader("📰 AI Analýza & Aktuální Zprávy")
                news_data = get_robust_news(ticker_symbol)
                titles = [n['title'] for n in news_data]
                
                if ai_on and gemini_api:
                    with st.spinner("🧠 AI studuje poslední zprávy..."):
                        sent, summ = analyze_sentiment_gemini(titles, gemini_api, ticker_symbol)
                        st.markdown(f"### AI Sentiment: {sent}")
                        st.info(summ)
                
                st.markdown("---")
                for n in news_data:
                    with st.expander(f"📰 {n['title']}"):
                        st.write(f"**Zdroj:** {n['publisher']}")
                        if n['link']: st.markdown(f"[🔗 Odkaz na článek]({n['link']})")

            # TAB 4: Insider Trading
            with t4:
                st.subheader("💼 Transakce managementu")
                ins = stock.insider_transactions
                if ins is not None and not ins.empty:
                    st.dataframe(ins.head(30), use_container_width=True)
                else:
                    st.info("ℹ️ Pro tento ticker nejsou data o insider trading momentálně dostupná.")

            # TAB 5: Skóre zdraví
            with t5:
                score, status, checks = calculate_health_score(info)
                s1, s2 = st.columns([1, 2])
                with s1:
                    st.metric("Skóre Zdraví", f"{score}/100")
                    st.markdown(f"### Status: {status}")
                with s2:
                    st.markdown("### 📋 Kontrolní seznam")
                    st.write(checks)
                st.progress(score / 100)

            # TAB 6: Makro & Peers
            with t6:
                st.subheader("🌍 Globální Tržní Kontext")
                macro = get_macro_data()
                if macro:
                    mcols = st.columns(len(macro))
                    for i, (name, d) in enumerate(macro.items()):
                        mcols[i].metric(name, f"{d['val']:.2f}", f"{d['chg']:.2f}%")
                
                st.markdown("---")
                st.subheader("👥 Srovnání s Konkurencí")
                peers_in = st.text_input("Tickery konkurence (oddělené čárkou)", value="MSFT, GOOGL, AMZN")
                if peers_in:
                    all_p = [ticker_symbol] + [x.strip().upper() for x in peers_in.split(",")]
                    p_list = []
                    for p in all_p:
                        try:
                            pi = yf.Ticker(p).info
                            p_list.append({"Ticker": p, "Cena": pi.get('currentPrice'), "P/E": pi.get('trailingPE'), "ROE": pi.get('returnOnEquity')})
                        except: pass
                    st.table(pd.DataFrame(p_list))

            # TAB 7: DCF Simulátor
            with t7:
                st.subheader("🧮 Interaktivní DCF Simulátor")
                st.caption("Namodelujte si férovou cenu podle svých odhadů.")
                d1, d2 = st.columns([1, 2])
                with d1:
                    g_rate = st.slider("Očekávaný růst (5 let)", 0.0, 0.4, 0.15, 0.01)
                    d_rate = st.slider("Diskontní sazba", 0.07, 0.20, 0.10, 0.01)
                with d2:
                    fcf = safe_float(info.get("freeCashflow"))
                    sh = safe_float(info.get("sharesOutstanding"))
                    if fcf and sh:
                        # Rychlá kalkulace terminální hodnoty
                        tv_fair = (fcf * (1+g_rate)**5 * 1.02) / (d_rate - 0.02)
                        pv_total = (fcf * 5) + (tv_fair / (1+d_rate)**5)
                        fair_price = pv_total / sh
                        st.metric("Tvoje Férová Cena", f"${fair_price:.2f}", f"{((fair_price/price)-1)*100:+.1f}% vs Trh")
                    else: st.warning("⚠️ Chybí data pro Cash Flow nebo Shares Outstanding.")

            st.markdown('<div class="footer">Aplikace Stock Analyzer Pro v9.5 | Data provided by Yahoo Finance</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
