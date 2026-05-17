"""🧴 Skincare ML Recommender — Main Application"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ensemble_system import EnsembleHybridSystem as HybridRecommenderSystem
except ImportError:
    from hybrid_model import HybridRecommenderSystem

from analytics import SkincareAnalytics

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Skincare ML", page_icon="🌸", layout="wide")

# ── LOAD CSS ──────────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    try:
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# ── LOADERS ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in ['dataset/processed/skincare_processed.csv',
                 'dataset/processed/skincare_sample.csv']:
        try:
            df = pd.read_csv(path)
            df = df[df['brand_name'] != 'Unknown']
            return df
        except FileNotFoundError:
            continue
    return None

@st.cache_resource
def get_hybrid_system():
    return HybridRecommenderSystem()

def get_comparison_results():
    """Hasil evaluasi penelitian — hardcoded, tidak berubah."""
    return pd.DataFrame({
        'Model': ['TF-IDF Only', 'RF Only', 'Hybrid Ensemble'],
        'R²':    [0.635,          0.762,     0.836],
        'MAE':   [0.125,          0.088,     0.075],
        'RMSE':  [0.156,          0.126,     0.104],
    })

def check_models():
    return all(os.path.exists(f) for f in
               ['models/tfidf_model.pkl', 'models/rf_model.pkl'])

# ── CHART HELPERS ─────────────────────────────────────────────────────────────
PAL = ['#E8A4B8', '#B5C9B7', '#C97A94']

def comparison_chart(df):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['R² Score ↑', 'MAE ↓', 'RMSE ↓'],
                        specs=[[{"type":"bar"}]*3])
    for i, col in enumerate(['R²','MAE','RMSE'], 1):
        fig.add_trace(go.Bar(x=df['Model'], y=df[col], marker_color=PAL,
                             text=df[col].round(3), textposition='outside',
                             showlegend=False), row=1, col=i)
    fig.update_layout(height=370, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font_family='DM Sans',
                      margin=dict(t=40,b=10))
    fig.update_xaxes(tickangle=-20, tickfont_size=11)
    fig.update_yaxes(gridcolor='#F5D5E2', gridwidth=0.5)
    return fig

def improvement_chart(df):
    ens = df[df['Model']=='Hybrid Ensemble']['R²'].values[0]
    rows = [{'vs': f"vs {r['Model']}",
             'pct': ((ens - r['R²']) / r['R²']) * 100}
            for _, r in df.iterrows()
            if r['Model'] != 'Hybrid Ensemble' and r['R²'] > 0]
    if not rows:
        return None
    imp = pd.DataFrame(rows)
    fig = px.bar(imp, x='pct', y='vs', orientation='h', text='pct',
                 color_discrete_sequence=['#C97A94'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=230, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font_family='DM Sans',
                      xaxis_title='Peningkatan (%)', yaxis_title='',
                      margin=dict(t=10,b=10))
    fig.update_xaxes(gridcolor='#F5D5E2')
    return fig

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = load_data()
if df is None:
    st.error("❌ Dataset tidak ditemukan. Jalankan `python data_pipeline.py` dulu.")
    st.stop()

st.session_state.df = df
hybrid_system = get_hybrid_system()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
  <h1>🌸 Skincare Recommender</h1>
  <p style="margin-top:-0.4rem; font-size:0.88rem;">
    Hybrid Ensemble Learning &nbsp;·&nbsp; TF-IDF &nbsp;·&nbsp; SVD &nbsp;·&nbsp; Random Forest
  </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dataset")
    st.metric("Produk", f"{len(df):,}")
    st.metric("Brand", f"{df['brand_name'].nunique()}")
    st.metric("Kategori", f"{df['default_category'].nunique()}")
    st.info(f"{'Full' if len(df)>3000 else 'Sample'} Dataset")
    st.markdown("---")
    st.markdown("### Model")
    if not st.session_state.models_trained:
        if check_models():
            if st.button("📥 Load Models", use_container_width=True):
                with st.spinner("Memuat..."):
                    if hybrid_system.load_trained_models():
                        st.session_state.models_trained = True
                        st.rerun()
            st.caption("atau")
            if st.button("🔄 Retrain", use_container_width=True):
                with st.spinner("Training..."):
                    hybrid_system.train_all_models()
                    st.session_state.models_trained = True
                    st.rerun()
        else:
            if st.button("⚡ Train Models", type="primary", use_container_width=True):
                with st.spinner("Training (~2-3 menit)..."):
                    hybrid_system.train_all_models()
                    st.session_state.models_trained = True
                    st.rerun()
    else:
        st.success("✅ Models Ready")

# ── METRIC STRIP ──────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col, label, val in zip(
    [c1,c2,c3,c4],
    ["Produk","Brand","Kategori","Avg Rating"],
    [f"{len(df):,}", f"{df['brand_name'].nunique()}",
     f"{df['default_category'].nunique()}", f"{df['average_rating'].mean():.2f}"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Rekomendasi", "🔍 Produk Serupa",
    "✨ Prediksi Rating", "📊 Model Comparison", "📈 Analytics"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Cari Produk Serupa")
    if not st.session_state.models_trained:
        st.warning("⚠️ Load atau train model dulu via sidebar.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            bf = st.selectbox("Brand", ["Semua"]+sorted(df['brand_name'].unique()), key="r1_brand")
        fdf = df[df['brand_name']==bf] if bf!="Semua" else df.copy()
        with c2:
            cf = st.selectbox("Kategori", ["Semua"]+sorted(fdf['default_category'].unique()), key="r1_cat")
        if cf!="Semua": fdf=fdf[fdf['default_category']==cf]
        with c3:
            prod = st.selectbox("Produk", fdf['product_name'].head(200), key="r1_prod")

        sel = df[df['product_name']==prod].iloc[0]
        with st.expander("Detail produk terpilih"):
            st.markdown(f"""
            <div class="rec-card">
            <p><b>{sel['product_name']}</b></p>
            <p>{sel['brand_name']} · {sel['default_category']} · ⭐ {sel['average_rating']:.2f}</p>
            </div>""", unsafe_allow_html=True)

        if st.button("🔍 Temukan Produk Serupa", type="primary", use_container_width=True):
            try:
                with st.spinner("Menganalisis..."):
                    pidx = df[df['product_name']==prod].index[0]
                    similar = hybrid_system.get_similar_products(pidx, n_recommendations=5)
                if similar.empty:
                    st.warning("Produk serupa tidak ditemukan.")
                else:
                    st.success("✅ Ditemukan 5 produk serupa!")
                    rows, raw = [], []
                    for i,idx in enumerate(similar.index, 1):
                        orig = df.index.get_loc(idx)
                        sc = hybrid_system.get_ensemble_score(orig)
                        p = df.loc[idx]
                        nm = (p['product_name'][:38]+"…") if len(p['product_name'])>38 else p['product_name']
                        rows.append({"#":i,"Produk":nm,"Brand":p['brand_name'],
                                     "Rating":f"⭐{p['average_rating']:.2f}",
                                     "TF-IDF":f"{sc['tfidf_score']:.3f}",
                                     "SVD":f"{sc['svd_score']:.3f}",
                                     "RF":f"{sc['rf_score']:.3f}",
                                     "Ensemble":f"{sc['ensemble_score']:.3f}"})
                        raw.append((p, sc))

                    avg_e = np.mean([r[1]['ensemble_score'] for r in raw])
                    rows.append({"#":"","Produk":"Rata-rata","Brand":"","Rating":"",
                                 "TF-IDF":"","SVD":"","RF":"",
                                 "Ensemble":f"{avg_e:.3f}"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    for i,(p,sc) in enumerate(raw,1):
                        pn = p.get('price_numeric',0)
                        price_str = f"Rp{pn:,}" if pn else "N/A"
                        st.markdown(f"""
                        <div class="rec-card">
                          <h4>{i}. {p['product_name']}</h4>
                          <p>{p['brand_name']} · {p['default_category']} · {price_str}</p>
                          <p>TF-IDF <span class="badge">{sc['tfidf_score']:.3f}</span>
                             SVD <span class="badge">{sc['svd_score']:.3f}</span>
                             RF <span class="badge">{sc['rf_score']:.3f}</span>
                             Ensemble <span class="badge badge-best">{sc['ensemble_score']:.3f}</span></p>
                        </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Find Similar Products")
    if not st.session_state.models_trained:
        st.warning("⚠️ Load model dulu.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            bf2 = st.selectbox("Brand",["Semua"]+sorted(df['brand_name'].unique()),key="t2_brand")
        fdf2 = df[df['brand_name']==bf2] if bf2!="Semua" else df
        with c2:
            prod2 = st.selectbox("Produk",fdf2['product_name'].head(100),key="t2_prod")
        if st.button("🔍 Find Similar", use_container_width=True):
            try:
                pidx2 = df[df['product_name']==prod2].index[0]
                sim2 = hybrid_system.get_similar_products(pidx2, n_recommendations=5)
                if not sim2.empty:
                    for i,(_,p) in enumerate(sim2.iterrows(),1):
                        st.markdown(f"""
                        <div class="rec-card">
                          <h4>{i}. {p['product_name']}</h4>
                          <p>{p['brand_name']} · ⭐ {p['average_rating']:.2f} ·
                          Similarity: <b>{p.get('similarity_score',0):.3f}</b></p>
                        </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Prediksi Rating Produk")
    if not st.session_state.models_trained:
        st.warning("⚠️ Load model dulu.")
    else:
        mode = st.radio("Mode:", ["📦 Produk yang Ada","✨ Produk Baru"], horizontal=True)

        if mode == "📦 Produk yang Ada":
            c1,c2 = st.columns([1,2])
            with c1:
                bf3 = st.selectbox("Brand",["All"]+sorted(df['brand_name'].unique()[:50]),key="t3_brand")
                fdf3 = df[df['brand_name']==bf3] if bf3!="All" else df
                prod3 = st.selectbox("Produk",fdf3['product_name'].head(200),key="t3_prod")
            with c2:
                if st.button("🎯 Predict",type="primary",use_container_width=True):
                    try:
                        p3 = df[df['product_name']==prod3].iloc[0]
                        orig3 = df.index.get_loc(df[df['product_name']==prod3].index[0])
                        sc3 = hybrid_system.get_ensemble_score(orig3)
                        pred_r = sc3['ensemble_score']*5.0
                        act_r  = p3['average_rating']
                        ca,cb = st.columns(2)
                        ca.metric("Actual Rating", f"⭐ {act_r:.2f}")
                        cb.metric("Predicted Rating", f"⭐ {pred_r:.2f}", delta=f"{pred_r-act_r:+.2f}")
                        st.dataframe(pd.DataFrame({
                            'Model':  ['TF-IDF','SVD','RF','Ensemble'],
                            'Score':  [sc3['tfidf_score'],sc3['svd_score'],sc3['rf_score'],sc3['ensemble_score']],
                            'Rating': [sc3['tfidf_score']*5,sc3['svd_score']*5,sc3['rf_score']*5,pred_r],
                        }), use_container_width=True, hide_index=True)
                        err = abs(pred_r-act_r)
                        if err<=0.1:   st.success("🎯 Excellent — error ≤ 0.1")
                        elif err<=0.3: st.info("✅ Good — error ≤ 0.3")
                        elif err<=0.5: st.warning("⚠️ Moderate — error ≤ 0.5")
                        else:          st.error("❌ Error > 0.5")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            c1,c2 = st.columns(2)
            with c1:
                new_nm = st.text_input("Nama Produk", placeholder="e.g., Anessa Sunscreen")
                bn = st.selectbox("Brand", sorted(df['brand_name'].unique()[:50]), key="t3_bnew")
                cn = st.selectbox("Kategori", sorted(df['default_category'].unique()), key="t3_cnew")
            with c2:
                rn = st.number_input("Estimasi Reviews",0,10000,100)
                wn = st.number_input("Estimasi Wishlist",0,5000,25)
                pn = st.number_input("Harga (Rp)",0,5000000,150000,step=10000)
            if st.button("✨ Predict Rating",type="primary",use_container_width=True):
                try:
                    pred_n = hybrid_system.predict_rating(bn,cn,rn,wn,pn)
                    st.markdown(f"""
                    <div class="best-card" style="text-align:center;">
                      <h3 style="font-size:1rem!important;font-style:normal!important;">
                        Predicted Rating — <em>{new_nm or 'produk baru'}</em>
                      </h3>
                      <div style="font-family:'Playfair Display',serif;font-size:3rem;
                                  color:#C97A94;line-height:1.2;">
                        ⭐ {pred_n:.2f}
                        <span style="font-size:1.1rem;color:#7A5C6A;">/5.0</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    if pred_n>=4.5:   st.success("🎉 Excellent Potential!")
                    elif pred_n>=4.0: st.info("✅ Good Potential")
                    elif pred_n>=3.5: st.warning("⚠️ Average Potential")
                    else:             st.error("❌ Low Potential")
                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL COMPARISON (hardcoded)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Model Performance Comparison")
    st.caption("Hasil evaluasi dari penelitian — TF-IDF, Random Forest, dan Hybrid Ensemble.")

    res = get_comparison_results()
    best = res.loc[res['R²'].idxmax()]

    st.markdown(f"""
    <div class="best-card">
      <h3>🏆 Best Model &nbsp;—&nbsp; {best['Model']}</h3>
      <p>R² <b>{best['R²']:.3f}</b> &nbsp;·&nbsp;
         MAE <b>{best['MAE']:.3f}</b> &nbsp;·&nbsp;
         RMSE <b>{best['RMSE']:.3f}</b></p>
      <p style="font-size:0.8rem;margin-top:0.3rem;">
        Hybrid Ensemble mengintegrasikan TF-IDF (content-based),
        SVD (collaborative filtering), dan Random Forest (metadata)
        melalui meta-learner untuk hasil prediksi terbaik.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### Tabel Evaluasi")
    st.dataframe(res, use_container_width=True, hide_index=True)

    st.markdown("##### Visualisasi")
    st.plotly_chart(comparison_chart(res), use_container_width=True)

    fig_i = improvement_chart(res)
    if fig_i:
        st.markdown("##### Peningkatan Ensemble vs Model Individual")
        st.plotly_chart(fig_i, use_container_width=True)

    ens_r2 = res[res['Model']=='Hybrid Ensemble']['R²'].values[0]
    tf_r2  = res[res['Model']=='TF-IDF Only']['R²'].values[0]
    rf_r2  = res[res['Model']=='RF Only']['R²'].values[0]
    c1,c2,c3 = st.columns(3)
    c1.metric("Ensemble R²", f"{ens_r2:.3f}")
    c2.metric("vs TF-IDF",   f"+{((ens_r2-tf_r2)/tf_r2)*100:.1f}%")
    c3.metric("vs RF Only",  f"+{((ens_r2-rf_r2)/rf_r2)*100:.1f}%")
    
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Dataset Analytics")
    analytics = SkincareAnalytics()
    analytics.df = df
    analytics.render_dashboard()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1rem;
border-top:1px solid #EDD5DF;color:#B89BA5;font-size:0.76rem;">
🌸 Skincare Recommender &nbsp;·&nbsp; Hybrid Ensemble Learning &nbsp;·&nbsp; 2026
</div>""", unsafe_allow_html=True)