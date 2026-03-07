import streamlit as st
import numpy as np
import tensorflow as tf
import os

st.set_page_config(
    page_title="Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
  html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; background:#0D0F14; color:#E8E8F0; }
  .main { background:#0D0F14; }
  .block-container { padding-top:2rem; }

  .hero {
      background:linear-gradient(135deg,#1a1d27 0%,#12141c 100%);
      border:1px solid #2a2d3a; border-radius:20px;
      padding:38px 50px; margin-bottom:28px; position:relative; overflow:hidden;
  }
  .hero::before {
      content:''; position:absolute; top:-60px; right:-60px;
      width:260px; height:260px;
      background:radial-gradient(circle,rgba(99,102,241,.13) 0%,transparent 70%);
      border-radius:50%;
  }
  .hero-title {
      font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800;
      background:linear-gradient(90deg,#a5b4fc,#818cf8,#6366f1);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 6px;
  }
  .hero-sub { color:#8b8fa8; font-size:1rem; margin:0; }

  .section-card {
      background:#13151f; border:1px solid #1e2130;
      border-radius:16px; padding:26px 30px; margin-bottom:20px;
  }
  .section-title {
      font-family:'Syne',sans-serif; font-size:.85rem; font-weight:700;
      color:#a5b4fc; letter-spacing:.1em; text-transform:uppercase; margin-bottom:18px;
  }

  .stButton>button {
      width:100%;
      background:linear-gradient(135deg,#6366f1,#818cf8);
      color:white; border:none; border-radius:12px;
      padding:16px 32px; font-family:'Syne',sans-serif;
      font-size:1.1rem; font-weight:700; letter-spacing:.04em; cursor:pointer;
  }
  .stButton>button:hover {
      background:linear-gradient(135deg,#4f46e5,#6366f1);
      transform:translateY(-1px); box-shadow:0 8px 25px rgba(99,102,241,.35);
  }

  .result-high { background:linear-gradient(135deg,#052e16,#064e3b); border:1px solid #059669; border-radius:16px; padding:30px; text-align:center; }
  .result-low  { background:linear-gradient(135deg,#27180a,#3a2108); border:1px solid #d97706; border-radius:16px; padding:30px; text-align:center; }
  .result-label { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; margin-bottom:6px; }
  .result-sub   { color:#94a3b8; font-size:.92rem; margin-bottom:14px; }
  .prob-bar-wrap{ background:#1e2130; border-radius:8px; height:10px; margin:12px 0 6px; overflow:hidden; }
  .prob-bar-fill{ height:100%; border-radius:8px; }

  .metric-row { display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }
  .metric-tile {
      flex:1; min-width:100px; background:#1a1d27;
      border:1px solid #2a2d3a; border-radius:12px; padding:16px 12px; text-align:center;
  }
  .metric-val { font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; color:#a5b4fc; }
  .metric-lbl { font-size:.72rem; color:#6b7280; margin-top:3px; }

  label { color:#c4c8e0 !important; font-size:.87rem !important; }
  hr    { border-color:#1e2130; }
</style>
""", unsafe_allow_html=True)


NUM_COLS = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']


OHE = {
    'workclass': [
        'Local-gov','Private','Self-emp-inc',
        'Self-emp-not-inc','State-gov','Without-pay'
    ],
    'education': [
        '11th','12th','1st-4th','5th-6th','7th-8th','9th',
        'Assoc-acdm','Assoc-voc','Bachelors','Doctorate',
        'HS-grad','Masters','Preschool','Prof-school','Some-college'
    ],
    'marital.status': [
        'Married-AF-spouse','Married-civ-spouse','Married-spouse-absent',
        'Never-married','Separated','Widowed'
    ],
    'occupation': [
        'Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing',
        'Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv',
        'Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving'
    ],
    'relationship': [
        'Not-in-family','Other-relative','Own-child','Unmarried','Wife'
    ],
    'race': ['Asian-Pac-Islander','Black','Other','White'],
    'sex':  ['Male'],   
    'native.country': [
        'Canada','China','Columbia','Cuba','Dominican-Republic','Ecuador',
        'El-Salvador','England','France','Germany','Greece','Guatemala',
        'Haiti','Holand-Netherlands','Honduras','Hong','Hungary','India',
        'Iran','Ireland','Italy','Jamaica','Japan','Laos','Mexico',
        'Nicaragua','Outlying-US(Guam-USVI-etc)','Peru','Philippines',
        'Poland','Portugal','Puerto-Rico','Scotland','South','Taiwan',
        'Thailand','Trinadad&Tobago','United-States','Vietnam','Yugoslavia'
    ],
}

WORKCLASS_UI   = ['Federal-gov'] + OHE['workclass']
EDUCATION_UI   = ['10th']        + OHE['education']
MARITAL_UI     = ['Divorced']    + OHE['marital.status']
OCCUPATION_UI  = ['Adm-clerical']+ OHE['occupation']
RELATION_UI    = ['Husband']     + OHE['relationship']
RACE_UI        = ['Amer-Indian-Eskimo'] + OHE['race']
SEX_UI         = ['Female','Male']
COUNTRY_UI     = ['Cambodia']    + OHE['native.country']


@st.cache_resource
def load_model():
    for name in ["model.keras","adult_model.keras","income_model.keras",
                 "model.h5","adult_model.h5","income_model.h5"]:
        if os.path.exists(name):
            return tf.keras.models.load_model(name)
    return None

model = load_model()


def build_vector(d: dict) -> np.ndarray:
    row = [float(d['age']), float(d['education.num']),
           float(d['capital.gain']), float(d['capital.loss']),
           float(d['hours.per.week'])]

    for feat, categories in OHE.items():
        val = d[feat]
        for cat in categories:
            row.append(1.0 if val == cat else 0.0)

    arr = np.array(row, dtype=np.float32).reshape(1, -1)
    assert arr.shape[1] == 95, f"Vector length {arr.shape[1]} ≠ 95"
    return arr

st.markdown("""
<div class="hero">
  <p class="hero-title">💼 Income Predictor</p>
  <p class="hero-sub">Adult Census Income · Deep Neural Network · Predicts whether annual income exceeds $50K</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ **Model file not found.** Place `model.keras` next to `app.py` then restart.", icon="🔌")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-card"><div class="section-title">👤 Personal Info</div>', unsafe_allow_html=True)
    age            = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
    sex            = st.selectbox("Sex",            SEX_UI)
    race           = st.selectbox("Race",           RACE_UI)
    native_country = st.selectbox("Native Country", COUNTRY_UI)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🎓 Education</div>', unsafe_allow_html=True)
    education     = st.selectbox("Education Level", EDUCATION_UI)
    education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card"><div class="section-title">🏢 Work Info</div>', unsafe_allow_html=True)
    workclass      = st.selectbox("Work Class",  WORKCLASS_UI)
    occupation     = st.selectbox("Occupation",  OCCUPATION_UI)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">👨‍👩‍👧 Family</div>', unsafe_allow_html=True)
    marital_status = st.selectbox("Marital Status", MARITAL_UI)
    relationship   = st.selectbox("Relationship",   RELATION_UI)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">💰 Capital</div>', unsafe_allow_html=True)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
col_btn, col_res = st.columns([1, 2], gap="large")

with col_btn:
    predict_clicked = st.button("🔮  Predict Income", use_container_width=True)

with col_res:
    if predict_clicked:
        d = {
            'age': age, 'education.num': education_num,
            'capital.gain': capital_gain, 'capital.loss': capital_loss,
            'hours.per.week': hours_per_week,
            'workclass': workclass, 'education': education,
            'marital.status': marital_status, 'occupation': occupation,
            'relationship': relationship, 'race': race,
            'sex': sex, 'native.country': native_country,
        }

        X = build_vector(d)

        if model is not None:
            prob_high = float(model.predict(X, verbose=0)[0][0])
            is_high   = prob_high >= 0.5

            if is_high:
                st.markdown(f"""
                <div class="result-high">
                  <div class="result-label" style="color:#34d399">🟢 Income &gt; $50K</div>
                  <div class="result-sub">High income bracket predicted by the neural network</div>
                  <div class="prob-bar-wrap"><div class="prob-bar-fill"
                    style="width:{prob_high*100:.1f}%;background:linear-gradient(90deg,#059669,#34d399)"></div></div>
                  <div style="color:#34d399;font-weight:600;font-size:1.1rem">{prob_high*100:.1f}% confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                  <div class="result-label" style="color:#f59e0b">🟡 Income ≤ $50K</div>
                  <div class="result-sub">Lower income bracket predicted by the neural network</div>
                  <div class="prob-bar-wrap"><div class="prob-bar-fill"
                    style="width:{(1-prob_high)*100:.1f}%;background:linear-gradient(90deg,#92400e,#f59e0b)"></div></div>
                  <div style="color:#f59e0b;font-weight:600;font-size:1.1rem">{(1-prob_high)*100:.1f}% confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile"><div class="metric-val">{age}</div><div class="metric-lbl">Age</div></div>
              <div class="metric-tile"><div class="metric-val">{hours_per_week}h</div><div class="metric-lbl">Hrs/Week</div></div>
              <div class="metric-tile"><div class="metric-val">{education_num}</div><div class="metric-lbl">Edu Num</div></div>
              <div class="metric-tile"><div class="metric-val">${capital_gain:,}</div><div class="metric-lbl">Cap. Gain</div></div>
              <div class="metric-tile"><div class="metric-val">${capital_loss:,}</div><div class="metric-lbl">Cap. Loss</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Model not loaded — add `model.keras` next to `app.py` and restart.", icon="🔌")

st.markdown("""
<br><hr>
<p style="text-align:center;color:#3d4155;font-size:.78rem">
  Adult Census Income Predictor · Keras Sequential (128→64→32→1) · 95 features · Binary Classification
</p>
""", unsafe_allow_html=True)