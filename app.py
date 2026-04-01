# ============================================================
#  Smart Resume Analyzer — Streamlit App
#  Run:  streamlit run app.py
#  Needs: model.pkl  tfidf.pkl  label_encoder.pkl
# ============================================================
import streamlit as st
import pickle, re, os
import PyPDF2
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le    = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, tfidf, le

model, tfidf, le = load_models()
STOP_WORDS = set(stopwords.words('english'))

# ── Skills lexicon ────────────────────────────────────────────
SKILLS = {
    'Programming'  : ['python','java','c++','c#','javascript','typescript','go','kotlin','swift','r','scala'],
    'Web Dev'      : ['html','css','react','angular','vue','node','django','flask','spring','bootstrap'],
    'Data / ML'    : ['machine learning','deep learning','nlp','tensorflow','pytorch','keras','scikit','pandas','numpy','data science'],
    'Databases'    : ['sql','mysql','postgresql','mongodb','redis','elasticsearch','oracle'],
    'Cloud/DevOps' : ['aws','azure','gcp','docker','kubernetes','jenkins','terraform','linux','git'],
    'Business'     : ['project management','agile','scrum','jira','excel','powerbi','tableau','sap'],
}

def clean_resume(text):
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)

def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    return ' '.join(p.extract_text() or '' for p in reader.pages)

def get_skills(text):
    tl = text.lower()
    return {cat: [s for s in sk if s in tl]
            for cat, sk in SKILLS.items() if any(s in tl for s in sk)}

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title='Smart Resume Analyzer',
    page_icon='📄',
    layout='centered'
)

# ── Header ────────────────────────────────────────────────────
st.title('📄 Smart Resume Analyzer')
st.markdown('**Upload your resume → AI predicts your best-match job role**')
st.markdown('---')

# ── Sidebar info ─────────────────────────────────────────────
with st.sidebar:
    st.header('ℹ️ About')
    st.markdown("""
    **Model:** Soft Voting Ensemble  
    (Logistic Regression + LinearSVC + MLP)

    **Features:** TF-IDF (50,000 features,  
    unigrams + bigrams)

    **Categories:** 25 job roles  
    **Dataset:** 2,484 labelled resumes
    """)
    st.markdown('---')
    st.caption('CSE — ML Lab Project')

# ── File upload ───────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader('Upload Resume (PDF only)', type=['pdf'])
with col2:
    st.markdown('<br>', unsafe_allow_html=True)
    analyze_btn = st.button('🔍 Analyze Resume', type='primary',
                             disabled=(uploaded_file is None))

# ── Analysis ─────────────────────────────────────────────────
if uploaded_file and analyze_btn:
    with st.spinner('Extracting text from PDF...'):
        raw_text = extract_pdf_text(uploaded_file)

    if not raw_text.strip():
        st.error('❌ Could not extract text. Please use a text-based PDF (not a scanned image).')
        st.stop()

    word_count = len(raw_text.split())
    st.info(f'📑 Extracted {word_count:,} words from your resume')

    with st.spinner('Analyzing...'):
        cleaned = clean_resume(raw_text)
        vec     = tfidf.transform([cleaned])
        proba   = model.predict_proba(vec)[0]
        top3_ix = proba.argsort()[-3:][::-1]

    # ── Predicted roles ───────────────────────────────────────
    st.markdown('### 🎯 Predicted Job Roles')
    medals = ['🥇', '🥈', '🥉']
    for rank, idx in enumerate(top3_ix):
        role = le.classes_[idx]
        conf = float(proba[idx])
        if rank == 0:
            st.success(f'{medals[rank]} **Best Match: {role}** — {conf*100:.1f}% confidence')
        else:
            st.info(f'{medals[rank]} {role} — {conf*100:.1f}%')
        st.progress(conf)

    st.markdown('---')

    # ── Skills ────────────────────────────────────────────────
    st.markdown('### 🛠 Skills Detected')
    skills_found = get_skills(raw_text)
    if skills_found:
        for domain, skill_list in skills_found.items():
            tags = '  '.join(f'`{s}`' for s in skill_list)
            st.markdown(f'**{domain}:** {tags}')
    else:
        st.warning('No predefined skills detected. Try a more detailed resume.')

    st.markdown('---')

    # ── Confidence breakdown ──────────────────────────────────
    with st.expander('📊 Full Confidence Breakdown (all categories)'):
        import pandas as pd
        full_df = pd.DataFrame({
            'Job Role': le.classes_,
            'Confidence (%)': (proba * 100).round(2)
        }).sort_values('Confidence (%)', ascending=False).reset_index(drop=True)
        st.dataframe(full_df, use_container_width=True)

    # ── Raw text preview ─────────────────────────────────────
    with st.expander('📄 View Extracted Resume Text'):
        st.text(raw_text[:3000] + ('\n... [truncated]' if len(raw_text) > 3000 else ''))

elif uploaded_file and not analyze_btn:
    st.info('📂 Resume uploaded. Click **Analyze Resume** to get results.')
else:
    st.markdown("""
    ### How it works
    1. 📤 Upload your resume as a PDF
    2. 🔍 Click **Analyze Resume**
    3. 🎯 Get your predicted job role + confidence
    4. 🛠 See which skills were detected
    """)
print('✅ app.py written')
