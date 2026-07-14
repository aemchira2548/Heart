import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 🎨 Custom CSS - Minimalist Design
# ==========================================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf3 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }
    
    /* Card ผลลัพธ์ */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .result-safe {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 6px solid #4CAF50;
    }
    .result-risk {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 6px solid #F44336;
    }
    .result-card h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.8rem;
    }
    .result-card p {
        margin: 0.3rem 0;
        font-size: 1.05rem;
    }
    
    /* Input Fields */
    .stNumberInput > label, .stSelectbox > label {
        font-weight: 500;
        color: #333;
    }
    
    /* ปุ่มหลัก */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
    }
    
    /* Divider */
    .stDivider {
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📦 โหลดโมเดล
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists('heart_disease_pipeline.joblib'):
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาวางไฟล์ `heart_disease_pipeline.joblib` ในโฟลเดอร์เดียวกัน")
        st.stop()
    pipeline = joblib.load('heart_disease_pipeline.joblib')
    return pipeline

pipeline = load_model()

# ==========================================
# 🎨 Header
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Predictor</h1>
    <p>ระบบประเมินความเสี่ยงโรคหัวใจด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 📋 Sidebar - Input Form
# ==========================================
with st.sidebar:
    st.markdown("### 📝 กรอกข้อมูลผู้ป่วย")
    st.markdown("---")
    
    # ข้อมูลพื้นฐาน
    st.markdown("#### 👤 ข้อมูลพื้นฐาน")
    age = st.number_input("อายุ (ปี)", min_value=20, max_value=100, value=55, step=1)
    sex = st.selectbox("เพศ", options=[1, 0], format_func=lambda x: "♂️ ชาย" if x==1 else "♀️ หญิง")
    
    st.markdown("---")
    st.markdown("#### 💓 อาการและสัญญาณชีพ")
    cp = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Typical Angina",
            2: "2 - Atypical Angina",
            3: "3 - Non-anginal Pain",
            4: "4 - Asymptomatic"
        }[x]
    )
    
    restbp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", min_value=0, max_value=250, value=130, step=1)
    chol = st.number_input("คอเลสเตอรอล (mg/dl)", min_value=0, max_value=600, value=220, step=1)
    fbs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl?", options=[0, 1], format_func=lambda x: "ใช่" if x==1 else "ไม่")
    restecg = st.selectbox(
        "ผล ECG ขณะพัก",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 - Normal", 1: "1 - ST-T Wave Abnormality", 2: "2 - LV Hypertrophy"}[x]
    )
    
    st.markdown("---")
    st.markdown("#### 🏃 การออกกำลังกาย")
    maxhr = st.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", min_value=50, max_value=220, value=150, step=1)
    exang = st.selectbox("เจ็บหน้าอกเมื่อออกกำลังกาย?", options=[0, 1], format_func=lambda x: "ใช่" if x==1 else "ไม่")
    oldpeak = st.number_input("ค่า ST Depression", min_value=-3.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    slope = st.selectbox(
        "ความชันของ ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}[x]
    )
    
    st.markdown("---")
    predict_button = st.button("🔍 ทำนายผล", use_container_width=True)

# ==========================================
# 🎯 Main Area - แสดงผล
# ==========================================
if predict_button:
    # สร้าง DataFrame จากข้อมูลผู้ใช้
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBP': restbp,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'RestingECG': restecg,
        'MaxHR': maxhr,
        'ExerciseAngina': exang,
        'Oldpeak': oldpeak,
        'ST_Slope': slope
    }])
    
    # ทำนาย
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]
    
    st.markdown("---")
    
    # แสดงผลลัพธ์แบบ Card
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-risk">
            <h2>⚠️ พบความเสี่ยงโรคหัวใจ</h2>
            <p><strong>ความมั่นใจของโมเดล:</strong> {probability[1]*100:.1f}%</p>
            <p style="margin-top: 1rem; font-size: 0.95rem; color: #555;">
                ผลลัพธ์นี้เป็นการประเมินเบื้องต้นจากโมเดล Machine Learning<br>
                กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการวินิจฉัยที่ถูกต้อง
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <h2>✅ ไม่พบความเสี่ยงโรคหัวใจ</h2>
            <p><strong>ความมั่นใจของโมเดล:</strong> {probability[0]*100:.1f}%</p>
            <p style="margin-top: 1rem; font-size: 0.95rem; color: #555;">
                ผลลัพธ์นี้เป็นการประเมินเบื้องต้นเท่านั้น<br>
                ควรตรวจสุขภาพประจำปีอย่างสม่ำเสมอ
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # แสดงข้อมูลที่ใช้ทำนาย
    with st.expander("📋 ดูข้อมูลที่ใช้ทำนาย", expanded=False):
        st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)

else:
    # หน้าต้อนรับ
    st.markdown("### 👈 เริ่มต้นใช้งาน")
    st.info("กรุณากรอกข้อมูลผู้ป่วยใน **Sidebar ด้านซ้ายมือ** แล้วกดปุ่ม **🔍 ทำนายผล**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧠 โมเดล", "Decision Tree")
    with col2:
        st.metric("📊 ข้อมูลฝึกสอน", "900+ samples")
    with col3:
        st.metric("🎯 ความแม่นยำ", "~82%")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;'>
        ⚕️ <strong>คำเตือน:</strong> ระบบนี้สร้างขึ้นเพื่อการศึกษาและวิจัยเท่านั้น 
        ไม่สามารถทดแทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้
    </div>
    """, unsafe_allow_html=True)