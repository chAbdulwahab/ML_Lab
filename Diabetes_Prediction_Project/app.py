import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import time
import os

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="AI Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (Vanilla CSS & Glassmorphism)
# ==========================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">

    <style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp, .main {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%) !important;
        color: white !important;
    }
    
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    
    /* Layout Elements */
    .title-text {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -20px;
        margin-bottom: 5px;
        filter: drop-shadow(0 2px 8px rgba(56, 189, 248, 0.2));
    }
    
    .subtitle {
        color: #cbd5e1;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 25px;
    }
    
    /* Card design */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .metric-card h2 {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }

    .metric-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    
    /* Custom Styling for Predict Button */
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        height: 3em !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
    }
    
    /* Sidebar premium styling */
    .sidebar-title {
        color: #38bdf8;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 20px;
    }
    
    .sidebar-desc {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #38bdf8;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# CACHED DATA LOADING & MULTI-MODEL TRAINING
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "diabetes_prediction_dataset.csv")
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def train_and_persist_all_models(df):
    df_encoded = df.copy()
    
    gender_encoder = LabelEncoder()
    smoking_encoder = LabelEncoder()
    
    df_encoded["gender"] = gender_encoder.fit_transform(df_encoded["gender"])
    df_encoded["smoking_history"] = smoking_encoder.fit_transform(df_encoded["smoking_history"])
    
    X = df_encoded.drop("diabetes", axis=1)
    y = df_encoded["diabetes"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    # Standard Scaler (essential for KNN and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest Classifier (Multi-threaded)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # 2. Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # 3. K-Nearest Neighbors (KNN)
    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    
    # Programmatically persist the primary models to disk (ensures exact folder tree compliance)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(rf_model, os.path.join(current_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(current_dir, "scaler.pkl"))
    
    # Save the secondary models for diagnostic completeness
    joblib.dump(lr_model, os.path.join(current_dir, "logistic_model.pkl"))
    joblib.dump(knn_model, os.path.join(current_dir, "knn_model.pkl"))
    
    models = {
        "Random Forest": (rf_model, rf_acc),
        "Logistic Regression": (lr_model, lr_acc),
        "KNN": (knn_model, knn_acc)
    }
    
    return models, gender_encoder, smoking_encoder, scaler, X.columns

# Load and train all 3 models
df = load_data()
models, gender_encoder, smoking_encoder, scaler, X_columns = train_and_persist_all_models(df)

# ==========================================
# TITLE & ALGORITHM SELECTION
# ==========================================
st.markdown("<div class='title-text'>🩺 AI Diabetes Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Professional Multi-Model Machine Learning Healthcare System</div>", unsafe_allow_html=True)

# Selectbox on the main page!
col_sel, col_empty = st.columns([2, 1])
with col_sel:
    selected_model_name = st.selectbox(
        "🧠 Choose Machine Learning Algorithm",
        ["Random Forest", "Logistic Regression", "KNN"],
        help="Select which machine learning model to use for the risk prediction analysis."
    )

selected_model, selected_accuracy = models[selected_model_name]

st.markdown("---")

# ==========================================
# SIDEBAR (Model Info & Specs)
# ==========================================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📊 Classifier Metrics</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.write(f"🌟 **Active Model**: {selected_model_name}")
    st.write("🎯 **Test Accuracy**: {:.2f}%".format(selected_accuracy * 100))
    if selected_model_name == "Random Forest":
        st.write("🌳 **Trees**: 200 estimators")
        st.write("📏 **Max Depth**: 10 splits")
    elif selected_model_name == "Logistic Regression":
        st.write("📈 **Optimization**: L-BFGS Solver")
        st.write("🔄 **Max Iterations**: 1000")
    else:
        st.write("👥 **Neighbors (K)**: 5 neighbors")
        st.write("📐 **Metric**: Minkowski Distance")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("⚠️ **Medical Notice**")
    st.caption(
        "Diagnostic inferences provided are for clinical screening and research. "
        "They are not a replacement for formal laboratory assays or physician advice."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 50px;'>v1.5.0 | AI Healthcare</p>", unsafe_allow_html=True)

# ==========================================
# CLINICAL TERM TRANSLATION MAPS
# ==========================================
SMOKING_MAP = {
    "Never Smoked": "never",
    "No Information Available": "No Info",
    "Active Smoker": "current",
    "Former Smoker": "former",
    "Ever Smoked (Occasionally/Past)": "ever",
    "Not Currently Smoking (Passive/Light)": "not current"
}

# ==========================================
# MAIN INTERFACE (Split Layout)
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='section-header'>🧾 Patient Profile Details</div>", unsafe_allow_html=True)
    
    gender = st.selectbox(
        "Patient Gender",
        ["Female", "Male", "Other"]
    )
    
    age = st.slider(
        "Patient Age (Years)",
        1,
        100,
        25
    )
    
    hypertension = st.selectbox(
        "High Blood Pressure History (Hypertension)",
        [0, 1],
        format_func=lambda x: "Yes, diagnosed with High Blood Pressure" if x == 1 else "No history of High Blood Pressure",
        help="Check 'Yes' if the patient has ever been diagnosed with hypertension or takes blood pressure medications."
    )
    
    heart_disease = st.selectbox(
        "Heart Disease History",
        [0, 1],
        format_func=lambda x: "Yes, diagnosed with Heart Disease" if x == 1 else "No history of Heart Disease",
        help="Check 'Yes' if the patient has diagnosed cardiovascular conditions or history of cardiac events."
    )
    
    smoking_friendly = st.selectbox(
        "Smoking History Status",
        list(SMOKING_MAP.keys()),
        help="Select the category that best describes the patient's tobacco exposure history."
    )
    # Translate back to raw clinical categories for model preprocessing
    smoking_history = SMOKING_MAP[smoking_friendly]
    
    bmi = st.slider(
        "Body Mass Index (BMI)",
        10.0,
        60.0,
        25.0,
        0.1,
        help="BMI measures body mass based on height and weight. Normal Range: 18.5 - 24.9. Overweight: 25.0 - 29.9. Obese: 30.0 or higher."
    )
    
    HbA1c_level = st.slider(
        "HbA1c Level (3-Month Average Blood Sugar %)",
        3.0,
        10.0,
        5.5,
        0.1,
        help="HbA1c tracks average blood glucose over the past 3 months. Normal Range: Below 5.7%. Prediabetes: 5.7% - 6.4%. Diabetes: 6.5% or higher."
    )
    
    blood_glucose_level = st.slider(
        "Fasting Blood Glucose Level (mg/dL)",
        50,
        300,
        120,
        help="Blood glucose level after fasting overnight. Normal Range: 70 - 100 mg/dL. Prediabetes: 100 - 125 mg/dL. Diabetes: 126 mg/dL or higher."
    )
    
    st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
    predict_button = st.button(f"🚀 Predict Risk with {selected_model_name}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-header'>📊 System Analytics</div>", unsafe_allow_html=True)
    
    # 1. Age Distribution (Overlaid & Normalized)
    df_plot = df.copy()
    df_plot["diabetes"] = df_plot["diabetes"].map({
        0: "0 (No / Non-Diabetic)",
        1: "1 (Yes / Diabetic)"
    })
    
    fig = px.histogram(
        df_plot,
        x="age",
        color="diabetes",
        nbins=30,
        barmode="overlay",
        histnorm="percent",
        title="Age Distribution Trends",
        color_discrete_map={"0 (No / Non-Diabetic)": "#06b6d4", "1 (Yes / Diabetic)": "#ef4444"},
        labels={"age": "Age", "diabetes": "Status"},
        opacity=0.75
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_family="Outfit",
        title_font_color="#38bdf8",
        title_font_size=18,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        yaxis_title="Percentage (%)"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Blood Glucose Level Box Plot
    fig2 = px.box(
        df_plot,
        x="diabetes",
        y="blood_glucose_level",
        color="diabetes",
        title="Glucose Level by Diagnosis",
        color_discrete_map={"0 (No / Non-Diabetic)": "#06b6d4", "1 (Yes / Diabetic)": "#ef4444"},
        labels={"diabetes": "Diagnosis", "blood_glucose_level": "Glucose Level (mg/dL)"}
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_family="Outfit",
        title_font_color="#38bdf8",
        title_font_size=18,
        showlegend=False
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# PREDICTION ENGINE & SUGGESTIONS
# ==========================================
if predict_button:
    st.markdown("---")
    st.markdown(f"<div class='section-header'>🧠 AI Prediction Result ({selected_model_name})</div>", unsafe_allow_html=True)
    
    with st.spinner(f"AI {selected_model_name} Model Analyzing Patient Profile..."):
        time.sleep(1.5)
        
    gender_encoded = gender_encoder.transform([gender])[0]
    smoking_encoded = smoking_encoder.transform([smoking_history])[0]
    
    # Formulate inputs as a DataFrame to completely eliminate scikit-learn warnings
    input_df = pd.DataFrame([[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        smoking_encoded,
        bmi,
        HbA1c_level,
        blood_glucose_level
    ]], columns=X_columns)
    
    # Scale inputs using the fitted StandardScaler
    input_scaled = scaler.transform(input_df)
    
    # Predictions
    prediction = selected_model.predict(input_scaled)[0]
    probability = selected_model.predict_proba(input_scaled)[0][1]
    
    # Diagnostic Metrics Row
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h2>Classifier Accuracy</h2>
                <h1>{selected_accuracy*100:.2f}%</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with result_col2:
        if prediction == 1:
            result_text = "1 (Yes / Diabetic)"
            text_color = "color: #ef4444;"
        else:
            result_text = "0 (No / Non-Diabetic)"
            text_color = "color: #34d399;"
            
        st.markdown(
            f"""
            <div class='metric-card'>
                <h2>Diagnostic Result</h2>
                <h1 style='{text_color}; font-size: 1.6rem;'>{result_text}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with result_col3:
        prob_color = "color: #ef4444;" if probability > 0.7 else ("color: #f59e0b;" if probability > 0.3 else "color: #34d399;")
        st.markdown(
            f"""
            <div class='metric-card'>
                <h2>Risk Probability</h2>
                <h1 style='{prob_color}'>{probability*100:.2f}%</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # Gauge Chart / Meter
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "Diabetes Risk Meter", 'font': {'size': 20, 'family': 'Outfit', 'color': '#38bdf8'}},
        number={'suffix': "%", 'font': {'color': 'white', 'family': 'Outfit'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#cbd5e1'},
            'bar': {'color': "#3b82f6"},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 30], 'color': "rgba(52, 211, 153, 0.15)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}
            ]
        }
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        height=320
    )
    st.plotly_chart(gauge, use_container_width=True)
    
    # Suggestions Container
    st.markdown("<div class='section-header'>💡 AI Health Suggestions</div>", unsafe_allow_html=True)
    
    if probability < 0.3:
        st.success("""
        ### ✅ Low diabetes risk detected.
        
        **Recommendations to Maintain Balance:**
        - **Dietary Control**: Continue a diet rich in complex fibers, high-quality proteins, and healthy monounsaturated fats.
        - **Active Lifestyle**: Target a minimum of 150 minutes of moderate aerobic exercise (e.g. brisk walking) weekly.
        - **Regular Testing**: Keep screening glucose levels during annual routine physicals.
        """)
    elif probability < 0.7:
        st.warning("""
        ### ⚠ Moderate diabetes risk detected.
        
        **Recommendations for Preventative Intervention:**
        - **Refine Intake**: Decrease intake of refined sugars, simple carbohydrates, and carbonated beverages.
        - **Weight Management**: Implement safe cardiovascular and strength training routines to optimize weight and insulin sensitivity.
        - **Self-Monitoring**: Regularly monitor fasting glucose levels and track metabolic profiles.
        """)
    else:
        st.error("""
        ### 🚨 High diabetes risk detected.
        
        **Immediate Action Recommendations:**
        - **Physician Consult**: Schedule an immediate consultation with a primary care physician or clinical endocrinologist.
        - **Biomarker Panels**: Request detailed testing, including fasting plasma glucose, oral glucose tolerance (OGTT), or full HbA1c panels.
        - **Urgent Lifestyle Adjustments**: Restrict calorie intakes, reduce stress factors, get 7-8 hours of sleep, and implement clinical monitoring.
        """)

# ==========================================
# FEATURE IMPORTANCE SECTION (Dynamic)
# ==========================================
st.markdown("---")

if selected_model_name == "Random Forest":
    st.markdown("<div class='section-header'>📈 Feature Importance (Random Forest Ensemble Weights)</div>", unsafe_allow_html=True)
    
    importance = selected_model.feature_importances_
    feature_df = pd.DataFrame({"Feature": X_columns, "Importance": importance})
    feature_df = feature_df.sort_values(by="Importance", ascending=True)
    
    # Map names to clinical, clean titles
    feature_df["Clean Name"] = feature_df["Feature"].map({
        "gender": "Gender",
        "age": "Age",
        "hypertension": "Hypertension History",
        "heart_disease": "Heart Disease History",
        "smoking_history": "Smoking History",
        "bmi": "Body Mass Index (BMI)",
        "HbA1c_level": "HbA1c Level",
        "blood_glucose_level": "Blood Glucose Level"
    })
    
    feature_fig = px.bar(
        feature_df,
        x="Importance",
        y="Clean Name",
        orientation="h",
        title="Random Forest Feature Decision Importances",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    feature_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_family="Outfit",
        title_font_color="#38bdf8",
        title_font_size=18,
        coloraxis_showscale=False
    )
    feature_fig.update_xaxes(showgrid=False)
    feature_fig.update_yaxes(showgrid=False)
    st.plotly_chart(feature_fig, use_container_width=True)

elif selected_model_name == "Logistic Regression":
    st.markdown("<div class='section-header'>📈 Feature Coefficients (Logistic Regression Decision Bounds)</div>", unsafe_allow_html=True)
    
    # Use absolute value of coefficients as relative decision weight
    importance = np.abs(selected_model.coef_[0])
    feature_df = pd.DataFrame({"Feature": X_columns, "Importance": importance})
    feature_df = feature_df.sort_values(by="Importance", ascending=True)
    
    # Map names to clinical, clean titles
    feature_df["Clean Name"] = feature_df["Feature"].map({
        "gender": "Gender",
        "age": "Age",
        "hypertension": "Hypertension History",
        "heart_disease": "Heart Disease History",
        "smoking_history": "Smoking History",
        "bmi": "Body Mass Index (BMI)",
        "HbA1c_level": "HbA1c Level",
        "blood_glucose_level": "Blood Glucose Level"
    })
    
    feature_fig = px.bar(
        feature_df,
        x="Importance",
        y="Clean Name",
        orientation="h",
        title="Logistic Regression Absolute Coefficients (Feature Impact)",
        color="Importance",
        color_continuous_scale="Plasma"
    )
    feature_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_family="Outfit",
        title_font_color="#38bdf8",
        title_font_size=18,
        coloraxis_showscale=False
    )
    feature_fig.update_xaxes(showgrid=False)
    feature_fig.update_yaxes(showgrid=False)
    st.plotly_chart(feature_fig, use_container_width=True)

else:
    st.markdown("<div class='section-header'>📈 Feature Weights Analysis</div>", unsafe_allow_html=True)
    st.info(
        "ℹ️ **K-Nearest Neighbors (KNN)** is an instance-based lazy learner and operates based on distance metrics between points. "
        "Consequently, it does not calculate parametric weights or coefficients. "
        "Switch your settings to **Random Forest** or **Logistic Regression** in the sidebar to view detailed dynamic feature weight analysis!"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <center>
    <p style='color:#64748b; font-size: 0.85rem;'>
    AI Diabetes Prediction Platform | Powered by Streamlit, Scikit-Learn & Plotly<br>
    © 2026 Machine Learning Healthcare Solutions
    </p>
    </center>
    """,
    unsafe_allow_html=True
)
