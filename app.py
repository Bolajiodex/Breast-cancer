"""
Breast Cancer Risk Assessment Tool

This Streamlit application provides an interactive interface for breast cancer risk assessment
based on biopsy measurements. It includes features for individual assessment, batch analysis,
and educational resources.

The application uses a trained Random Forest model to predict the likelihood of malignancy
based on cell measurements from breast tissue samples.
"""

# Standard library imports
import pickle

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Constants
PAGE_TITLE = "Breast Cancer Risk Assessment"
PAGE_ICON = "🎗️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Set page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Custom CSS with dark theme and high contrast
CUSTOM_CSS = """
    <style>
    /* Main container styling */
    .main {
        background-color: #1a1a1a !important;
        padding: 2rem;
    }

    /* Page title */
    .stMarkdown h1 {
        color: #ffffff !important;
        font-size: 2.5em !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        padding: 1rem !important;
        background: linear-gradient(135deg, #2d3748, #1a202c) !important;
        border-radius: 10px !important;
        border: 1px solid #4a5568 !important;
    }

    /* Section headers */
    .stMarkdown h3 {
        color: #60a5fa !important;
        font-size: 1.8em !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid #3b82f6 !important;
    }

    /* Radio button container */
    .stRadio > div {
        background-color: #2d3748 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #4a5568 !important;
    }

    /* Radio button question labels */
    .stRadio > label {
        color: #60a5fa !important;
        font-size: 1.3em !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        display: block !important;
        background-color: #1a202c !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        border-left: 5px solid #3b82f6 !important;
    }

    /* Radio button options */
    .stRadio div[role="radiogroup"] label {
        color: #e2e8f0 !important;
        font-size: 1.1em !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem 0 !important;
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        display: block !important;
        cursor: pointer !important;
    }

    /* Radio button hover state */
    .stRadio div[role="radiogroup"] label:hover {
        background-color: #374151 !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
        transform: translateX(5px) !important;
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.3) !important;
    }

    /* Selected radio button */
    .stRadio div[role="radiogroup"] [data-checked="true"] {
        background-color: #1e40af !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 2px solid #60a5fa !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.4) !important;
    }

    /* Submit button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1e40af) !important;
        color: white !important;
        padding: 1rem 2rem !important;
        font-size: 1.2em !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        margin-top: 2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Submit button hover state */
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Separator styling */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(to right, transparent, #3b82f6, transparent) !important;
        margin: 2rem 0 !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a202c, #2d3748) !important;
        padding: 2rem 1rem !important;
    }

    .css-1d391kg .css-1v0mbdj {
        color: #e2e8f0 !important;
        font-size: 1.2em !important;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    .css-1d391kg .css-1v0mbdj:hover {
        background-color: rgba(96, 165, 250, 0.2) !important;
        transform: translateX(5px) !important;
    }

    /* Make all text white in the main content area */
    .stMarkdown p, .stMarkdown li, .stText {
        color: #e2e8f0 !important;
    }

    /* Examination instruction boxes */
    .examination-box {
        background-color: #1a202c !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2) !important;
    }

    .examination-box h2 {
        color: #60a5fa !important;
        font-size: 1.5em !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #3b82f6 !important;
        padding-bottom: 0.5rem !important;
    }

    .examination-box p {
        color: #e2e8f0 !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
        margin-bottom: 1rem !important;
    }

    .examination-box ul {
        list-style-type: none !important;
        padding-left: 0 !important;
        margin-top: 1rem !important;
    }

    .examination-box li {
        color: #e2e8f0 !important;
        font-size: 1.1em !important;
        padding: 0.5rem 0 0.5rem 1.5rem !important;
        position: relative !important;
    }

    .examination-box li:before {
        content: "•" !important;
        color: #60a5fa !important;
        font-weight: bold !important;
        position: absolute !important;
        left: 0 !important;
    }

    /* Step numbers */
    .step-number {
        background-color: #3b82f6 !important;
        color: white !important;
        width: 2rem !important;
        height: 2rem !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-right: 0.5rem !important;
        font-weight: bold !important;
    }

    /* Risk factors columns container */
    .risk-factors-container {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 2rem !important;
        margin: 1.5rem 0 !important;
    }

    /* Risk factor boxes */
    .risk-box {
        padding: 1.5rem !important;
        border-radius: 12px !important;
        height: 100% !important;
    }

    /* Uncontrollable factors box */
    .uncontrollable-box {
        background-color: #1a1a2e !important;
        border: 2px solid #ff6b6b !important;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.1) !important;
    }

    /* Controllable factors box */
    .controllable-box {
        background-color: #1a2e1a !important;
        border: 2px solid #4ade80 !important;
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.1) !important;
    }

    /* Risk factor headers */
    .risk-box h3 {
        font-size: 1.5em !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
    }

    .uncontrollable-box h3 {
        color: #ff6b6b !important;
        border-bottom: 2px solid #ff6b6b !important;
    }

    .controllable-box h3 {
        color: #4ade80 !important;
        border-bottom: 2px solid #4ade80 !important;
    }

    /* Risk factor lists */
    .risk-box ul {
        list-style-type: none !important;
        padding-left: 0 !important;
        margin: 0 !important;
    }

    .risk-box li {
        color: #e2e8f0 !important;
        font-size: 1.1em !important;
        padding: 0.5rem 0 0.5rem 1.5rem !important;
        position: relative !important;
        line-height: 1.5 !important;
    }

    .uncontrollable-box li:before {
        content: "•" !important;
        color: #ff6b6b !important;
        font-weight: bold !important;
        position: absolute !important;
        left: 0 !important;
    }

    .controllable-box li:before {
        content: "•" !important;
        color: #4ade80 !important;
        font-weight: bold !important;
        position: absolute !important;
        left: 0 !important;
    }
    </style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def load_data():
    """
    Load the trained model, feature names, and feature ranges from pickle files.
    
    Returns:
        tuple: Contains the following elements:
            - model (RandomForestClassifier): Trained model for prediction
            - feature_names (list): Names of features used by the model
            - feature_ranges (dict): Min and max values for each feature
            
    Raises:
        FileNotFoundError: If any required file is missing
        pickle.UnpicklingError: If there's an error loading pickle files
        ValueError: If loaded data is invalid or incompatible
    """
    try:
        with open('model/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        with open('model/feature_ranges.pkl', 'rb') as f:
            feature_ranges = pickle.load(f)
            
        # Validate loaded data
        if not isinstance(model, RandomForestClassifier):
            raise ValueError("Loaded model is not a RandomForestClassifier")
            
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Invalid or empty feature names")
            
        if not isinstance(feature_ranges, dict) or not feature_ranges:
            raise ValueError("Invalid or empty feature ranges")
            
        return model, feature_names, feature_ranges
        
    except FileNotFoundError as e:
        st.error(f"Required model file not found: {str(e)}")
        st.info("Please ensure all model files are present in the 'model' directory")
        raise
        
    except pickle.UnpicklingError as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Model files may be corrupted or incompatible")
        raise
        
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        raise

def predict_cancer_risk(model, features, feature_names):
    """
    Predict cancer risk using the loaded model.
    
    Args:
        model (RandomForestClassifier): Trained model for prediction
        features (list): List of feature values for prediction
        feature_names (list): Names of features in correct order
        
    Returns:
        tuple: Contains prediction (0 or 1) and probability scores
        
    Raises:
        ValueError: If features are invalid or incompatible
    """
    try:
        # Validate input
        if len(features) != len(feature_names):
            raise ValueError("Number of features doesn't match model requirements")
            
        # Reshape features for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise

def validate_input_range(value, feature_name, feature_ranges):
    """
    Validate if an input value is within the acceptable range.
    
    Args:
        value (float): Input value to validate
        feature_name (str): Name of the feature being validated
        feature_ranges (dict): Dictionary containing valid ranges for features
        
    Returns:
        bool: True if value is within range, False otherwise
    """
    if feature_name not in feature_ranges:
        return True
        
    min_val, max_val = feature_ranges[feature_name]
    return min_val <= value <= max_val

def create_sidebar():
    """
    Create and configure the sidebar with navigation options.
    
    Returns:
        str: Selected page from the sidebar
    """
    with st.sidebar:
        st.image("assets/logo.png", use_column_width=True)
        st.title("Navigation")
        return st.radio(
            "Select a Page",
            ["Home", "Risk Assessment", "Self-Examination Guide", "Risk Factors & Prevention"],
            key="navigation"
        )

def display_home_page():
    """Display the home page with application overview and instructions."""
    st.title("Breast Health Assessment Tool")
    
    st.markdown("""
    ### Welcome to Your Breast Health Companion
    
    This user-friendly tool helps you understand and monitor your breast health. We provide:
    
    1. **Risk Assessment**: Answer simple questions about your health to understand your risk factors
    2. **Self-Examination Guide**: Learn how to perform monthly self-examinations with step-by-step guidance
    3. **Risk Factors & Prevention**: Understand what factors affect breast health and learn prevention strategies
    
    #### Why Use This Tool?
    - Easy to understand information about breast health
    - Step-by-step guidance for self-examination
    - Practical tips for maintaining breast health
    - Educational resources and support information
    
    #### Important Note
    This tool is for educational purposes only and should not replace professional medical advice.
    Always consult with healthcare providers for medical decisions.
    """)

def display_risk_assessment():
    """Display a simplified risk assessment form with user-friendly inputs."""
    st.title("Breast Health Risk Assessment")
    
    st.markdown("""
    ### Simple Risk Assessment Questionnaire
    
    Please answer these questions to help understand your breast health risk factors.
    Your privacy is important - this information is not stored or shared.
    """)
    
    try:
        # Create user-friendly form
        with st.form("risk_assessment_form"):
            # Personal Information
            st.subheader("Basic Information")
            age = st.slider("Age", 18, 100, 25)
            
            # Family History
            st.subheader("Family History")
            family_history = st.selectbox(
                "Has anyone in your immediate family had breast cancer?",
                ["No", "Yes - Mother", "Yes - Sister", "Yes - Multiple family members"]
            )
            
            # Lifestyle Factors
            st.subheader("Lifestyle Factors")
            exercise = st.select_slider(
                "How often do you exercise?",
                options=["Rarely", "1-2 times/month", "1-2 times/week", "3+ times/week"]
            )
            
            diet = st.select_slider(
                "How would you rate your diet?",
                options=["Poor", "Fair", "Good", "Excellent"]
            )
            
            # Medical History
            st.subheader("Medical History")
            
            # Previous biopsies with labels next to each button
            previous_biopsies = st.radio(
                "Have you had previous breast biopsies?",
                ["No", "Yes - Benign", "Yes - Abnormal", "Unsure"],
                key="biopsies",
                horizontal=False,
                index=0
            )
            
            st.markdown("---")
            
            # Regular checkups with labels next to each button
            regular_checkups = st.radio(
                "Do you get regular medical check-ups?",
                ["Yes", "No", "Sometimes"],
                key="checkups",
                horizontal=False,
                index=0
            )
            
            # Submit button
            submitted = st.form_submit_button("Assess My Risk")
            
            if submitted:
                # Calculate risk level based on inputs
                risk_factors = 0
                
                # Age risk
                if age > 50:
                    risk_factors += 1
                
                # Family history risk
                if "Yes" in family_history:
                    risk_factors += 2 if "Multiple" in family_history else 1
                
                # Lifestyle risks
                if exercise == "Rarely":
                    risk_factors += 1
                if diet == "Poor":
                    risk_factors += 1
                
                # Medical history risks
                if "Yes" in previous_biopsies:
                    risk_factors += 1
                if regular_checkups == "No":
                    risk_factors += 1
                
                # Display results
                st.markdown("### Your Risk Assessment Results")
                
                # Risk level determination
                risk_level = "Low" if risk_factors <= 2 else "Moderate" if risk_factors <= 4 else "High"
                risk_color = "#28a745" if risk_level == "Low" else "#ffc107" if risk_level == "Moderate" else "#dc3545"
                
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}25; border: 2px solid {risk_color}'>
                    <h4 style='color: {risk_color}'>Risk Level: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Recommendations")
                recommendations = []
                
                if age > 40:
                    recommendations.append("• Schedule regular mammograms as recommended by your healthcare provider")
                if "Yes" in family_history:
                    recommendations.append("• Consider genetic counseling to understand your family history better")
                if exercise == "Rarely":
                    recommendations.append("• Try to incorporate more physical activity into your routine")
                if diet == "Poor":
                    recommendations.append("• Consider consulting a nutritionist for dietary guidance")
                if regular_checkups == "No":
                    recommendations.append("• Schedule regular check-ups with your healthcare provider")
                
                st.markdown("\n".join(recommendations))
                
                st.markdown("""
                ### Next Steps
                1. Share these results with your healthcare provider
                2. Schedule recommended screenings
                3. Review our Self-Examination Guide
                4. Learn more about prevention in our Risk Factors & Prevention section
                """)
    
    except Exception as e:
        st.error("An error occurred during the assessment. Please try again.")

def display_self_examination_guide():
    """Display a comprehensive guide for breast self-examination."""
    st.title("Breast Self-Examination Guide")
    
    st.markdown("""
    ### Monthly Self-Examination Guide
    
    Regular self-examination helps you become familiar with your breasts' normal look and feel,
    making it easier to notice any changes that might need medical attention.
    """)
    
    # Visual Examination Section
    st.markdown("""
    <div class="examination-box">
        <h2><span class="step-number">1</span> Visual Examination</h2>
        <p>Stand in front of a mirror with good lighting and look for:</p>
        <ul>
            <li>Changes in size, shape, or color</li>
            <li>Dimpling or puckering of the skin</li>
            <li>Inverted nipples or changes in nipple position</li>
            <li>Redness, rash, or swelling</li>
        </ul>
        <p>Check these in four positions:</p>
        <ul>
            <li>Arms at sides</li>
            <li>Arms raised above head</li>
            <li>Hands on hips</li>
            <li>Leaning forward</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Physical Examination Section
    st.markdown("""
    <div class="examination-box">
        <h2><span class="step-number">2</span> Physical Examination</h2>
        <p>Lying down:</p>
        <ul>
            <li>Use your right hand to examine left breast, and vice versa</li>
            <li>Use the pads of your three middle fingers</li>
            <li>Use three different pressure levels: light, medium, and firm</li>
            <li>Move in a pattern (circular, up-and-down, or wedge)</li>
            <li>Cover the entire breast area, including:</li>
            <li style="padding-left: 2rem;">• Collarbone to bra line</li>
            <li style="padding-left: 2rem;">• Armpit to cleavage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # What to Look For Section
    st.markdown("""
    <div class="examination-box">
        <h2>What to Look For</h2>
        <p>Contact your healthcare provider if you notice:</p>
        <ul>
            <li>New lumps or thickening</li>
            <li>Changes in size or shape</li>
            <li>Skin changes (dimpling, redness, or scaling)</li>
            <li>Nipple changes or discharge</li>
            <li>New pain in one spot that doesn't go away</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_risk_factors_prevention():
    """Display information about risk factors and prevention strategies."""
    st.title("Understanding Risk Factors & Prevention")
    
    st.markdown("""
    ### Risk Factors
    Understanding risk factors helps you make informed decisions about your breast health.
    Remember that having risk factors doesn't mean you'll develop breast cancer.
    """)
    
    # Risk Factors Section with improved styling
    st.markdown("""
    <div class="risk-factors-container">
        <div class="risk-box uncontrollable-box">
            <h3>Factors You Can't Control</h3>
            <ul>
                <li>Being female</li>
                <li>Getting older</li>
                <li>Family history</li>
                <li>Genetic mutations</li>
                <li>Dense breast tissue</li>
                <li>Previous radiation exposure</li>
                <li>Early menstruation/Late menopause</li>
            </ul>
        </div>
        <div class="risk-box controllable-box">
            <h3>Factors You Can Control</h3>
            <ul>
                <li>Maintaining a healthy weight</li>
                <li>Regular physical activity</li>
                <li>Limiting alcohol consumption</li>
                <li>Not smoking</li>
                <li>Healthy diet</li>
                <li>Breastfeeding when possible</li>
                <li>Limiting hormone therapy</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prevention Strategies Section using Streamlit native components
    st.header("Prevention Strategies")

    # 1. Healthy Lifestyle Choices
    st.subheader("1. Healthy Lifestyle Choices")
    with st.container():
        st.markdown("🏃‍♀️ **Exercise Regularly**")
        st.markdown("""
        - Aim for 150 minutes of moderate activity per week
        - Include both cardio and strength training
        - Find activities you enjoy
        """)
        
        st.markdown("🥗 **Maintain a Healthy Diet**")
        st.markdown("""
        - Eat plenty of local fruits and vegetables
        - Choose whole grains
        - Limit processed foods
        - Include traditional Nigerian vegetables rich in nutrients
        """)
        
        st.markdown("🧘‍♀️ **Manage Stress**")
        st.markdown("""
        - Practice relaxation techniques
        - Get adequate sleep
        - Maintain work-life balance
        """)

    # 2. Regular Screening
    st.subheader("2. Regular Screening")
    with st.container():
        st.markdown("📅 **Recommended Screening Schedule**")
        st.markdown("""
        - Monthly self-examinations
        - Clinical breast exams every 1-3 years (20s and 30s)
        - Annual clinical breast exams (40 and older)
        - Mammograms as recommended by your healthcare provider
        """)

    # Additional Resources Section
    st.header("Additional Resources")

    # Nigerian Healthcare Resources
    st.subheader("Nigerian Healthcare Resources")
    with st.container():
        st.markdown("**Local Support Organizations**")
        st.markdown("""
        - Nigerian Cancer Society: [www.nigeriancancersociety.org](https://nigeriancancersociety.org)
        - Breast Cancer Association of Nigeria (BRECAN): [www.brecan.org](https://brecan.org)
        - Run for a Cure Africa: [www.rfca.org.ng](https://rfca.org.ng)
        """)
        
        st.markdown("**Major Cancer Centers in Nigeria**")
        st.markdown("""
        - National Hospital Abuja Cancer Center
        - Lagos University Teaching Hospital (LUTH)
        - University College Hospital (UCH), Ibadan
        """)
        
        st.markdown("**24/7 Cancer Support Helplines**")
        st.markdown("""
        - Nigerian Cancer Society Helpline: 0800-CANCER-HELP
        - BRECAN Support Line: +234-xxx-xxxx-xxxx
        """)

    # International Resources
    st.subheader("International Resources")
    with st.container():
        st.markdown("**Global Organizations**")
        st.markdown("""
        - World Health Organization (WHO) Cancer Programme: [www.who.int/cancer](https://www.who.int/cancer)
        - Union for International Cancer Control: [www.uicc.org](https://www.uicc.org)
        - American Cancer Society: [www.cancer.org](https://www.cancer.org)
        """)
        
        st.markdown("**Research & Information**")
        st.markdown("""
        - International Agency for Research on Cancer: [www.iarc.who.int](https://www.iarc.who.int)
        - Global Cancer Observatory: [gco.iarc.fr](https://gco.iarc.fr)
        """)

    # Emergency Contacts
    st.subheader("Emergency Contacts")
    with st.container():
        st.markdown("**Nigeria**")
        st.markdown("""
        - Emergency Medical Services: 112
        - Nigerian Cancer Society Emergency Line: 0800-CANCER-HELP
        """)
        
        st.markdown("**Your Healthcare Provider**")
        st.markdown("""
        - Keep your doctor's contact information readily available
        - Know the location of the nearest emergency medical center
        """)

# Main application logic
try:
    # Sidebar navigation
    page = create_sidebar()

    # Page routing
    if page == "Home":
        display_home_page()
    elif page == "Risk Assessment":
        display_risk_assessment()
    elif page == "Self-Examination Guide":
        display_self_examination_guide()
    elif page == "Risk Factors & Prevention":
        display_risk_factors_prevention()
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page and try again.") 