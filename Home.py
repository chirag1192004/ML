import streamlit as st
import base64
import os

# --- INITIAL CONFIGURATION ---
# Check if theme state exists, if not, set default to 'dark'
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Set the page configuration
st.set_page_config(
    page_title="Professional ML/DL Portfolio Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME STYLING AND SWITCH FUNCTIONALITY ---

# Define color schemes
DARK_COLORS = {
    "background": "#121212",
    "secondary_background": "#1E1E1E",
    "primary_color": "#00BCD4", # Teal/Cyan Accent
    "text_color": "#F0F0F0",
    "card_color": "#282828",
    "tag_color": "#4CAF50", # Green for Tech
    "demo_button_bg": "#4CAF50", # Green for Demo Button
    "demo_button_hover": "#388E3C" # Darker Green for hover
}

LIGHT_COLORS = {
    "background": "#FFFFFF",
    "secondary_background": "#F0F0F0",
    "primary_color": "#303F9F", # Indigo Accent
    "text_color": "#212121",
    "card_color": "#EAEAEA",
    "tag_color": "#388E3C", # Darker Green for visibility
    "demo_button_bg": "#388E3C", 
    "demo_button_hover": "#4CAF50"
}

COLORS = DARK_COLORS if st.session_state.theme == 'dark' else LIGHT_COLORS

# Function to toggle theme
def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Custom CSS function
def set_custom_css(colors):
    css = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* General Theme Settings */
        .stApp {{
            background-color: {colors['background']};
            color: {colors['text_color']};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {colors['secondary_background']};
        }}

        /* Main title styling */
        .big-font {{
            font-size: 52px !important;
            font-weight: 800;
            color: {colors['primary_color']};
        }}
        /* Subheader styling */
        .subheader-font {{
            font-size: 20px !important;
            font-weight: 400;
            color: {colors['text_color']};
        }}
        
        /* Project Card Styling */
        .project-card {{
            background-color: {colors['card_color']};
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            border-left: 6px solid {colors['primary_color']};
            transition: transform 0.3s;
        }}

        .project-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }}

        .project-card h4 {{
            color: {colors['text_color']};
            margin-bottom: 10px;
            font-size: 1.5em;
        }}
        
        /* Tech Stack Tags */
        .tech-tag {{
            display: inline-block;
            background-color: {colors['tag_color']};
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            margin-right: 5px;
            margin-top: 5px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        /* --- NEW DEMO BUTTON STYLE --- */
        .demo-button {{
            display: block;
            margin-top: 20px;
            padding: 10px 15px;
            background-color: {colors['demo_button_bg']};
            color: white !important; /* Ensure text color is white */
            text-align: center;
            font-weight: 700;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            width: 100%; 
        }}
        .demo-button:hover {{
            background-color: {colors['demo_button_hover']};
            transform: translateY(-1px);
        }}

        /* Apply button style to Streamlit markdown anchor tags */
        .stMarkdown a:last-child {{
            /* Selects the last link in the project card, which is our intended button */
            display: block;
            margin-top: 20px;
            padding: 10px 15px;
            background-color: {colors['demo_button_bg']};
            color: white !important; 
            text-align: center;
            font-weight: 700;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            width: 100%; 
        }}
        .stMarkdown a:last-child:hover {{
             background-color: {colors['demo_button_hover']} !important;
             transform: translateY(-1px);
        }}

        /* General cleanup */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply CSS based on current theme state
set_custom_css(COLORS)


# --- THEME SWITCH UI ---
theme_icon = "🌙 Dark Mode" if st.session_state.theme == 'dark' else "☀️ Light Mode"
st.sidebar.button(theme_icon, on_click=toggle_theme)


# --- PROJECT DATA (Mapped to your folder structure) ---
PROJECTS = {
    "1_Sentiment_Analysis": {
        "title": "Amazon Review Sentiment Analyzer 💬",
        "logo": "fas fa-comments",
        "description": "A robust ML pipeline for classifying Amazon product reviews (from the Alexa dataset) as positive or negative. Ready for custom text input.",
        "tech_stack": ["Scikit-learn", "NLP", "Text Classification", "Pickle"]
    },
    "2_Employee_Turnover": {
        "title": "Employee Turnover Predictor 🧑‍💼",
        "logo": "fas fa-users-slash",
        "description": "A supervised learning model (optimized for HR Analytics) to forecast which employees are at a high risk of leaving the company, based on various employment metrics.",
        "tech_stack": ["XGBoost/LogReg", "Classification", "HR Analytics", "Feature Engineering"]
    },
    "3_Loan_Approval": {
        "title": "Loan Approval Prediction System 🏦",
        "logo": "fas fa-hand-holding-dollar",
        "description": "Predicts the likelihood of a loan application being approved. The model considers factors like credit history, income, education, and marital status. A key tool for financial assessment.",
        "tech_stack": ["Machine Learning", "Classification", "Financial Modeling", "Data Cleaning"]
    },
    "4_AI_Chatbot": {
        "title": "Conversational AI Chatbot 🤖",
        "logo": "fas fa-robot",
        "description": "A deep learning-based chatbot using PyTorch for natural language interaction and intent recognition, trained on a custom intents dataset.",
        "tech_stack": ["Deep Learning", "PyTorch", "NLP", "NLTK", "Intent Recognition"]
    },
    "5_Earthquake_Viz": {
        "title": "Global Earthquake Visualization 🌍",
        "logo": "fas fa-globe-americas",
        "description": "An interactive data dashboard using Folium to map and visualize global earthquake data (1995-2023). Allows users to filter earthquakes by magnitude.",
        "tech_stack": ["Data Visualization", "Geospatial", "Folium", "Pandas", "Mapping"]
    },
}

# --- HEADER SECTION ---
st.markdown('<p class="big-font">Machine Learning & Deep Learning Portfolio Hub 🧠</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-font">A centralized, interactive showcase of professional, high-impact AI/ML projects.</p>', unsafe_allow_html=True)
st.markdown("---") 

st.markdown("### Welcome to the Portfolio Hub")
st.write(
    "Use the **navigation links in the sidebar** to seamlessly explore each project. "
    "Every page is a fully interactive Streamlit application showcasing the model's functionality and insights."
)
st.markdown("---")

# --- PROJECT DISPLAY (HOME PAGE) ---
st.markdown("## Featured Projects Overview")

# Use columns for better visual organization on wide screens
cols = st.columns(2)
col_index = 0

for key, data in PROJECTS.items():
    # Construct the correct Streamlit path
    project_path = f"/{key}" 

    with cols[col_index % 2]:
        # Start the custom HTML for the card
        st.markdown(f'<div class="project-card">', unsafe_allow_html=True)
        
        # Display logo and title
        st.markdown(f"<h4><i class='{data['logo']}'></i> {data['title']}</h4>", unsafe_allow_html=True)
        
        # Display description 
        description_text = data["description"]
        st.markdown(f'<p style="color:{COLORS["text_color"]}CC; margin-bottom: 15px;">{description_text}</p>', unsafe_allow_html=True)
        
        # Display Tech Stack
        st.markdown(f'<p style="color:{COLORS["primary_color"]}; font-weight: 600;">Tech Stack:</p>', unsafe_allow_html=True)
        tech_tags_html = "".join([f'<span class="tech-tag">{tag}</span>' for tag in data['tech_stack']])
        st.markdown(tech_tags_html, unsafe_allow_html=True)
        
        # --- NEW LIVE DEMO BUTTON (FIXED) ---
        # We use a standard Streamlit markdown link but style it heavily with CSS 
        # to ensure the link works reliably across Streamlit versions.
        
        st.markdown(f"""
            [<i class="fas fa-play-circle"></i> View Live Demo]({"/pages"})
        """)
        
        # End the custom HTML for the card
        st.markdown("</div>", unsafe_allow_html=True)
    col_index += 1


# --- CONTACT & SOCIAL SECTION (Sidebar) ---

st.sidebar.markdown("---")
st.sidebar.header("Connect & Contact")

# Define your personal links (PLACEHOLDER TEXT IS CLEARLY MARKED)
GITHUB_URL = "https://github.com/YOUR_GITHUB_PROFILE"
LINKEDIN_URL = "https://www.linkedin.com/in/YOUR_LINKEDIN_PROFILE"
GMAIL_ADDRESS = "your.email.address@gmail.com"

# Using clean Streamlit markdown with Font Awesome icons
st.sidebar.markdown(f"""
<p style='color:{COLORS["primary_color"]}; font-weight: 600;'>🔗 Professional Links:</p>
[<i class="fab fa-linkedin fa-lg"></i> LinkedIn]({LINKEDIN_URL})

[<i class="fab fa-github fa-lg"></i> GitHub]({GITHUB_URL})

<p style='color:{COLORS["primary_color"]}; font-weight: 600; margin-top: 15px;'>📧 Direct Contact:</p>
[<i class="fas fa-envelope fa-lg"></i> Email (Gmail)](mailto:{GMAIL_ADDRESS})
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Remember to replace all 'YOUR_...' links above!")
