import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="MotoPredict Pro", page_icon="🏎️", layout="wide")

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle, #1a1a1a 0%, #000000 100%);
        color: #e0e0e0;
    }
    .prediction-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .mpg-display {
        font-size: 80px;
        font-weight: 900;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    h1, h2, h3 { color: #4facfe !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE (Hardcoded to fix HTTPError) ---
@st.cache_resource
def init_model():
    # Hardcoded mtcars data (essential columns: model, wt, mpg)
    data = {
        'model': ['Mazda RX4', 'Mazda RX4 Wag', 'Datsun 710', 'Hornet 4 Drive', 'Hornet Sportabout', 'Valiant', 'Duster 360', 'Merc 240D', 'Merc 230', 'Merc 280', 'Merc 280C', 'Merc 450SE', 'Merc 450SL', 'Merc 450SLC', 'Cadillac Fleetwood', 'Lincoln Continental', 'Chrysler Imperial', 'Fiat 128', 'Honda Civic', 'Toyota Corolla', 'Toyota Corona', 'Dodge Challenger', 'AMC Javelin', 'Camaro Z28', 'Pontiac Firebird', 'Fiat X1-9', 'Porsche 914-2', 'Lotus Europa', 'Ford Pantera L', 'Ferrari Dino', 'Maserati Bora', 'Volvo 142E'],
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
        'wt': [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78]
    }
    df = pd.DataFrame(data)
    
    # Train Model
    X = df[['wt']]
    y = df['mpg']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

model, df = init_model()

# --- SIDEBAR ---
st.sidebar.title("MotoPredict Pro")
st.sidebar.markdown("---")
st.sidebar.write("⚡ **Mode:** Local Data (Offline)")
st.sidebar.info("The HTTP Error was solved by embedding the dataset directly into the app.")

# --- MAIN INTERFACE ---
st.title("🏎️ Car Intelligence Dashboard")
st.write("Unique ML Model for Mileage Prediction")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### ⚙️ Vehicle Specs")
    input_wt = st.slider("Set Car Weight (1000 lbs)", 
                        min_value=float(df.wt.min()), 
                        max_value=float(df.wt.max()), 
                        value=3.2, step=0.1)
    
    prediction = model.predict([[input_wt]])[0]
    
    st.markdown(f"""
        <div class="prediction-container">
            <p style="letter-spacing: 3px; font-size: 14px; color: #888;">PREDICTED EFFICIENCY</p>
            <h1 class="mpg-display">{prediction:.1f}</h1>
            <p style="font-size: 18px; color: #4facfe;">MILES PER GALLON</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Visual Data Analysis")
    fig = px.scatter(df, x="wt", y="mpg", hover_name="model", 
                     template="plotly_dark", color="mpg",
                     color_continuous_scale="Viridis")
    
    # Add predicted point
    fig.add_trace(go.Scatter(x=[input_wt], y=[prediction], mode='markers',
                             marker=dict(color='#ff00ff', size=18, symbol='star', line=dict(width=2, color="white")),
                             name="Prediction"))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)
st.caption("Developed by Ragul Saravanan | Dataset: mtcars (Built-in)")
