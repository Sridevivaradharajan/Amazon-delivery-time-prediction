import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import requests
import pickle
import lightgbm as lgb
from io import BytesIO

def init_session_state():
    """Initialize session state variables"""
    if 'single_prediction_result' not in st.session_state:
        st.session_state.single_prediction_result = None
    if 'bulk_prediction_result' not in st.session_state:
        st.session_state.bulk_prediction_result = None
    if 'last_uploaded_file_name' not in st.session_state:
        st.session_state.last_uploaded_file_name = None
        
# Page configuration
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhances styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    h1 {
        color: #1a202c;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #2d3748;
        font-weight: 600;
    }
    
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #4a5568;
        font-weight: 500;
        padding: 8px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #718096;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: white !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stTimeInput label, .stDateInput label {
        color: #2d3748 !important;
        font-weight: 500;
        font-size: 14px;
    }
    
    .uploadedFile {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stInfo {
        background-color: #ebf8ff;
        border-left: 4px solid #4299e1;
        padding: 16px;
        border-radius: 8px;
    }
    
    .stSuccess {
        background-color: #f0fff4;
        border-left: 4px solid #48bb78;
        padding: 16px;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #fff5f5;
        border-left: 4px solid #f56565;
        padding: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and metrics
@st.cache_resource
def load_model_and_metrics():
    try:
        url = "https://raw.githubusercontent.com/Sridevivaradharajan/Amazon-delivery-time-prediction/main/Model.pkl"
        response = requests.get(url)
        if response.status_code == 200:
            model = pickle.load(BytesIO(response.content))
            metrics = {
                'r2': 0.8225,
                'rmse': 21.7800,
                'mae': 16.9278
            }
            return model, metrics
        else:
            st.error("Could not fetch model from GitHub.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Home Page
def home_page(metrics):
    st.markdown("""
        <style>
        /* Base style for all metric-card containers */
        .metric-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 20px 25px;
            border-radius: 16px;
            margin: 20px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }

        /* Purple border */
        .purple-card {
            border: 2px solid #764ba2;
        }

        /* Navy blue border */
        .navy-card {
            border: 2px solid #1a237e;
        }

        /* Make all section headings (except hero h1) light blue */
        .metric-card h3,
        h2 {
            color: #4facfe !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 60px 40px; 
            border-radius: 20px; 
            text-align: center; 
            color: #1a1f36; 
            box-shadow: 0 20px 50px rgba(0,0,0,0.25);
            border: 2px solid #1a237e;
            margin-bottom: 40px;
        ">
            <h1 style='font-size: 48px; font-weight: 700; margin-bottom: 12px; color: #1a1f36;'>
                Amazon Delivery Time Predictor
            </h1>
            <p style='font-size: 20px; font-weight: 400; opacity: 0.9;'>
                Predict delivery times accurately using advanced machine learning insights
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='metric-card purple-card' style='text-align: center;'>
                <h3>Precision Delivery</h3>
                <p style='color: #495057; font-size: 15px; line-height: 1.6;'>
                    High accuracy delivery time predictions using optimized LightGBM model
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='metric-card navy-card' style='text-align: center;'>
                <h3>Real-Time Insights</h3>
                <p style='color: #495057; font-size: 15px; line-height: 1.6;'>
                    Instant predictions for single or batch orders with fast processing
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='metric-card purple-card' style='text-align: center;'>
                <h3>Data Analytics</h3>
                <p style='color: #495057; font-size: 15px; line-height: 1.6;'>
                    Comprehensive dashboards for actionable delivery insights
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Performance
    st.markdown("""
        <h2 style='text-align: center; font-weight: 700; margin: 40px 0 30px 0;'>
            Model Performance Metrics
        </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="R² Score", value=f"{metrics['r2']*100:.2f}%", delta="Excellent Fit")
    with col2:
        st.metric(label="RMSE", value=f"{metrics['rmse']:.2f}", delta="Low Error", delta_color="inverse")
    with col3:
        st.metric(label="MAE", value=f"{metrics['mae']:.2f}", delta="Precise", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Highlights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class='metric-card navy-card'>
                <h3>Delivery Prediction</h3>
                <ul style='color: #495057; font-size: 15px; line-height: 2;'>
                    <li>Single order prediction with detailed insights</li>
                    <li>Batch CSV upload for multiple orders</li>
                    <li>Instant delivery time estimates</li>
                    <li>Downloadable prediction results in CSV</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='metric-card purple-card'>
                <h3>Data Analysis & Reporting</h3>
                <ul style='color: #495057; font-size: 15px; line-height: 2;'>
                    <li>Upload custom delivery datasets</li>
                    <li>Interactive charts and distribution analysis</li>
                    <li>Insights by distance, area, agent rating, and time</li>
                    <li>Exportable reports for decision making</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Haversine function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance (in km) between two lat/lon points"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Prepare input function
def prepare_input(input_data):
    """Prepare single prediction input with all required features in exact order"""
    # Calculate distance
    distance = haversine(
        input_data['Store_Latitude'], 
        input_data['Store_Longitude'],
        input_data['Drop_Latitude'], 
        input_data['Drop_Longitude']
    )
    
    # Create distance bin
    if distance < 5:
        distance_bin = '0_5km'
    elif distance < 10:
        distance_bin = '5_10km'
    elif distance < 15:
        distance_bin = '10_15km'
    elif distance < 20:
        distance_bin = '15_20km'
    else:
        distance_bin = '20+km'
    
    # Create DataFrame with EXACT feature order matching training
    df = pd.DataFrame([{
        # Numeric features (in order)
        'Agent_Age': input_data['Agent_Age'],
        'Agent_Rating': input_data['Agent_Rating'],
        'Store_Latitude': input_data['Store_Latitude'],
        'Store_Longitude': input_data['Store_Longitude'],
        'Drop_Latitude': input_data['Drop_Latitude'],
        'Drop_Longitude': input_data['Drop_Longitude'],
        'Distance': distance,
        'Hour': input_data['Hour'],
        'Month': input_data['Month'],
        'Prep_Time_Min': input_data['Prep_Time_Min'],
        # Categorical features (in order)
        'Weather': input_data['Weather'],
        'Traffic': input_data['Traffic'],
        'Vehicle': input_data['Vehicle'],
        'Area': input_data['Area'],
        'Category': input_data['Category'],
        'Distance_Bin': distance_bin,
        'DayOfWeek': input_data['DayOfWeek']
    }])
    
    return df

# Prediction Page
def prediction_page(model, metrics):
    st.markdown("""
        <h1 style='text-align: center; color: #1a202c; font-weight: 700; margin-bottom: 10px;'>
            Delivery Time Prediction
        </h1>
        <p style='text-align: center; color: #718096; font-size: 16px; margin-bottom: 30px;'>
            Predict delivery times using advanced machine learning algorithms
        </p>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.warning("No model loaded. Please train a model first.")
        return
    
    # Display model metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R²", f"{metrics['r2']*100:.2f}%")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.2f} min")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.2f} min")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])
    
    # ----------- Single Prediction Tab -----------
    with tab1:
        st.markdown("### Enter Delivery Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Agent Information")
            agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30, key="agent_age")
            agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1, key="agent_rating")
            
            st.markdown("#### Location Details")
            store_lat = st.number_input("Store Latitude", value=28.6139, format="%.4f", key="store_lat")
            store_lon = st.number_input("Store Longitude", value=77.2090, format="%.4f", key="store_lon")
            drop_lat = st.number_input("Drop Latitude", value=28.7041, format="%.4f", key="drop_lat")
            drop_lon = st.number_input("Drop Longitude", value=77.1025, format="%.4f", key="drop_lon")
            
            st.markdown("#### Order Details")
            hour = st.slider("Order Hour (24h format)", min_value=0, max_value=23, value=12, key="hour")
            month = st.selectbox("Order Month", list(range(1, 13)), index=2, key="month")
            prep_time = st.number_input("Preparation Time (minutes)", min_value=0, max_value=180, value=30, key="prep_time")
        
        with col2:
            st.markdown("#### Delivery Conditions")
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Fog', 'Sandstorms'], key="weather")
            traffic = st.selectbox("Traffic", ['Low', 'Medium', 'High', 'Jam', 'Very High'], key="traffic")
            vehicle = st.selectbox("Vehicle Type", ['bike', 'scooter', 'electric_scooter', 'motorcycle'], key="vehicle")
            area = st.selectbox("Area Type", ['Urban', 'Metropolitian', 'Semi-Urban', 'Rural'], key="area")
            category = st.selectbox("Product Category", 
                                    ['Electronics', 'Clothing', 'Grocery', 'Books', 
                                     'Home', 'Sports', 'Jewelry', 'Beauty', 'Toys', 
                                     'Food', 'Cosmetics', 'Fashion', 'Health', 
                                     'Furniture', 'Accessories', 'Footwear'], key="category")
            
            st.markdown("#### Additional Information")
            day_of_week = st.selectbox("Day of Week", 
                                       ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                        'Friday', 'Saturday', 'Sunday'], key="day_of_week")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("Predict Delivery Time", use_container_width=True, key="predict_btn")
        
        # Handle prediction and store in session state
        if predict_button:
            try:
                input_data = {
                    'Agent_Age': agent_age,
                    'Agent_Rating': agent_rating,
                    'Store_Latitude': store_lat,
                    'Store_Longitude': store_lon,
                    'Drop_Latitude': drop_lat,
                    'Drop_Longitude': drop_lon,
                    'Hour': hour,
                    'Month': month,
                    'Prep_Time_Min': prep_time,
                    'Weather': weather,
                    'Traffic': traffic,
                    'Vehicle': vehicle,
                    'Area': area,
                    'Category': category,
                    'DayOfWeek': day_of_week
                }
                
                df_input = prepare_input(input_data)
                
                # Convert categorical columns to category dtype
                categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Distance_Bin', 'DayOfWeek']
                for col in categorical_cols:
                    df_input[col] = df_input[col].astype('category')
                
                prediction = model.predict(df_input)[0]
                distance = df_input['Distance'].values[0]
                
                # Store result in session state
                st.session_state.single_prediction_result = {
                    'prediction': prediction,
                    'distance': distance,
                    'input_data': input_data
                }
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
        
        # Display results from session state
        if st.session_state.single_prediction_result is not None:
            result = st.session_state.single_prediction_result
            prediction = result['prediction']
            distance = result['distance']
            input_data = result['input_data']
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 50px 40px; border-radius: 16px; text-align: center; 
                                color: white; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);'>
                        <h2 style='color: white; margin-bottom: 10px; font-weight: 600; font-size: 24px;'>
                            Predicted Delivery Time
                        </h2>
                        <h1 style='color: white; font-size: 80px; margin: 20px 0; font-weight: 700;'>
                            {prediction:.1f}
                        </h1>
                        <h3 style='color: white; font-size: 28px; font-weight: 500;'>minutes</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            hours = int(prediction // 60)
            mins = int(prediction % 60)
            st.markdown(f"""
                <p style='text-align: center; font-size: 20px; color: #2d3748; margin-top: 10px;'>
                    Estimated Delivery: <strong>{hours} hour(s) and {mins} minute(s)</strong>
                </p>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Distance", f"{distance:.2f} km", delta="Calculated")
            with col2:
                avg_speed = (distance / (prediction / 60)) if prediction > 0 else 0
                st.metric("Avg Speed", f"{avg_speed:.2f} km/h", delta="Estimated")
            with col3:
                current_datetime = datetime.now()
                estimated_arrival = current_datetime + pd.Timedelta(minutes=prediction)
                st.metric("ETA", estimated_arrival.strftime("%H:%M"), delta=estimated_arrival.strftime("%d %b"))
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Key Influencing Factors")
            
            factors_col1, factors_col2 = st.columns(2)
            
            with factors_col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Weather:</strong> {input_data['Weather']} conditions</p>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Traffic:</strong> {input_data['Traffic']} congestion</p>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Vehicle:</strong> {input_data['Vehicle']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with factors_col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Area:</strong> {input_data['Area']}</p>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Category:</strong> {input_data['Category']}</p>
                        <p style='color: #4a5568; margin: 8px 0; font-size: 16px;'><strong>Agent Rating:</strong> {input_data['Agent_Rating']}/5.0</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # ----------- Bulk Prediction Tab -----------
    with tab2:
        st.markdown("### Upload CSV File for Batch Predictions")
        st.info("Your CSV should contain: Agent_Age, Agent_Rating, Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude, Hour, Month, Prep_Time_Min, Weather, Traffic, Vehicle, Area, Category, DayOfWeek")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="bulk_upload")
        with col2:
            sample_data = pd.DataFrame({
                'Agent_Age': [30, 25, 35],
                'Agent_Rating': [4.5, 4.2, 4.8],
                'Store_Latitude': [28.6139, 28.6500, 28.5800],
                'Store_Longitude': [77.2090, 77.2300, 77.1900],
                'Drop_Latitude': [28.7041, 28.7200, 28.6500],
                'Drop_Longitude': [77.1025, 77.1100, 77.1500],
                'Hour': [12, 14, 16],
                'Month': [3, 3, 3],
                'Prep_Time_Min': [30, 25, 35],
                'Weather': ['Sunny', 'Cloudy', 'Rainy'],
                'Traffic': ['Medium', 'High', 'Low'],
                'Vehicle': ['bike', 'scooter', 'motorcycle'],
                'Area': ['Urban', 'Metropolitian', 'Urban'],
                'Category': ['Electronics', 'Grocery', 'Clothing'],
                'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday']
            })
            csv_template = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Sample Template",
                data=csv_template,
                file_name="delivery_prediction_template.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template"
            )
        
        if uploaded_file is not None:
            try:
                # Cache the uploaded file data to avoid re-reading
                @st.cache_data
                def load_uploaded_csv(file):
                    return pd.read_csv(file)
                
                df = load_uploaded_csv(uploaded_file)
                st.success(f"File uploaded successfully! {len(df)} records found.")
                
                st.markdown("### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check if we need to clear previous results when new file is uploaded
                if 'last_uploaded_file_name' not in st.session_state:
                    st.session_state.last_uploaded_file_name = None
                
                if st.session_state.last_uploaded_file_name != uploaded_file.name:
                    st.session_state.bulk_prediction_result = None
                    st.session_state.last_uploaded_file_name = uploaded_file.name
                
                # Only show button if predictions haven't been generated yet
                if (st.session_state.bulk_prediction_result is None or 
                    'Predicted_Delivery_Time_Minutes' not in st.session_state.bulk_prediction_result.columns):
                    
                    if st.button("Generate Predictions", use_container_width=True, key="generate_bulk"):
                        with st.spinner("Processing predictions..."):
                            # Calculate distance for all rows
                            df['Distance'] = df.apply(lambda row: haversine(
                                row['Store_Latitude'], row['Store_Longitude'],
                                row['Drop_Latitude'], row['Drop_Longitude']
                            ), axis=1)
                            
                            # Create distance bins
                            max_distance = df['Distance'].max()
                            if max_distance <= 0:
                                st.error("All calculated distances are zero or negative. Please check coordinates.")
                                st.stop()
                            
                            # Create bins that ensure monotonic increase
                            if max_distance <= 5:
                                bins = [0, max_distance + 0.1]
                                labels = ['0_5km']
                            elif max_distance <= 10:
                                bins = [0, 5, max_distance + 0.1]
                                labels = ['0_5km', '5_10km']
                            elif max_distance <= 15:
                                bins = [0, 5, 10, max_distance + 0.1]
                                labels = ['0_5km', '5_10km', '10_15km']
                            elif max_distance <= 20:
                                bins = [0, 5, 10, 15, max_distance + 0.1]
                                labels = ['0_5km', '5_10km', '10_15km', '15_20km']
                            else:
                                bins = [0, 5, 10, 15, 20, max_distance + 0.1]
                                labels = ['0_5km', '5_10km', '10_15km', '15_20km', '20+km']
                            
                            df['Distance_Bin'] = pd.cut(df['Distance'], bins=bins, labels=labels, right=False)
                            df['Distance_Bin'] = df['Distance_Bin'].astype(str)
                            
                            # Reorder columns to match training order
                            feature_order = [
                                'Agent_Age', 'Agent_Rating', 
                                'Store_Latitude', 'Store_Longitude',
                                'Drop_Latitude', 'Drop_Longitude',
                                'Distance', 'Hour', 'Month', 'Prep_Time_Min',
                                'Weather', 'Traffic', 'Vehicle', 'Area', 
                                'Category', 'Distance_Bin', 'DayOfWeek'
                            ]
                            
                            # Check for missing columns
                            missing_cols = [col for col in feature_order if col not in df.columns]
                            if missing_cols:
                                st.error(f"Missing required columns: {missing_cols}")
                                st.stop()
                            
                            # Create prediction DataFrame with exact feature order
                            df_pred = df[feature_order].copy()
                            
                            # Convert categorical columns to category dtype
                            categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Distance_Bin', 'DayOfWeek']
                            for col in categorical_cols:
                                df_pred[col] = df_pred[col].astype('category')
                            
                            # Make predictions
                            predictions = model.predict(df_pred)
                            df['Predicted_Delivery_Time_Minutes'] = predictions
                            df['Predicted_Hours'] = (predictions // 60).astype(int)
                            df['Predicted_Minutes_Remaining'] = (predictions % 60).astype(int)
                            
                            # Store in session state AFTER predictions are added
                            st.session_state.bulk_prediction_result = df
                            st.experimental_rerun()   # Force rerun to show results
                
                # Display results from session state ONLY if predictions exist
                if (st.session_state.bulk_prediction_result is not None and 
                    'Predicted_Delivery_Time_Minutes' in st.session_state.bulk_prediction_result.columns):
                    df_results = st.session_state.bulk_prediction_result
                    
                    st.success("Predictions completed!")
                    
                    st.markdown("### Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Orders", f"{len(df_results):,}")
                    with col2: st.metric("Avg Time", f"{df_results['Predicted_Delivery_Time_Minutes'].mean():.1f} min")
                    with col3: st.metric("Min Time", f"{df_results['Predicted_Delivery_Time_Minutes'].min():.1f} min")
                    with col4: st.metric("Max Time", f"{df_results['Predicted_Delivery_Time_Minutes'].max():.1f} min")
                    
                    st.dataframe(df_results[['Agent_Age', 'Distance', 'Weather', 'Traffic', 
                                    'Category', 'Predicted_Delivery_Time_Minutes', 
                                    'Predicted_Hours', 'Predicted_Minutes_Remaining']], use_container_width=True)
                    
                    st.markdown("### Prediction Distribution")
                    
                    fig = px.histogram(df_results, x='Predicted_Delivery_Time_Minutes', nbins=30,
                                     title="Distribution of Predicted Delivery Times")
                    fig.update_traces(marker_color='#667eea', marker_line_color='white', marker_line_width=1.5)
                    fig.update_layout(
                        xaxis_title="Delivery Time (minutes)", 
                        yaxis_title="Count",
                        template="plotly_white",
                        font=dict(family="Inter, sans-serif"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add option to generate new predictions
                    st.markdown("---")
                    st.markdown("<h4 style='text-align: center; color: #333;'>Actions</h4>", unsafe_allow_html=True)
                    
                    # Use three columns for balanced centering
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        # Gradient-style button (custom CSS below)
                        st.markdown(
                            """
                            <style>
                            div[data-testid="stButton"] > button[kind="secondary"] {
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                border: none;
                                border-radius: 10px;
                                font-weight: 600;
                                padding: 0.6em 1em;
                                transition: all 0.3s ease;
                            }
                            div[data-testid="stButton"] > button[kind="secondary"]:hover {
                                transform: scale(1.02);
                                background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("Generate New Predictions", use_container_width=True, key="regenerate_bulk"):
                            st.session_state.bulk_prediction_result = None
                            st.experimental_rerun()
                    
                    with col2:
                        csv_output = df_results.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv_output,
                            file_name=f"delivery_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_predictions"
                        )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
                st.info("Please ensure your CSV file matches the required format.")

# User Data Analysis Page 
def user_data_analysis_page():
    st.markdown("""
        <h1 style='text-align: center; color: #1a202c; font-weight: 700; margin-bottom: 10px;'>
            Custom Data Analysis
        </h1>
        <p style='text-align: center; color: #718096; font-size: 16px; margin-bottom: 30px;'>
            Upload and analyze your delivery data with interactive visualizations
        </p>
    """, unsafe_allow_html=True)
    
    st.info("Upload a CSV file with delivery data to unlock comprehensive insights and visualizations.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='user_analysis')
    
    if uploaded_file is not None:
        try:
            # Read and cache the dataframe
            @st.cache_data
            def load_and_process_data(file):
                df = pd.read_csv(file)
                
                # Calculate distance if needed
                if all(col in df.columns for col in ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']):
                    if 'Distance' not in df.columns:
                        df['Distance'] = df.apply(lambda row: haversine(
                            row['Store_Latitude'], row['Store_Longitude'],
                            row['Drop_Latitude'], row['Drop_Longitude']
                        ), axis=1)
                
                return df
            
            df = load_and_process_data(uploaded_file)
            
            st.success(f"File uploaded successfully! {len(df):,} records loaded and ready for analysis.")
            
            # Data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistical Summary
            st.markdown("### Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #667eea; margin-bottom: 16px;'>Dataset Information</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("Total Records", f"{len(df):,}")
                    st.metric("Total Features", len(df.columns))
                with info_col2:
                    st.metric("Numeric Features", len(df.select_dtypes(include=['number']).columns))
                    st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'category']).columns))
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #667eea; margin-bottom: 16px;'>Key Metrics</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    if 'Delivery_Time' in df.columns:
                        st.metric("Avg Delivery Time", f"{df['Delivery_Time'].mean():.1f} min")
                        st.metric("Max Delivery Time", f"{df['Delivery_Time'].max():.1f} min")
                    elif 'Distance' in df.columns:
                        st.metric("Avg Distance", f"{df['Distance'].mean():.1f} km")
                        st.metric("Max Distance", f"{df['Distance'].max():.1f} km")
                
                with metric_col2:
                    if 'Agent_Rating' in df.columns:
                        st.metric("Avg Agent Rating", f"{df['Agent_Rating'].mean():.2f}/5.0")
                        st.metric("Top Rating", f"{df['Agent_Rating'].max():.2f}/5.0")
                    elif 'Delivery_Time' in df.columns:
                        st.metric("Min Delivery Time", f"{df['Delivery_Time'].min():.1f} min")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create tabs for organized visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                "Distributions", "Relationships", "Time Analysis", "Geographic Analysis", "Data Quality & Correlation"
            ])
            
            # Tab 1: Distributions - ONLY LOAD WHEN TAB IS SELECTED
            with viz_tab1:
                st.markdown("#### Distribution Analysis")
                
                # Use expander to further optimize loading
                with st.expander("View Distribution Charts", expanded=True):
                    if 'Delivery_Time' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = px.histogram(df, x='Delivery_Time', nbins=30, title="Delivery Time Distribution", marginal="box")
                            fig1.update_traces(marker_color='#667eea', marker_line_color='white', marker_line_width=1.5)
                            fig1.update_layout(template="plotly_white", xaxis_title="Delivery Time (minutes)", yaxis_title="Frequency", showlegend=False, height=400)
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            fig_box = px.box(df, y='Delivery_Time', title="Delivery Time Spread & Outliers")
                            fig_box.update_traces(marker_color='#764ba2')
                            fig_box.update_layout(template="plotly_white", yaxis_title="Delivery Time (minutes)", showlegend=False, height=400)
                            st.plotly_chart(fig_box, use_container_width=True)
                    
                    if 'Distance' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_dist = px.histogram(df, x='Distance', nbins=30, title="Distance Distribution", marginal="violin")
                            fig_dist.update_traces(marker_color='#4facfe', marker_line_color='white', marker_line_width=1.5)
                            fig_dist.update_layout(template="plotly_white", xaxis_title="Distance (km)", yaxis_title="Frequency", showlegend=False, height=400)
                            st.plotly_chart(fig_dist, use_container_width=True)
                        with col2:
                            if 'Agent_Rating' in df.columns:
                                fig_rating = px.histogram(df, x='Agent_Rating', title="Agent Rating Distribution", nbins=20)
                                fig_rating.update_traces(marker_color='#f093fb', marker_line_color='white', marker_line_width=1.5)
                                fig_rating.update_layout(template="plotly_white", xaxis_title="Agent Rating", yaxis_title="Frequency", showlegend=False, height=400)
                                st.plotly_chart(fig_rating, use_container_width=True)
            
            # Tab 2: Relationships
            with viz_tab2:
                st.markdown("#### Relationship Analysis")
                
                with st.expander("View Relationship Charts", expanded=True):
                    if 'Distance' in df.columns and 'Delivery_Time' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Limit data points for scatter plot performance
                            plot_df = df.sample(n=min(1000, len(df)), random_state=42)
                            fig3 = px.scatter(plot_df, x='Distance', y='Delivery_Time',
                                              color='Traffic' if 'Traffic' in df.columns else None,
                                              title="Distance vs Delivery Time", trendline="ols")
                            fig3.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
                            fig3.update_layout(template="plotly_white", xaxis_title="Distance (km)", yaxis_title="Delivery Time (minutes)", height=400)
                            st.plotly_chart(fig3, use_container_width=True)
                            if len(df) > 1000:
                                st.caption(f"Showing 1,000 of {len(df):,} points for performance")
                        
                        with col2:
                            if 'Agent_Rating' in df.columns:
                                plot_df = df.sample(n=min(1000, len(df)), random_state=42)
                                fig_rating_time = px.scatter(plot_df, x='Agent_Rating', y='Delivery_Time',
                                                            color='Vehicle' if 'Vehicle' in df.columns else None,
                                                            title="Agent Rating vs Delivery Time")
                                fig_rating_time.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
                                fig_rating_time.update_layout(template="plotly_white", xaxis_title="Agent Rating", yaxis_title="Delivery Time (minutes)", height=400)
                                st.plotly_chart(fig_rating_time, use_container_width=True)
                    
                    if 'Category' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            cat_counts = df['Category'].value_counts().reset_index()
                            cat_counts.columns = ['Category', 'Count']
                            fig2 = px.bar(cat_counts, x='Category', y='Count', title="Orders by Product Category")
                            fig2.update_traces(marker_color='#667eea')
                            fig2.update_layout(template="plotly_white", xaxis_tickangle=-45, showlegend=False, height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                        with col2:
                            fig_pie = px.pie(cat_counts, values='Count', names='Category', title="Category Distribution")
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            fig_pie.update_layout(template="plotly_white", height=400)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    if 'Weather' in df.columns and 'Traffic' in df.columns:
                        fig_heatmap = px.density_heatmap(df, x='Weather', y='Traffic', title="Weather vs Traffic Conditions Heatmap")
                        fig_heatmap.update_layout(template="plotly_white", height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Tab 3: Time Analysis
            with viz_tab3:
                st.markdown("#### Time-Based Analysis")
                
                with st.expander("View Time-Based Charts", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Hour' in df.columns:
                            hour_counts = df['Hour'].value_counts().sort_index().reset_index()
                            hour_counts.columns = ['Hour', 'Count']
                            fig5 = px.bar(hour_counts, x='Hour', y='Count', title="Orders by Hour of Day")
                            fig5.update_traces(marker_color='#667eea')
                            fig5.update_layout(template="plotly_white", xaxis_title="Hour (24-hour format)", yaxis_title="Number of Orders", showlegend=False, height=400)
                            st.plotly_chart(fig5, use_container_width=True)
                    with col2:
                        if 'DayOfWeek' in df.columns:
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            day_counts = df['DayOfWeek'].value_counts().reindex(day_order).reset_index()
                            day_counts.columns = ['Day', 'Count']
                            fig_day = px.bar(day_counts, x='Day', y='Count', title="Orders by Day of Week")
                            fig_day.update_traces(marker_color='#764ba2')
                            fig_day.update_layout(template="plotly_white", xaxis_title="Day of Week", yaxis_title="Number of Orders", showlegend=False, height=400)
                            st.plotly_chart(fig_day, use_container_width=True)
                    
                    if 'Month' in df.columns:
                        month_counts = df['Month'].value_counts().sort_index().reset_index()
                        month_counts.columns = ['Month', 'Count']
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        month_counts['Month_Name'] = month_counts['Month'].apply(lambda x: month_names[int(x)-1] if 1 <= x <= 12 else str(x))
                        fig_month = px.line(month_counts, x='Month_Name', y='Count', title="Orders by Month", markers=True)
                        fig_month.update_traces(line_color='#4facfe', marker=dict(size=10, color='#667eea', line=dict(width=2, color='white')))
                        fig_month.update_layout(template="plotly_white", xaxis_title="Month", yaxis_title="Number of Orders", height=400)
                        st.plotly_chart(fig_month, use_container_width=True)
            
            # Tab 4: Geographic Analysis
            with viz_tab4:
                st.markdown("#### Geographic Analysis")
                
                with st.expander("View Geographic Charts", expanded=True):
                    if all(col in df.columns for col in ['Store_Latitude', 'Store_Longitude']):
                        try:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("##### Store Locations Map")
                                # Sample for performance
                                map_df = df[['Store_Latitude', 'Store_Longitude']].dropna().sample(n=min(500, len(df)), random_state=42).rename(columns={'Store_Latitude': 'lat', 'Store_Longitude': 'lon'})
                                if not map_df.empty:
                                    st.map(map_df, zoom=10)
                                    if len(df) > 500:
                                        st.caption(f"Showing 500 of {len(df):,} locations")
                            with col2:
                                if 'Delivery_Time' in df.columns:
                                    st.markdown("##### Performance by Location")
                                    sample_df = df.sample(n=min(500, len(df)), random_state=42)
                                    fig_geo = px.scatter_mapbox(sample_df, lat='Store_Latitude', lon='Store_Longitude',
                                                                color='Delivery_Time', size='Delivery_Time',
                                                                hover_data=['Distance'] if 'Distance' in df.columns else None,
                                                                zoom=10, title="Delivery Time by Location")
                                    fig_geo.update_layout(mapbox_style="open-street-map", height=400)
                                    st.plotly_chart(fig_geo, use_container_width=True)
                        except Exception:
                            st.info("Could not render geographic visualizations.")
                    
                    if 'Area' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            area_counts = df['Area'].value_counts().reset_index()
                            area_counts.columns = ['Area', 'Count']
                            fig_area = px.bar(area_counts, x='Area', y='Count', title="Orders by Area Type")
                            fig_area.update_traces(marker_color='#667eea')
                            fig_area.update_layout(template="plotly_white", showlegend=False, height=400)
                            st.plotly_chart(fig_area, use_container_width=True)
                        with col2:
                            if 'Delivery_Time' in df.columns:
                                fig_area_time = px.box(df, x='Area', y='Delivery_Time', title="Delivery Time by Area Type")
                                fig_area_time.update_traces(marker_color='#764ba2')
                                fig_area_time.update_layout(template="plotly_white", yaxis_title="Delivery Time (minutes)", height=400)
                                st.plotly_chart(fig_area_time, use_container_width=True)
            
            # Tab 5: Data Quality & Correlation
            with viz_tab5:
                with st.expander("View Correlation Matrix", expanded=True):
                    st.markdown("### Feature Correlation Matrix")
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        # Limit to max 15 columns for performance
                        if len(numeric_cols) > 15:
                            numeric_cols = numeric_cols[:15]
                            st.info(f"Showing top 15 numeric features for performance")
                        
                        corr = df[numeric_cols].corr()
                        fig6 = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu_r',
                            zmid=0,
                            text=np.round(corr.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            showscale=True,
                            colorbar=dict(title="Correlation")
                        ))
                        fig6.update_layout(title="Correlation Heatmap (Numeric Features)", template="plotly_white", height=600, xaxis_showgrid=False, yaxis_showgrid=False)
                        st.plotly_chart(fig6, use_container_width=True)

                st.markdown("### Data Quality Report")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='metric-card'><h4 style='color: #667eea;'>Missing Values</h4></div>", unsafe_allow_html=True)
                    missing = df.isnull().sum()
                    missing_df = pd.DataFrame({'Feature': missing.index, 'Missing Count': missing.values, 'Percentage': (missing.values / len(df) * 100).round(2)})
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    if len(missing_df) > 0:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("No missing values detected")
                
                with col2:
                    st.markdown("<div class='metric-card'><h4 style='color: #667eea;'>Duplicate Records</h4></div>", unsafe_allow_html=True)
                    duplicates = df.duplicated().sum()
                    if duplicates > 0:
                        st.warning(f"Found {duplicates} duplicate records ({(duplicates/len(df)*100):.2f}%)")
                    else:
                        st.success("No duplicate records detected")
                
                with col3:
                    st.markdown("<div class='metric-card'><h4 style='color: #667eea;'>Data Types</h4></div>", unsafe_allow_html=True)
                    dtype_counts = df.dtypes.value_counts()
                    dtype_df = pd.DataFrame({'Type': dtype_counts.index.astype(str), 'Count': dtype_counts.values})
                    st.dataframe(dtype_df, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download Processed Data
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name=f"processed_delivery_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_analysis_data"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure the CSV file is properly formatted.")

# About Page
def about_page():
    st.markdown("""
    <style>
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 20px 25px;
        border-radius: 16px;
        margin: 20px 0; /* Matches home page */
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .purple-card { border: 2px solid #764ba2; }
    .navy-card { border: 2px solid #1a237e; }
    .metric-card h3, h2 { color: #4facfe !important; }
    .metric-card p, .metric-card li { color: #495057; font-size: 15px; line-height: 1.8; }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section (same padding & margin as home_page)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 60px 40px; 
        border-radius: 20px; 
        text-align: center; 
        color: #1a1f36; 
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
        border: 2px solid #1a237e;
        margin-bottom: 40px;
    ">
        <h1 style='font-size: 48px; font-weight: 700; margin-bottom: 12px; color: #1a1f36;'>
            About This App
        </h1>
        <p style='font-size: 20px; font-weight: 400; opacity: 0.9; max-width: 800px; margin: 0 auto;'>
            Delivering reliable and accurate delivery time predictions powered by LightGBM with Optuna optimization. Provides advanced analytics and actionable insights to improve logistics efficiency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards (Two-column layout)
    col1, col2 = st.columns(2, gap="large")  # Ensures same spacing as home_page

    with col1:
        st.markdown("""
        <div class='metric-card purple-card' style='text-align: left;'>
            <h3>Technology Stack</h3>
            <ul>
                <li><strong>Model:</strong> LightGBM with Optuna tuning</li>
                <li><strong>Framework:</strong> Streamlit UI</li>
                <li><strong>Visualizations:</strong> Plotly charts</li>
                <li><strong>Data Processing:</strong> Pandas & NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card navy-card' style='text-align: left;'>
            <h3>Key Features</h3>
            <ul>
                <li>Real time delivery predictions</li>
                <li>Batch processing for multiple orders</li>
                <li>Interactive dashboards & analytics</li>
                <li>Geographic distance & area insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# MAIN APP
def main():
    init_session_state()  # Initialize first
    model, metrics = load_model_and_metrics()  # Load model once
    
    # Sidebar
    st.sidebar.markdown("<h2 style='color: white; margin-bottom: 2rem;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Prediction", "Data Analysis", "About"], label_visibility="collapsed")
    
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; color: white;">
            <h4 style="color: white; margin-bottom: 1rem;">Quick Info</h4>
            <p style="font-size: 0.9rem; line-height: 1.6; opacity: 0.9;">
                This AI powered system uses advanced machine learning to predict delivery times
                with high accuracy based on multiple real world factors.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 3rem;">
            <p>© 2025 Amazon Delivery Predictor</p>
            <p>Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "Home":
        home_page(metrics)
    elif page == "Prediction":
        prediction_page(model, metrics)
    elif page == "Data Analysis":
        user_data_analysis_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()













