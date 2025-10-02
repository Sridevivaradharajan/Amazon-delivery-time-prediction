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

def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.load(BytesIO(response.content))
        return model
    else:
        st.error("Failed to download the model.")
        return None

# Extract the file ID from the Google Drive link
file_id = '14bpwlmue2FZo1-lCwu7kCCLlJkAET4eo'

# Load the model
model = load_model_from_drive(file_id)

if model:
    st.success("Model loaded successfully!")
    # Now you can use the model for predictions

# Page configuration
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
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
    
    .stSelectbox label, .stNumberInput label, .stTimeInput label, .stDateInput label {
        color: #2d3748;
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
        with open('lightgbm_optuna_tuned.pkl', 'rb') as f:
            model = pickle.load(f)
        metrics = {
            'r2': 0.8221,
            'rmse': 21.8009,
            'mae': 16.9764
        }
        return model, metrics
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'lightgbm_optuna_tuned.pkl' is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Haversine distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Prepare input data
def prepare_input(data_dict):
    distance = haversine(
        data_dict['Store_Latitude'], data_dict['Store_Longitude'],
        data_dict['Drop_Latitude'], data_dict['Drop_Longitude']
    )
    
    if distance < 5:
        distance_bin = '0-5'
    elif distance < 10:
        distance_bin = '5-10'
    elif distance < 15:
        distance_bin = '10-15'
    elif distance < 20:
        distance_bin = '15-20'
    else:
        distance_bin = '20+'
    
    features = {
        'Agent_Age': data_dict['Agent_Age'],
        'Agent_Rating': data_dict['Agent_Rating'],
        'Store_Latitude': data_dict['Store_Latitude'],
        'Store_Longitude': data_dict['Store_Longitude'],
        'Drop_Latitude': data_dict['Drop_Latitude'],
        'Drop_Longitude': data_dict['Drop_Longitude'],
        'Distance': distance,
        'Hour': data_dict['Hour'],
        'Month': data_dict['Month'],
        'Prep_Time_Min': data_dict['Prep_Time_Min'],
        'Weather': data_dict['Weather'],
        'Traffic': data_dict['Traffic'],
        'Vehicle': data_dict['Vehicle'],
        'Area': data_dict['Area'],
        'Category': data_dict['Category'],
        'Distance_Bin': distance_bin,
        'DayOfWeek': data_dict['DayOfWeek']
    }
    
    return pd.DataFrame([features])

# Home Page
def home_page(metrics):
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 60px 40px; border-radius: 16px; text-align: center; 
                    color: white; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);'>
            <h1 style='color: white; font-size: 52px; margin-bottom: 16px; font-weight: 700;'>
                Amazon Delivery Time Predictor
            </h1>
            <p style='font-size: 22px; opacity: 0.95; font-weight: 400; letter-spacing: 0.3px;'>
                Advanced Machine Learning System for Accurate Delivery Time Estimation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 48px; margin-bottom: 12px;'>🎯</div>
                <h3 style='color: #667eea; margin-bottom: 8px;'>High Accuracy</h3>
                <p style='color: #718096; font-size: 15px; line-height: 1.6;'>
                    Advanced LightGBM model optimized with Optuna hyperparameter tuning
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 48px; margin-bottom: 12px;'>⚡</div>
                <h3 style='color: #667eea; margin-bottom: 8px;'>Real-Time Predictions</h3>
                <p style='color: #718096; font-size: 15px; line-height: 1.6;'>
                    Instant delivery time estimates with sub-second response time
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 48px; margin-bottom: 12px;'>📊</div>
                <h3 style='color: #667eea; margin-bottom: 8px;'>Deep Analytics</h3>
                <p style='color: #718096; font-size: 15px; line-height: 1.6;'>
                    Comprehensive delivery insights with interactive visualizations
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style='text-align: center; color: #1a202c; font-weight: 700; margin: 40px 0 30px 0;'>
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 20px;'>Delivery Prediction</h3>
                <ul style='color: #4a5568; font-size: 15px; line-height: 2;'>
                    <li>Single order prediction with detailed insights</li>
                    <li>Bulk CSV upload for batch processing</li>
                    <li>Instant time estimates with confidence metrics</li>
                    <li>Downloadable results in CSV format</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 20px;'>Data Analysis</h3>
                <ul style='color: #4a5568; font-size: 15px; line-height: 2;'>
                    <li>Upload custom delivery data for analysis</li>
                    <li>Interactive visualizations and charts</li>
                    <li>Comprehensive performance metrics</li>
                    <li>Exportable insights and reports</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Use the sidebar to navigate between different sections of the application.")

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
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R²", f"{metrics['r2']*100:.2f}%")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.2f} hrs")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.2f} hrs")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])
    
    with tab1:
        st.markdown("### Enter Delivery Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Agent Information")
            agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
            agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            
            st.markdown("#### Location Details")
            store_lat = st.number_input("Store Latitude", value=28.6139, format="%.4f")
            store_lon = st.number_input("Store Longitude", value=77.2090, format="%.4f")
            drop_lat = st.number_input("Drop Latitude", value=28.7041, format="%.4f")
            drop_lon = st.number_input("Drop Longitude", value=77.1025, format="%.4f")
            
            st.markdown("#### Order Details")
            order_date = st.date_input("Order Date", value=datetime.now())
            order_time = st.time_input("Order Time", value=time(12, 0))
            prep_time = st.number_input("Preparation Time (minutes)", min_value=0, max_value=180, value=30)
        
        with col2:
            st.markdown("#### Delivery Conditions")
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Fog', 'Windy'])
            traffic = st.selectbox("Traffic", ['Low', 'Medium', 'High', 'Jam'])
            vehicle = st.selectbox("Vehicle Type", ['Bike', 'Scooter', 'Van', 'Truck'])
            area = st.selectbox("Area Type", ['Urban', 'Metropolitan', 'Semi-Urban'])
            category = st.selectbox("Product Category", 
                                    ['Electronics', 'Clothing', 'Grocery', 'Books', 
                                     'Home', 'Sports', 'Jewelry', 'Beauty'])
            
            st.markdown("#### Additional Information")
            day_of_week = st.selectbox("Day of Week", 
                                       ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                        'Friday', 'Saturday', 'Sunday'])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("Predict Delivery Time", use_container_width=True)
        
        if predict_button:
            try:
                input_data = {
                    'Agent_Age': agent_age,
                    'Agent_Rating': agent_rating,
                    'Store_Latitude': store_lat,
                    'Store_Longitude': store_lon,
                    'Drop_Latitude': drop_lat,
                    'Drop_Longitude': drop_lon,
                    'Hour': order_time.hour,
                    'Month': order_date.month,
                    'Prep_Time_Min': prep_time,
                    'Weather': weather,
                    'Traffic': traffic,
                    'Vehicle': vehicle,
                    'Area': area,
                    'Category': category,
                    'DayOfWeek': day_of_week
                }
                
                df_input = prepare_input(input_data)
                
                categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Distance_Bin', 'DayOfWeek']
                for col in categorical_cols:
                    df_input[col] = df_input[col].astype('category')
                
                prediction = model.predict(df_input)[0]
                
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
                            <h3 style='color: white; font-size: 28px; font-weight: 500;'>hours</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                distance = df_input['Distance'].values[0]
                
                with col1:
                    st.metric("Distance", f"{distance:.2f} km", delta="Calculated")
                with col2:
                    st.metric("Avg Speed", f"{distance/prediction:.2f} km/h", delta="Estimated")
                with col3:
                    delivery_datetime = datetime.combine(order_date, order_time)
                    estimated_arrival = delivery_datetime + pd.Timedelta(hours=prediction)
                    st.metric("ETA", estimated_arrival.strftime("%H:%M"), delta=estimated_arrival.strftime("%d %b"))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Key Influencing Factors")
                
                factors_col1, factors_col2 = st.columns(2)
                
                with factors_col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Weather:</strong> {weather} conditions</p>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Traffic:</strong> {traffic} congestion</p>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Vehicle:</strong> {vehicle}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with factors_col2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Area:</strong> {area}</p>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Category:</strong> {category}</p>
                            <p style='color: #4a5568; margin: 8px 0;'><strong>Agent Rating:</strong> {agent_rating}/5.0</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("### Upload CSV File for Batch Predictions")
        
        st.info("Your CSV should contain: Agent_Age, Agent_Rating, Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude, Hour, Month, Prep_Time_Min, Weather, Traffic, Vehicle, Area, Category, DayOfWeek")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        with col2:
            if st.button("Download Sample Template", use_container_width=True):
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
                    'Vehicle': ['Bike', 'Scooter', 'Van'],
                    'Area': ['Urban', 'Metropolitan', 'Urban'],
                    'Category': ['Electronics', 'Grocery', 'Clothing'],
                    'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday']
                })
                
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="delivery_prediction_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! {len(df)} records found.")
                
                st.markdown("### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("Generate Predictions", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        df['Distance'] = df.apply(lambda row: haversine(
                            row['Store_Latitude'], row['Store_Longitude'],
                            row['Drop_Latitude'], row['Drop_Longitude']
                        ), axis=1)
                        
                        bins = [0, 5, 10, 15, 20, df['Distance'].max()+1]
                        labels = ['0-5', '5-10', '10-15', '15-20', '20+']
                        df['Distance_Bin'] = pd.cut(df['Distance'], bins=bins, labels=labels, right=False)
                        
                        categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Distance_Bin', 'DayOfWeek']
                        for col in categorical_cols:
                            df[col] = df[col].astype('category')
                        
                        predictions = model.predict(df)
                        df['Predicted_Delivery_Time'] = predictions
                    
                    st.success("Predictions completed!")
                    
                    st.markdown("### Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Orders", f"{len(df):,}")
                    with col2:
                        st.metric("Avg Time", f"{df['Predicted_Delivery_Time'].mean():.1f} hrs")
                    with col3:
                        st.metric("Min Time", f"{df['Predicted_Delivery_Time'].min():.1f} hrs")
                    with col4:
                        st.metric("Max Time", f"{df['Predicted_Delivery_Time'].max():.1f} hrs")
                    
                    st.dataframe(df[['Agent_Age', 'Distance', 'Weather', 'Traffic', 
                                    'Category', 'Predicted_Delivery_Time']], use_container_width=True)
                    
                    st.markdown("### Prediction Distribution")
                    
                    fig = px.histogram(df, x='Predicted_Delivery_Time', nbins=30,
                                     title="Distribution of Predicted Delivery Times",
                                     color_discrete_sequence=['#667eea'])
                    fig.update_layout(
                        xaxis_title="Delivery Time (hours)", 
                        yaxis_title="Count",
                        template="plotly_white",
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"delivery_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file matches the required format.")

# User Data Analysis Page - ENHANCED WITH COLORS
def user_data_analysis_page():
    st.markdown("""
        <h1 style='text-align: center; color: #1a202c; font-weight: 700; margin-bottom: 10px;'>
            Custom Data Analysis
        </h1>
        <p style='text-align: center; color: #718096; font-size: 16px; margin-bottom: 30px;'>
            Upload and analyze your delivery data with interactive visualizations
        </p>
    """, unsafe_allow_html=True)
    
    st.info("📁 Upload a CSV file with delivery data to unlock comprehensive insights and visualizations.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='user_analysis')
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df):,} records loaded and ready for analysis.")
            
            # Calculate distance
            if all(col in df.columns for col in ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']):
                if 'Distance' not in df.columns:
                    df['Distance'] = df.apply(lambda row: haversine(
                        row['Store_Latitude'], row['Store_Longitude'],
                        row['Drop_Latitude'], row['Drop_Longitude']
                    ), axis=1)
                    st.info("📍 Distance calculated from coordinates using Haversine formula")
            
            # Data preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistical Summary
            st.markdown("### 📈 Statistical Summary")
            
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
                        st.metric("Avg Delivery Time", f"{df['Delivery_Time'].mean():.1f} hrs")
                        st.metric("Max Delivery Time", f"{df['Delivery_Time'].max():.1f} hrs")
                    elif 'Distance' in df.columns:
                        st.metric("Avg Distance", f"{df['Distance'].mean():.1f} km")
                        st.metric("Max Distance", f"{df['Distance'].max():.1f} km")
                
                with metric_col2:
                    if 'Agent_Rating' in df.columns:
                        st.metric("Avg Agent Rating", f"{df['Agent_Rating'].mean():.2f}/5.0")
                        st.metric("Top Rating", f"{df['Agent_Rating'].max():.2f}/5.0")
                    elif 'Delivery_Time' in df.columns:
                        st.metric("Min Delivery Time", f"{df['Delivery_Time'].min():.1f} hrs")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create tabs for organized visualizations
            st.markdown("### 🎨 Interactive Visualizations")
            
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "📊 Distributions", "📈 Relationships", "⏰ Time Analysis", "🗺️ Geographic Analysis"
            ])
            
            # Tab 1: Distributions
            with viz_tab1:
                st.markdown("#### Distribution Analysis")
                
                if 'Delivery_Time' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.histogram(
                            df, x='Delivery_Time', nbins=30,
                            title="Delivery Time Distribution",
                            color_discrete_sequence=['#667eea'],
                            marginal="box"
                        )
                        fig1.update_traces(marker_line_color='white', marker_line_width=1.5)
                        fig1.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Delivery Time (hours)",
                            yaxis_title="Frequency",
                            showlegend=False
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig_box = px.box(
                            df, y='Delivery_Time',
                            title="Delivery Time Spread & Outliers",
                            color_discrete_sequence=['#764ba2']
                        )
                        fig_box.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            yaxis_title="Delivery Time (hours)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                
                if 'Distance' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_dist = px.histogram(
                            df, x='Distance', nbins=30,
                            title="Distance Distribution",
                            color_discrete_sequence=['#4facfe'],
                            marginal="violin"
                        )
                        fig_dist.update_traces(marker_line_color='white', marker_line_width=1.5)
                        fig_dist.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Distance (km)",
                            yaxis_title="Frequency",
                            showlegend=False
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        if 'Agent_Rating' in df.columns:
                            fig_rating = px.histogram(
                                df, x='Agent_Rating',
                                title="Agent Rating Distribution",
                                color_discrete_sequence=['#f093fb'],
                                nbins=20
                            )
                            fig_rating.update_traces(marker_line_color='white', marker_line_width=1.5)
                            fig_rating.update_layout(
                                template="plotly_white",
                                font=dict(family="Inter, sans-serif"),
                                xaxis_title="Agent Rating",
                                yaxis_title="Frequency",
                                showlegend=False
                            )
                            st.plotly_chart(fig_rating, use_container_width=True)
            
            # Tab 2: Relationships
            with viz_tab2:
                st.markdown("#### Relationship Analysis")
                
                if 'Distance' in df.columns and 'Delivery_Time' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        color_col = 'Traffic' if 'Traffic' in df.columns else None
                        fig3 = px.scatter(
                            df, x='Distance', y='Delivery_Time',
                            color=color_col,
                            hover_data=df.columns.tolist(),
                            title="Distance vs Delivery Time",
                            trendline="ols",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig3.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                        fig3.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Distance (km)",
                            yaxis_title="Delivery Time (hours)"
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        if 'Agent_Rating' in df.columns:
                            fig_rating_time = px.scatter(
                                df, x='Agent_Rating', y='Delivery_Time',
                                color='Vehicle' if 'Vehicle' in df.columns else None,
                                title="Agent Rating vs Delivery Time",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig_rating_time.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                            fig_rating_time.update_layout(
                                template="plotly_white",
                                font=dict(family="Inter, sans-serif"),
                                xaxis_title="Agent Rating",
                                yaxis_title="Delivery Time (hours)"
                            )
                            st.plotly_chart(fig_rating_time, use_container_width=True)
                
                if 'Category' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cat_counts = df['Category'].value_counts().reset_index()
                        cat_counts.columns = ['Category', 'Count']
                        fig2 = px.bar(
                            cat_counts, x='Category', y='Count',
                            title="Orders by Product Category",
                            color='Count',
                            color_continuous_scale='Viridis'
                        )
                        fig2.update_layout(
                            template="plotly_white",
                            xaxis_tickangle=-45,
                            font=dict(family="Inter, sans-serif"),
                            showlegend=False
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with col2:
                        fig_pie = px.pie(
                            cat_counts, values='Count', names='Category',
                            title="Category Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif")
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                if 'Weather' in df.columns and 'Traffic' in df.columns:
                    fig_heatmap = px.density_heatmap(
                        df, x='Weather', y='Traffic',
                        title="Weather vs Traffic Conditions Heatmap",
                        color_continuous_scale='Purples',
                        marginal_x="box",
                        marginal_y="box"
                    )
                    fig_heatmap.update_layout(
                        template="plotly_white",
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Tab 3: Time Analysis
            with viz_tab3:
                st.markdown("#### Time-Based Analysis")
                
                if 'Order_DateTime' in df.columns:
                    try:
                        df['Order_DateTime'] = pd.to_datetime(df['Order_DateTime'])
                        ts = df.set_index('Order_DateTime').resample('D').size().reset_index(name='Orders')
                        
                        fig4 = px.line(
                            ts, x='Order_DateTime', y='Orders',
                            title="Daily Orders Over Time",
                            markers=True
                        )
                        fig4.update_traces(
                            line_color='#667eea',
                            marker=dict(size=6, color='#764ba2', line=dict(width=2, color='white'))
                        )
                        fig4.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Date",
                            yaxis_title="Number of Orders",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    except Exception:
                        st.info("Could not parse 'Order_DateTime' column.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Hour' in df.columns:
                        hour_counts = df['Hour'].value_counts().sort_index().reset_index()
                        hour_counts.columns = ['Hour', 'Count']
                        fig5 = px.bar(
                            hour_counts, x='Hour', y='Count',
                            title="Orders by Hour of Day",
                            color='Count',
                            color_continuous_scale=['#667eea', '#764ba2', '#f093fb']
                        )
                        fig5.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Hour (24-hour format)",
                            yaxis_title="Number of Orders",
                            showlegend=False
                        )
                        st.plotly_chart(fig5, use_container_width=True)
                
                with col2:
                    if 'DayOfWeek' in df.columns:
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = df['DayOfWeek'].value_counts().reindex(day_order).reset_index()
                        day_counts.columns = ['Day', 'Count']
                        
                        fig_day = px.bar(
                            day_counts, x='Day', y='Count',
                            title="Orders by Day of Week",
                            color='Count',
                            color_continuous_scale='Sunset'
                        )
                        fig_day.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            xaxis_title="Day of Week",
                            yaxis_title="Number of Orders",
                            showlegend=False
                        )
                        st.plotly_chart(fig_day, use_container_width=True)
                
                if 'Month' in df.columns:
                    month_counts = df['Month'].value_counts().sort_index().reset_index()
                    month_counts.columns = ['Month', 'Count']
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month_counts['Month_Name'] = month_counts['Month'].apply(
                        lambda x: month_names[int(x)-1] if 1 <= x <= 12 else str(x)
                    )
                    
                    fig_month = px.line(
                        month_counts, x='Month_Name', y='Count',
                        title="Orders by Month",
                        markers=True
                    )
                    fig_month.update_traces(
                        line_color='#4facfe',
                        marker=dict(size=10, color='#667eea', line=dict(width=2, color='white'))
                    )
                    fig_month.update_layout(
                        template="plotly_white",
                        font=dict(family="Inter, sans-serif"),
                        xaxis_title="Month",
                        yaxis_title="Number of Orders"
                    )
                    st.plotly_chart(fig_month, use_container_width=True)
            
            # Tab 4: Geographic Analysis
            with viz_tab4:
                st.markdown("#### Geographic Analysis")
                
                if all(col in df.columns for col in ['Store_Latitude', 'Store_Longitude']):
                    try:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Store Locations Heatmap")
                            map_df = df[['Store_Latitude', 'Store_Longitude']].dropna().rename(
                                columns={'Store_Latitude': 'lat', 'Store_Longitude': 'lon'}
                            )
                            if not map_df.empty:
                                st.map(map_df, zoom=10)
                        
                        with col2:
                            if 'Delivery_Time' in df.columns:
                                st.markdown("##### Performance by Location")
                                fig_geo = px.scatter_mapbox(
                                    df, lat='Store_Latitude', lon='Store_Longitude',
                                    color='Delivery_Time',
                                    size='Delivery_Time',
                                    hover_data=['Distance'] if 'Distance' in df.columns else None,
                                    color_continuous_scale='Viridis',
                                    zoom=10,
                                    title="Delivery Time by Location"
                                )
                                fig_geo.update_layout(
                                    mapbox_style="open-street-map",
                                    font=dict(family="Inter, sans-serif"),
                                    height=400
                                )
                                st.plotly_chart(fig_geo, use_container_width=True)
                    except Exception:
                        st.info("Could not render geographic visualizations.")
                
                if 'Area' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        area_counts = df['Area'].value_counts().reset_index()
                        area_counts.columns = ['Area', 'Count']
                        fig_area = px.bar(
                            area_counts, x='Area', y='Count',
                            title="Orders by Area Type",
                            color='Area',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_area.update_layout(
                            template="plotly_white",
                            font=dict(family="Inter, sans-serif"),
                            showlegend=False
                        )
                        st.plotly_chart(fig_area, use_container_width=True)
                    
                    with col2:
                        if 'Delivery_Time' in df.columns:
                            fig_area_time = px.box(
                                df, x='Area', y='Delivery_Time',
                                title="Delivery Time by Area Type",
                                color='Area',
                                color_discrete_sequence=px.colors.qualitative.Safe
                            )
                            fig_area_time.update_layout(
                                template="plotly_white",
                                font=dict(family="Inter, sans-serif"),
                                yaxis_title="Delivery Time (hours)"
                            )
                            st.plotly_chart(fig_area_time, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Correlation Matrix
            st.markdown("### 🔗 Feature Correlation Matrix")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
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
                fig6.update_layout(
                    title="Correlation Heatmap (Numeric Features)",
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif"),
                    width=800,
                    height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False
                )
                st.plotly_chart(fig6, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Data Quality Report
            st.markdown("### 🔍 Data Quality Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #667eea;'>Missing Values</h4>
                    </div>
                """, unsafe_allow_html=True)
                missing = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Feature': missing.index,
                    'Missing Count': missing.values,
                    'Percentage': (missing.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values detected")
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #667eea;'>Duplicate Records</h4>
                    </div>
                """, unsafe_allow_html=True)
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"Found {duplicates} duplicate records ({(duplicates/len(df)*100):.2f}%)")
                else:
                    st.success("No duplicate records detected")
            
            with col3:
                st.markdown("""
                    <div class='metric-card'>
                        <h4 style='color: #667eea;'>Data Types</h4>
                    </div>
                """, unsafe_allow_html=True)
                dtype_counts = df.dtypes.value_counts()
                dtype_df = pd.DataFrame({
                    'Type': dtype_counts.index.astype(str),
                    'Count': dtype_counts.values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Summary Statistics
            st.markdown("### 📋 Summary Statistics")
            if len(df.select_dtypes(include=['number']).columns) > 0:
                summary_stats = df.describe().round(2)
                st.dataframe(summary_stats, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Processed Data",
                    data=csv,
                    file_name=f"processed_delivery_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure the CSV file is properly formatted.")

# About Page
def about_page():
    st.markdown("""
        <h1 style='text-align: center; color: #1a202c; font-weight: 700; margin-bottom: 10px;'>
            About This App
        </h1>
        <p style='text-align: center; color: #718096; font-size: 16px; margin-bottom: 30px;'>
            Powered by LightGBM with Optuna optimization for accurate delivery predictions
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 16px;'>Technology Stack</h3>
                <ul style='color: #4a5568; font-size: 15px; line-height: 2;'>
                    <li><strong>Model:</strong> LightGBM with Optuna hyperparameter tuning</li>
                    <li><strong>Framework:</strong> Streamlit for interactive UI</li>
                    <li><strong>Visualizations:</strong> Plotly for dynamic charts</li>
                    <li><strong>Data Processing:</strong> Pandas & NumPy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 16px;'>Key Features</h3>
                <ul style='color: #4a5568; font-size: 15px; line-height: 2;'>
                    <li>Real-time delivery time predictions</li>
                    <li>Batch processing for multiple orders</li>
                    <li>Interactive data analysis dashboard</li>
                    <li>Geographic distance calculations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card' style='margin-top: 20px;'>
            <h3 style='color: #667eea; margin-bottom: 16px;'>Model Features</h3>
            <p style='color: #4a5568; font-size: 15px; line-height: 1.8;'>
                The prediction model considers multiple factors including geographic distance (calculated using 
                Haversine formula), agent demographics and ratings, temporal patterns (hour, day, month), 
                weather conditions, traffic levels, vehicle types, area characteristics, and product categories 
                to deliver accurate delivery time estimates.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main
def main():
    model, metrics = load_model_and_metrics()
    
    st.sidebar.markdown("<h2 style='color: white; margin-bottom: 6px;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Prediction", "Data Analysis", "About"])
    
    default_metrics = {'r2':0,'rmse':0,'mae':0}
    
    if page == "Home":
        home_page(metrics if metrics else default_metrics)
    elif page == "Prediction":
        prediction_page(model, metrics if metrics else default_metrics)
    elif page == "Data Analysis":
        user_data_analysis_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":

    main()
