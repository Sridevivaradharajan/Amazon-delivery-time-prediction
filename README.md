# Amazon Delivery Time Prediction
## Project Overview

This project aims to predict Amazon delivery times using machine learning techniques. By analyzing various features such as agent attributes, geographic coordinates, weather conditions, and traffic data, the model provides accurate estimated delivery times (ETAs). The goal is to enhance operational efficiency, optimize resource allocation, and improve customer satisfaction.

---

## Key Components

* **Data Analysis & Preprocessing**: The dataset includes features like agent age, rating, store and drop coordinates, order details, weather, traffic, and vehicle type. Preprocessing steps involve handling missing values, encoding categorical variables, and scaling numerical features.

* **Modeling**: Various machine learning models, including Linear Regression, Ridge Regression, Gradient Boosting, XGBoost, LightGBM, and Random Forest, are trained and evaluated to predict delivery times.

* **Model Evaluation**: Performance metrics such as R² Score, RMSE, and MAE are used to assess model accuracy. Feature importance analysis helps identify key factors influencing delivery times.

* **Deployment**: The trained model is saved as a pickle file (`Model.pkl`) and can be integrated into applications for real-time delivery time prediction.

---

## Repository Structure

* `Amazon.ipynb`: Jupyter notebook containing data analysis, preprocessing, and model training code.

* `Amazon_app.py`: Python script for deploying the trained model and making predictions.

* `Model.pkl`: Serialized machine learning model for inference.

* `amazon_delivery.csv`: Dataset used for training and evaluation.

* `requirements.txt`: List of Python dependencies required to run the project.

---

## Usage Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Sridevivaradharajan/Amazon-delivery-time-prediction.git
   cd Amazon-delivery-time-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python Amazon_app.py
   ```
---

## 📈 Future Enhancements

* **Real-Time Data Integration**: Incorporate real-time GPS, weather, and traffic data to provide dynamic delivery time predictions.

* **Deep Learning Models**: Explore the use of deep learning techniques for improved accuracy and scalability.

* **User Interface**: Develop a web-based interface for users to input order details and receive estimated delivery times.

---
