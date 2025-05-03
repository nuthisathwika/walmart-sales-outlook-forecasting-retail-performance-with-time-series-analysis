import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(layout="wide", page_title="Walmart Sales Analysis")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Walmart Sales Analysis Dashboard")
st.markdown("""
This interactive dashboard analyzes Walmart store sales data to uncover patterns, 
predict future sales, and understand the impact of various factors on weekly sales.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Feature Engineering", 
                           "Modeling", "Time Series Analysis"])

# Load data function with caching
@st.cache_data
def load_data():
    df_store = pd.read_csv('stores.csv')
    df_features = pd.read_csv('features.csv')
    df_train = pd.read_csv('train.csv')
    
    # Merge datasets
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    
    # Clean data
    df.drop(['IsHoliday_y'], axis=1, inplace=True)
    df.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)
    df = df.loc[df['Weekly_Sales'] > 0]
    
    # Add holiday flags
    df["Date"] = pd.to_datetime(df["Date"])
    df['Super_Bowl'] = df['Date'].isin([datetime(2010,2,12), datetime(2011,2,11), datetime(2012,2,10)]).astype(int)
    df['Labor_Day'] = df['Date'].isin([datetime(2010,9,10), datetime(2011,9,9), datetime(2012,9,7)]).astype(int)
    df['Thanksgiving'] = df['Date'].isin([datetime(2010,11,26), datetime(2011,11,25)]).astype(int)
    df['Christmas'] = df['Date'].isin([datetime(2010,12,31), datetime(2011,12,30)]).astype(int)
    
    # Extract temporal features
    df['week'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    return df

df = load_data()

# Data Overview Page
if page == "Data Overview":
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Data Dimensions")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write(f"Number of unique stores: {df['Store'].nunique()}")
    st.write(f"Number of unique departments: {df['Dept'].nunique()}")
    
    st.subheader("Store-Department Sales Matrix")
    store_dept_table = pd.pivot_table(df, index='Store', columns='Dept', values='Weekly_Sales', aggfunc=np.mean)
    st.dataframe(store_dept_table.style.background_gradient(cmap='Blues'))

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sales Distribution", "Holiday Impact", "Store Analysis", "Temporal Patterns"])
    
    with tab1:
        st.subheader("Weekly Sales Distribution")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df['Weekly_Sales'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Weekly Sales")
        st.pyplot(fig)
        
        st.subheader("Sales by Store Type")
        type_group = df.groupby('Type')['Weekly_Sales'].mean()
        fig, ax = plt.subplots(figsize=(8,6))
        type_group.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Holiday vs Non-Holiday Sales")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Specific Holiday Impact")
        holiday = st.selectbox("Select Holiday", ['Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas'])
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=holiday, y='Weekly_Sales', hue='Type', data=df, ax=ax)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Sales by Department")
        fig, ax = plt.subplots(figsize=(15,6))
        sns.barplot(x='Dept', y='Weekly_Sales', data=df, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Sales by Store")
        fig, ax = plt.subplots(figsize=(15,6))
        sns.barplot(x='Store', y='Weekly_Sales', data=df, ax=ax)
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Monthly Sales Pattern")
        monthly_sales = pd.pivot_table(df, values="Weekly_Sales", columns="year", index="month")
        fig, ax = plt.subplots(figsize=(12,6))
        monthly_sales.plot(ax=ax)
        st.pyplot(fig)
        
        st.subheader("Weekly Sales Pattern")
        weekly_sales = pd.pivot_table(df, values="Weekly_Sales", columns="year", index="week")
        fig, ax = plt.subplots(figsize=(12,6))
        weekly_sales.plot(ax=ax)
        st.pyplot(fig)

# Feature Engineering Page
elif page == "Feature Engineering":
    st.header("Feature Engineering")
    
    st.subheader("Correlation Analysis")
    df_encoded = df.copy()
    type_group = {'A':1, 'B':2, 'C':3}
    df_encoded['Type'] = df_encoded['Type'].replace(type_group)
    
    # Convert boolean columns to int
    bool_cols = ['Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas', 'IsHoliday']
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype(int)
    
    # Drop some columns for correlation analysis
    df_corr = df_encoded.drop(['Date', 'Temperature', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment'], axis=1)
    
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df_corr.corr().abs(), ax=ax)
    st.pyplot(fig)
    
    st.subheader("Feature Importance")
    # Prepare data for feature importance
    df_new = df_encoded.sort_values(by='Date', ascending=True)
    train_data = df_new[:int(0.7*(len(df_new)))]
    test_data = df_new[int(0.7*(len(df_new))):]
    
    target = "Weekly_Sales"
    used_cols = [c for c in df_new.columns.to_list() if c not in [target, 'Date']]
    
    X_train = train_data[used_cols]
    X_test = test_data[used_cols]
    y_train = train_data[target]
    y_test = test_data[target]
    
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                             max_features='sqrt', min_samples_split=10)
    scaler = RobustScaler()
    pipe = make_pipeline(scaler, rf)
    pipe.fit(X_train, y_train)
    
    # Plot feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title("Feature Importances")
    ax.bar(range(X_train.shape[1]), importances[indices],
           color="r", align="center")
    ax.set_xticks(range(X_train.shape[1]))
    ax.set_xticklabels(np.array(X_train.columns)[indices], rotation=90)
    ax.set_xlim([-1, X_train.shape[1]])
    st.pyplot(fig)

# Modeling Page
elif page == "Modeling":
    st.header("Predictive Modeling")
    
    # Prepare data
    df_encoded = df.copy()
    type_group = {'A':1, 'B':2, 'C':3}
    df_encoded['Type'] = df_encoded['Type'].replace(type_group)
    
    bool_cols = ['Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas', 'IsHoliday']
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype(int)
    
    # Drop some columns
    drop_cols = ['Temperature', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
    df_encoded = df_encoded.drop(drop_cols, axis=1)
    
    # Sort by date and split
    df_encoded = df_encoded.sort_values(by='Date', ascending=True)
    train_data = df_encoded[:int(0.7*(len(df_encoded)))]
    test_data = df_encoded[int(0.7*(len(df_encoded))):]
    
    target = "Weekly_Sales"
    used_cols = [c for c in df_encoded.columns.to_list() if c not in [target, 'Date']]
    
    X_train = train_data[used_cols]
    X_test = test_data[used_cols]
    y_train = train_data[target]
    y_test = test_data[target]
    
    # WMAE function
    def wmae_test(test, pred, X_test):
        weights = X_test['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)
        error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
        return error
    
    # Model training and evaluation
    st.subheader("Random Forest Model Performance")
    
    if st.button("Train Random Forest Model"):
        with st.spinner("Training model..."):
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                                     max_features='sqrt', min_samples_split=10)
            scaler = RobustScaler()
            pipe = make_pipeline(scaler, rf)
            pipe.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = pipe.predict(X_train)
            y_pred_test = pipe.predict(X_test)
            
            # Calculate WMAE
            train_wmae = wmae_test(y_train, y_pred_train, X_train)
            test_wmae = wmae_test(y_test, y_pred_test, X_test)
            
            # Display results
            st.success("Model training completed!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train WMAE", f"{train_wmae:.2f}")
            with col2:
                st.metric("Test WMAE", f"{test_wmae:.2f}")
            
            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(12,6))
            ax.scatter(y_test, y_pred_test, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Weekly Sales")
            st.pyplot(fig)

# Time Series Analysis Page
# In the Time Series Analysis page (replace the existing code):

elif page == "Time Series Analysis":
    st.header("Time Series Analysis")
    
    # Prepare weekly data - only use numeric columns that can be averaged
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Weekly_Sales' not in numeric_cols:
        numeric_cols.append('Weekly_Sales')
    
    # Create weekly resampled data
    try:
        df_week = df.set_index('Date')[numeric_cols].resample('W').mean()
    except Exception as e:
        st.error(f"Error resampling data: {str(e)}")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Decomposition", "Stationarity", "Forecasting"])
    
    with tab1:
        st.subheader("Time Series Decomposition")
        
        if df_week.empty or 'Weekly_Sales' not in df_week.columns:
            st.error("Weekly sales data not available for decomposition")
        else:
            try:
                # Perform decomposition
                result = seasonal_decompose(df_week['Weekly_Sales'].dropna(), 
                                          model='additive', 
                                          period=20)
                
                # Plot decomposition
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
                result.observed.plot(ax=ax1)
                ax1.set_ylabel('Observed')
                result.trend.plot(ax=ax2)
                ax2.set_ylabel('Trend')
                result.seasonal.plot(ax=ax3)
                ax3.set_ylabel('Seasonal')
                result.resid.plot(ax=ax4)
                ax4.set_ylabel('Residual')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in decomposition: {str(e)}")
    
    with tab2:
        st.subheader("Stationarity Analysis")
        
        if df_week.empty or 'Weekly_Sales' not in df_week.columns:
            st.error("Weekly sales data not available for analysis")
        else:
            # Original series
            st.markdown("*Original Time Series:*")
            fig, ax = plt.subplots(figsize=(12,4))
            df_week['Weekly_Sales'].plot(ax=ax)
            st.pyplot(fig)
            
            # Differenced series
            st.markdown("*Differenced Time Series (helps make data stationary):*")
            df_week_diff = df_week['Weekly_Sales'].diff().dropna()
            fig, ax = plt.subplots(figsize=(12,4))
            df_week_diff.plot(ax=ax)
            st.pyplot(fig)
            
            # ADF Test
            st.markdown("*Augmented Dickey-Fuller Test for Stationarity:*")
            try:
                adf_result = adfuller(df_week['Weekly_Sales'].dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"p-value: {adf_result[1]:.4f}")
                st.write("Critical Values:")
                for key, value in adf_result[4].items():
                    st.write(f"   {key}: {value:.4f}")
                
                if adf_result[1] <= 0.05:
                    st.success("The time series is stationary (p-value ≤ 0.05)")
                else:
                    st.warning("The time series is not stationary (p-value > 0.05)")
            except Exception as e:
                st.error(f"Error in stationarity test: {str(e)}")
    
    with tab3:
        st.subheader("Time Series Forecasting")
        
        if df_week.empty or 'Weekly_Sales' not in df_week.columns:
            st.error("Weekly sales data not available for forecasting")
        else:
            # Split data
            train_size = int(0.7 * len(df_week))
            train_data = df_week.iloc[:train_size]
            test_data = df_week.iloc[train_size:]
            
            # Model selection
            model_type = st.selectbox("Select Model", ["Exponential Smoothing", "ARIMA"])
            
            if st.button("Run Forecasting"):
                with st.spinner("Running time series forecast..."):
                    try:
                        if model_type == "Exponential Smoothing":
                            # Exponential Smoothing
                            model = ExponentialSmoothing(
                                train_data['Weekly_Sales'],
                                seasonal_periods=20,
                                seasonal='additive',
                                trend='additive'
                            ).fit()
                            forecast = model.forecast(len(test_data))
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(12,6))
                            ax.plot(train_data.index, train_data['Weekly_Sales'], label='Train')
                            ax.plot(test_data.index, test_data['Weekly_Sales'], label='Test')
                            ax.plot(test_data.index, forecast, label='ETS Forecast')
                            ax.legend()
                            ax.set_title("Exponential Smoothing Forecast")
                            st.pyplot(fig)
                        
                        else:  # ARIMA
                            st.subheader("ARIMA Model Parameters")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                p = st.slider("AR order (p)", 0, 3, 1)
                            with col2:
                                d = st.slider("Difference order (d)", 0, 2, 1)
                            with col3:
                                q = st.slider("MA order (q)", 0, 3, 1)
                            
                            # Fit ARIMA model
                            model = ARIMA(train_data['Weekly_Sales'].dropna(), order=(p,d,q))
                            model_fit = model.fit()
                            
                            # Forecast
                            forecast = model_fit.forecast(steps=len(test_data))
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(12,6))
                            ax.plot(train_data.index, train_data['Weekly_Sales'], label='Train')
                            ax.plot(test_data.index, test_data['Weekly_Sales'], label='Test')
                            ax.plot(test_data.index, forecast, label='ARIMA Forecast')
                            ax.legend()
                            ax.set_title(f"ARIMA({p},{d},{q}) Forecast")
                            st.pyplot(fig)
                            
                            # Show model summary
                            st.subheader("ARIMA Model Summary")
                            st.text(str(model_fit.summary()))
                    except Exception as e:
                        st.error(f"Error in forecasting: {str(e)}")