import streamlit as st
import joblib
import numpy as np


st.set_page_config(
    page_title="Machine Learning Approaches for Stock Market Trend Prediction",
    page_icon="📈",
    layout="centered"
)


def predict_price(Open, High, Low, Volume):
    test_data = np.array([[Open, High, Low, Volume]])
    trained_model = joblib.load("model.pkl")  
    predictions = trained_model.predict(test_data)
    return predictions


st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Project Details", "Stock Predictor"])



if page == "Home":
    st.image("static/home.gif")

    st.markdown(
        """
        ## 🚀 Machine Learning Approaches for Stock Market Trend Prediction 💰📊
        Navigate using the sidebar to view 📑 Project Details or 🤖 Predict Stock Prices.
        """
    )



elif page == "Project Details":
    st.image("static/details.gif")

    st.markdown("## 📊 Machine Learning Approaches for Stock Market Trend Prediction 📈💹")

    st.markdown("### 🔹 Abstract")
    st.write(
        "This project explores how **Machine Learning (ML)** techniques can be applied "
        "to forecast stock market movements. Financial data is inherently volatile and non-linear, "
        "making it a challenging problem. By leveraging regression models, ensemble learning, "
        "and deep learning techniques, the project aims to provide better insights into stock price trends."
    )

    st.markdown("### 🔹 Introduction")
    st.write(
        "The stock market serves as a crucial component of global financial systems, providing companies "
        "with opportunities to raise capital and investors with a platform to build wealth. "
        "Predicting stock market trends has always been a challenging research problem due to its high volatility "
        "and dependence on multiple factors like company performance, global events, and investor psychology."
    )

    st.markdown("### 🔹 Existing Methods")
    st.write(
        "Traditional methods such as *Technical Analysis* and *Fundamental Analysis* are widely used "
        "but often fail to capture complex, non-linear relationships. Statistical approaches like ARIMA and "
        "GARCH models perform well for time-series forecasting but struggle with unpredictable fluctuations."
    )

    st.markdown("### 🔹 Proposed Approach")
    st.write("Our approach integrates multiple machine learning models such as:")
    st.markdown(
        """
        - ✅ **Logistic Regression** for binary trend classification (Up/Down).  
        - ✅ **Random Forest** for robust, non-linear prediction using ensemble learning.  
        - ✅ **Decision Tree** for interpretable predictions based on historical features.  
        - ✅ **LSTM (Long Short-Term Memory)** for capturing temporal dependencies in stock sequences.  
        """
    )

    st.markdown("### 🔹 Advantages")
    st.markdown(
        """
        - 🚀 Improved accuracy through ML-based models.  
        - 🔍 Ability to handle large-scale, multi-dimensional data.  
        - 📊 Provides investors with actionable insights.  
        - 🤖 Adaptive to new trends and patterns in financial markets.  
        """
    )

    st.markdown("### 🔹 Applications")
    st.markdown(
        """
        - 💰 Stock price forecasting for investors.  
        - 📉 Risk management and portfolio optimization.  
        - 🏦 Automated trading strategies.  
        - 📊 Economic trend analysis for policymakers.  
        """
    )

    st.markdown("### 🔹 Conclusion")
    st.write(
        "Machine Learning approaches significantly enhance the accuracy of stock market predictions compared "
        "to traditional models. By combining regression, ensemble learning, and deep learning, "
        "this project demonstrates how intelligent systems can provide valuable insights for both individual "
        "and institutional investors."
    )




elif page == "Stock Predictor":
    st.image("static/output.gif")

    st.markdown("## 🤖 Stock Price Predictor 📈💹")

    with st.form("prediction_form"):
        open_val = st.number_input("🏦 Opening Price", step=0.01)
        high = st.number_input("📈 Highest Price", step=0.01)
        low = st.number_input("📉 Lowest Price", step=0.01)
        volume = st.number_input("📊 Volume", step=1, format="%d")

        submitted = st.form_submit_button("🔮 Predict Price")

        if submitted:
            predicts = predict_price(open_val, high, low, volume)
            value = str(round(predicts[0], 2))
            st.success(f"📢 Prediction Result: Predicted Stock Price = ${value} 💰")
