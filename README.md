# Portfolio Risk Intelligence

Predicting return direction and VaR breaches using Machine Learning.

## Stocks
AAPL, MSFT, JPM, XOM — Period: 2015 to 2023

## Tasks
- Task 1: Predict if portfolio return goes UP or DOWN
- Task 2: Predict if portfolio loss exceeds VaR threshold

## Algorithms
- Logistic Regression
- Decision Tree
- Random Forest

## How to Run

Install dependencies:
pip install -r requirements.txt

Run Notebook:
jupyter notebook Portfolio_Risk_Intelligence.ipynb

Run Dashboard:
streamlit run app.py

## 🚀 Live Demo

🔗 [Click here to view the app](https://portfolio-risk-intelligence-tqm3xpsckulyuhuxscogq4.streamlit.app/)

## Key Findings
- Risk is more predictable than returns
- 50% AUC for direction = markets are efficient
- Volatility clustering is strongest signal for breach prediction
