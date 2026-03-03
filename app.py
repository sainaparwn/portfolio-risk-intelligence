import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.tree import plot_tree
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Intelligence",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────
# LOAD MODELS AND DATA
# ─────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        'lr_return' : joblib.load('portfolio_model/lr_return.pkl'),
        'lr_var'    : joblib.load('portfolio_model/lr_var.pkl'),
        'dt_return' : joblib.load('portfolio_model/dt_return.pkl'),
        'dt_var'    : joblib.load('portfolio_model/dt_var.pkl'),
        'rf_return' : joblib.load('portfolio_model/rf_return.pkl'),
        'rf_var'    : joblib.load('portfolio_model/rf_var.pkl'),
        'scaler'    : joblib.load('portfolio_model/scaler.pkl'),
    }

@st.cache_data
def load_data():
    returns  = pd.read_csv('portfolio_model/returns.csv',
                           index_col=0, parse_dates=True).squeeze()
    var      = pd.read_csv('portfolio_model/var.csv',
                           index_col=0, parse_dates=True).squeeze()
    y_ret    = pd.read_csv('portfolio_model/y_return.csv',
                           index_col=0, parse_dates=True).squeeze()
    y_var    = pd.read_csv('portfolio_model/y_var.csv',
                           index_col=0, parse_dates=True).squeeze()
    features = pd.read_csv('portfolio_model/features.csv',
                           index_col=0, parse_dates=True)
    with open('portfolio_model/results.json') as f:
        results = json.load(f)
    return returns, var, y_ret, y_var, features, results

models                                    = load_models()
returns, var, y_ret, y_var, features, res = load_data()

# Test set
split          = int(len(features) * 0.80)
X_test         = features.iloc[split:]
y_ret_test     = y_ret.iloc[split:]
y_var_test     = y_var.iloc[split:]
X_test_sc      = models['scaler'].transform(X_test)

# All probabilities
prob = {
    'lr_ret' : models['lr_return'].predict_proba(X_test_sc)[:,1],
    'lr_var' : models['lr_var'].predict_proba(X_test_sc)[:,1],
    'dt_ret' : models['dt_return'].predict_proba(X_test_sc)[:,1],
    'dt_var' : models['dt_var'].predict_proba(X_test_sc)[:,1],
    'rf_ret' : models['rf_return'].predict_proba(X_test_sc)[:,1],
    'rf_var' : models['rf_var'].predict_proba(X_test_sc)[:,1],
}
pred = {
    'lr_ret' : models['lr_return'].predict(X_test_sc),
    'lr_var' : models['lr_var'].predict(X_test_sc),
    'dt_ret' : models['dt_return'].predict(X_test_sc),
    'dt_var' : models['dt_var'].predict(X_test_sc),
    'rf_ret' : models['rf_return'].predict(X_test_sc),
    'rf_var' : models['rf_var'].predict(X_test_sc),
}

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
st.sidebar.title("📈 Portfolio Risk")
st.sidebar.markdown("---")

page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📊 Data Overview",
    "📋 Model Results",
    "🔮 Predict",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Stocks:** AAPL, MSFT, JPM, XOM")
st.sidebar.markdown("**Period:** 2015 — 2023")
st.sidebar.markdown("**Models:** LR | DT | RF")

# ═════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════
if page == "🏠 Home":

    st.title("📈 Portfolio Risk Intelligence")
    st.markdown("Predicting **Return Direction** and **VaR Breaches** using Machine Learning")
    st.markdown("---")

    # 4 top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading Days",
              f"{res['dataset']['total_days']:,}")
    c2.metric("VaR Breach Days",
              f"{res['dataset']['breach_days']}",
              f"{res['dataset']['breach_rate']:.1%} of all days")
    c3.metric("Avg VaR Threshold",
              f"{res['dataset']['avg_var']:.2%}")
    c4.metric("Return Kurtosis",
              f"{res['dataset']['kurtosis']:.2f}",
              "Fat tails (normal = 3.0)")

    st.markdown("---")

    # Two tasks explained
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Task 1 — Return Direction")
        st.info("""
        **Question:** Will portfolio go UP or DOWN tomorrow?

        **Why:** Helps decide position sizing

        **Challenge:** Markets are efficient.
        Direction is near random. AUC ≈ 0.50 is expected.
        """)

    with c2:
        st.subheader("⚠️ Task 2 — VaR Breach")
        st.warning("""
        **Question:** Will loss exceed danger threshold?

        **Why:** Risk managers need crash warnings

        **Challenge:** Only 5% of days are breaches.
        Class imbalance makes this hard.
        """)

    st.markdown("---")

    # Three models
    st.subheader("🤖 Three Models Compared")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("""
        **📐 Logistic Regression**

        Simple linear model.
        Learns weights for each feature.
        Fast and interpretable.
        Used as our baseline.
        """)
    with c2:
        st.warning("""
        **🌳 Decision Tree**

        Learns IF-THEN rules.
        IF vol high AND momentum negative
        → predict BREACH.
        Captures nonlinear patterns.
        """)
    with c3:
        st.success("""
        **🌲 Random Forest**

        100 Decision Trees combined.
        Averages all predictions.
        Errors cancel out.
        Most robust model.
        """)

# ═════════════════════════════════════
# PAGE 2 — DATA OVERVIEW
# ═════════════════════════════════════
elif page == "📊 Data Overview":

    st.title("📊 Data Overview")
    st.markdown("---")

    # Chart 1: Returns + VaR
    st.subheader("Portfolio Returns with VaR Threshold")
    st.caption("Blue = daily returns | Orange = VaR threshold | Red dots = breach days")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(returns.index, returns,
            color='steelblue', linewidth=0.8,
            alpha=0.8, label='Daily Return')
    ax.plot(var.index, var,
            color='orange', linewidth=1.5,
            label='Rolling VaR 5%')

    # align index
    ret_aligned = returns.loc[y_var.index]
    breach_pts  = ret_aligned[y_var == 1]
    ax.scatter(breach_pts.index, breach_pts.values,
               color='red', s=20, zorder=5,
               label=f'VaR Breach ({len(breach_pts)} days)')
    ax.set_ylabel('Daily Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Chart 2: Distribution
    st.subheader("Return Distribution — Fat Tails")
    st.caption("Real returns have more extreme events than normal distribution predicts")

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.hist(returns, bins=80, density=True,
             color='steelblue', alpha=0.7,
             label='Actual Returns')
    x = np.linspace(returns.min(), returns.max(), 200)
    ax2.plot(x,
             stats.norm.pdf(x, returns.mean(), returns.std()),
             color='red', linewidth=2,
             label='Normal Distribution')
    ax2.axvline(var.mean(), color='orange',
                linewidth=2, linestyle='--',
                label='Average VaR')
    kurt = stats.kurtosis(returns)
    ax2.text(0.01, 0.95,
             f'Kurtosis = {kurt:.2f}  (normal = 3.0)',
             transform=ax2.transAxes,
             color='red', fontsize=11,
             verticalalignment='top')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # Chart 3: Breaches per year
    st.subheader("VaR Breaches Per Year")

    breach_yr = y_var.groupby(y_var.index.year).sum()
    colors_yr = ['red' if v > 12.6 else 'steelblue'
                 for v in breach_yr.values]

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(breach_yr.index, breach_yr.values,
            color=colors_yr, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax3.axhline(y=12.6, color='orange',
                linestyle='--', linewidth=2,
                label='Expected ~13/year')
    ax3.set_ylabel('Number of Breaches')
    ax3.set_title('VaR Breaches Per Year  '
                  '(Red = above average, Blue = below average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.info("2020 should be highest (COVID crash). "
            "2017 should be lowest (calmest year).")

# ═════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═════════════════════════════════════
elif page == "📋 Model Results":

    st.title("📋 Model Results")
    st.markdown("---")

    # Metrics table
    st.subheader("Performance Metrics")
    metric_names = ['Accuracy', 'Precision',
                    'Recall', 'F1', 'AUC']

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Task 1 — Return Direction")
        st.dataframe(pd.DataFrame({
            'Metric': metric_names,
            'Log Reg': [round(v,4) for v in res['lr_t1']],
            'Dec Tree': [round(v,4) for v in res['dt_t1']],
            'Rnd Forest': [round(v,4) for v in res['rf_t1']],
        }), hide_index=True, use_container_width=True)
        st.info("All models near 0.5 AUC → markets are efficient")

    with c2:
        st.markdown("#### Task 2 — VaR Breach")
        st.dataframe(pd.DataFrame({
            'Metric': metric_names,
            'Log Reg': [round(v,4) for v in res['lr_t2']],
            'Dec Tree': [round(v,4) for v in res['dt_t2']],
            'Rnd Forest': [round(v,4) for v in res['rf_t2']],
        }), hide_index=True, use_container_width=True)
        st.warning("Use AUC and F1 — not accuracy — for imbalanced data")

    st.markdown("---")

    # ROC curves
    st.subheader("ROC Curves — All Models")
    st.caption("Closer to top-left corner = better model | Diagonal = random guessing")

    model_colors = {
        'Logistic Reg' : ('steelblue', 'lr_ret', 'lr_var'),
        'Decision Tree': ('orange',    'dt_ret', 'dt_var'),
        'Random Forest': ('green',     'rf_ret', 'rf_var'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mname, (color, k_ret, k_var) in model_colors.items():
        for ax, y_true, k in zip(
            axes,
            [y_ret_test, y_var_test],
            [k_ret, k_var]
        ):
            fpr, tpr, _ = roc_curve(y_true, prob[k])
            auc = roc_auc_score(y_true, prob[k])
            ax.plot(fpr, tpr, color=color,
                    linewidth=2,
                    label=f'{mname} ({auc:.3f})')

    for ax, title in zip(axes, [
        'Task 1 — Return Direction',
        'Task 2 — VaR Breach'
    ]):
        ax.plot([0,1],[0,1],'--',
                color='gray', label='Random (0.500)')
        ax.set_title(title)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Breach detection
    st.subheader("VaR Breach Detection — How Many Did Each Model Catch?")

    total      = res['dataset']['total_breaches']
    caught_lr  = res['dataset']['caught_lr']
    caught_dt  = res['dataset']['caught_dt']
    caught_rf  = res['dataset']['caught_rf']

    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression",
              f"{caught_lr} / {total} caught",
              f"{caught_lr/total:.0%} recall")
    c2.metric("Decision Tree",
              f"{caught_dt} / {total} caught",
              f"{caught_dt/total:.0%} recall")
    c3.metric("Random Forest",
              f"{caught_rf} / {total} caught",
              f"{caught_rf/total:.0%} recall")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.bar(
        ['Logistic\nRegression',
         'Decision\nTree',
         'Random\nForest'],
        [caught_lr, caught_dt, caught_rf],
        color=['steelblue', 'orange', 'green'],
        alpha=0.85, edgecolor='black',
        linewidth=0.5
    )
    ax2.axhline(y=total, color='red',
                linestyle='--', linewidth=2,
                label=f'Total = {total}')
    for bar, val in zip(bars, [caught_lr, caught_dt, caught_rf]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f'{val}/{total}',
                 ha='center', fontweight='bold',
                 fontsize=12)
    ax2.set_ylabel('Breaches Caught')
    ax2.set_title('Breach Detection Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # Confusion matrices
    st.subheader("Confusion Matrices — Task 2 VaR Breach")

    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
    for ax, title, pk in zip(
        axes3,
        ['Logistic Regression',
         'Decision Tree',
         'Random Forest'],
        ['lr_var', 'dt_var', 'rf_var']
    ):
        cm = confusion_matrix(y_var_test, pred[pk])
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues', ax=ax,
                    xticklabels=['SAFE', 'BREACH'],
                    yticklabels=['SAFE', 'BREACH'])
        ax.set_title(title)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")

    # Feature importances
    st.subheader("Feature Importances — Task 2 VaR Breach")
    st.caption("Which features does each model rely on most?")

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4))
    for ax, model_key, title, color in zip(
        axes4,
        ['dt_var', 'rf_var'],
        ['Decision Tree', 'Random Forest'],
        ['orange', 'green']
    ):
        imp = pd.Series(
            models[model_key].feature_importances_,
            index=features.columns
        ).sort_values(ascending=True)
        imp.plot(kind='barh', ax=ax,
                 color=color, alpha=0.8)
        ax.set_title(f'{title} — Feature Importances')
        ax.set_xlabel('Importance Score')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ═════════════════════════════════════
# PAGE 4 — PREDICT
# ═════════════════════════════════════
elif page == "🔮 Predict":

    st.title("🔮 Live Prediction")
    st.markdown("Adjust market conditions and see what all 3 models predict.")
    st.markdown("---")

    # Preset scenarios
    st.subheader("Quick Scenarios")
    scenario = st.radio(
        "Pick a scenario or use sliders below:",
        ["🎛️ Custom",
         "🔴 Crisis (COVID-like)",
         "🟢 Calm Bull Market",
         "🟡 Mixed Signals"],
        horizontal=True
    )

    if scenario == "🔴 Crisis (COVID-like)":
        v = dict(l1=-4.0, l2=-3.0, l3=-2.0,
                 v5=6.0, v20=4.0,
                 m5=-2.0, m20=-1.0, mom=-12.0)
    elif scenario == "🟢 Calm Bull Market":
        v = dict(l1=0.5, l2=0.3, l3=0.2,
                 v5=0.4, v20=0.6,
                 m5=0.3, m20=0.2, mom=3.0)
    elif scenario == "🟡 Mixed Signals":
        v = dict(l1=-1.0, l2=0.8, l3=-0.5,
                 v5=2.5, v20=1.8,
                 m5=0.1, m20=0.2, mom=-2.0)
    else:
        v = dict(l1=0.0, l2=0.0, l3=0.0,
                 v5=1.0, v20=1.2,
                 m5=0.0, m20=0.0, mom=0.0)

    st.markdown("---")

    # Input sliders
    st.subheader("Market Conditions")
    c1, c2 = st.columns(2)

    with c1:
        l1  = st.slider("Yesterday Return (%)",
                        -10.0, 10.0, v['l1'], 0.1) / 100
        l2  = st.slider("2 Days Ago Return (%)",
                        -10.0, 10.0, v['l2'], 0.1) / 100
        l3  = st.slider("3 Days Ago Return (%)",
                        -10.0, 10.0, v['l3'], 0.1) / 100
        mom = st.slider("10-Day Momentum (%)",
                        -20.0, 20.0, v['mom'], 0.5) / 100

    with c2:
        v5  = st.slider("5-Day Volatility (%)",
                        0.1, 10.0, v['v5'], 0.1) / 100
        v20 = st.slider("20-Day Volatility (%)",
                        0.1, 8.0, v['v20'], 0.1) / 100
        m5  = st.slider("5-Day Mean Return (%)",
                        -5.0, 5.0, v['m5'], 0.1) / 100
        m20 = st.slider("20-Day Mean Return (%)",
                        -3.0, 3.0, v['m20'], 0.1) / 100

    # Build input
    inp = pd.DataFrame([[
        l1, l2, l3, v5, v20, m5, m20, mom
    ]], columns=features.columns)
    inp_sc = models['scaler'].transform(inp)

    # Predictions from all models
    p_lr_ret = models['lr_return'].predict_proba(inp_sc)[0][1]
    p_dt_ret = models['dt_return'].predict_proba(inp_sc)[0][1]
    p_rf_ret = models['rf_return'].predict_proba(inp_sc)[0][1]
    p_lr_var = models['lr_var'].predict_proba(inp_sc)[0][1]
    p_dt_var = models['dt_var'].predict_proba(inp_sc)[0][1]
    p_rf_var = models['rf_var'].predict_proba(inp_sc)[0][1]

    st.markdown("---")

    # Task 1 results
    st.subheader("Task 1 — Return Direction Predictions")
    c1, c2, c3 = st.columns(3)

    for col, name, p in zip(
        [c1, c2, c3],
        ['Logistic Regression',
         'Decision Tree',
         'Random Forest'],
        [p_lr_ret, p_dt_ret, p_rf_ret]
    ):
        with col:
            if p > 0.5:
                col.success(f"**{name}**\n\n📈 UP\n\nP(UP) = {p:.1%}")
            else:
                col.error(f"**{name}**\n\n📉 DOWN\n\nP(UP) = {p:.1%}")

    st.markdown("---")

    # Task 2 results
    st.subheader("Task 2 — VaR Breach Risk")
    c1, c2, c3 = st.columns(3)

    for col, name, p in zip(
        [c1, c2, c3],
        ['Logistic Regression',
         'Decision Tree',
         'Random Forest'],
        [p_lr_var, p_dt_var, p_rf_var]
    ):
        with col:
            if p > 0.5:
                col.error(
                    f"**{name}**\n\n"
                    f"🚨 HIGH RISK\n\n"
                    f"P(breach) = {p:.1%}"
                )
            elif p > 0.25:
                col.warning(
                    f"**{name}**\n\n"
                    f"⚠️ MEDIUM RISK\n\n"
                    f"P(breach) = {p:.1%}"
                )
            else:
                col.success(
                    f"**{name}**\n\n"
                    f"✅ LOW RISK\n\n"
                    f"P(breach) = {p:.1%}"
                )

    st.markdown("---")

    # Probability bar chart
    st.subheader("Probability Chart")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    mnames = ['Log Reg', 'Dec Tree', 'Rnd Forest']

    # Task 1
    probs_t1 = [p_lr_ret, p_dt_ret, p_rf_ret]
    axes[0].bar(mnames, probs_t1,
                color=['green' if p > 0.5 else 'red'
                       for p in probs_t1],
                alpha=0.8, edgecolor='black')
    axes[0].axhline(0.5, color='black',
                    linestyle='--', linewidth=1.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('P(UP) — Return Direction')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, p in enumerate(probs_t1):
        axes[0].text(i, p + 0.02, f'{p:.1%}',
                     ha='center', fontweight='bold')

    # Task 2
    probs_t2 = [p_lr_var, p_dt_var, p_rf_var]
    axes[1].bar(mnames, probs_t2,
                color=['red' if p > 0.5
                       else 'orange' if p > 0.25
                       else 'green'
                       for p in probs_t2],
                alpha=0.8, edgecolor='black')
    axes[1].axhline(0.5, color='black',
                    linestyle='--', linewidth=1.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('P(Breach) — VaR Breach Risk')
    axes[1].set_ylabel('Probability')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, p in enumerate(probs_t2):
        axes[1].text(i, p + 0.02, f'{p:.1%}',
                     ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Scenario tip
    st.markdown("---")
    if scenario == "🔴 Crisis (COVID-like)":
        st.error("Crisis conditions: high vol + negative momentum "
                 "→ expect high breach probability from all models")
    elif scenario == "🟢 Calm Bull Market":
        st.success("Calm conditions: low vol + positive momentum "
                   "→ expect low breach probability from all models")
    elif scenario == "🟡 Mixed Signals":
        st.warning("Mixed signals: models may disagree "
                   "→ shows how each algorithm handles uncertainty differently")
    else:
        st.info("Try the preset scenarios above "
                "or drag sliders to explore how models respond")