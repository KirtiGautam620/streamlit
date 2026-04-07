import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Linear Regression Visualizer")
st.write("Understand how Linear Regression learns!")

# Sidebar
st.sidebar.header("Controls")
m = st.sidebar.slider("Slope (m)", -10.0, 10.0, 1.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)
noise = st.sidebar.slider("Noise", 0.0, 20.0, 5.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 500, 100)
add_outliers = st.sidebar.checkbox("Add Outliers")

# Data
def generate_data(n=50, noise=5):
    X = np.linspace(0, 10, n)
    y = 2 * X + 3 + np.random.randn(n) * noise
    return X, y

X, y = generate_data(noise=noise)

# Add outliers BEFORE plotting
if add_outliers:
    y[:5] += np.random.randn(5) * 50

# Prediction
y_pred = m * X + b

# MSE
def compute_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

mse = compute_mse(y, y_pred)

# Gradient Descent
def gradient_descent(X, y, m, b, lr, epochs):
    n = len(X)
    history = []

    for _ in range(epochs):
        y_pred = m * X + b

        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

        history.append(compute_mse(y, y_pred))

    return m, b, history

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Line Fit", "Error", "Gradient Descent", "Loss Surface"
])

# ---------------- TAB 1 ----------------
with tab1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, y, label="Data")
    
    # Residuals
    for i in range(len(X)):
        ax1.plot([X[i], X[i]], [y[i], y_pred[i]], linestyle='dashed')

    ax1.plot(X, y_pred, color="red", label="Model")
    ax1.legend()

    st.pyplot(fig1)

# ---------------- TAB 2 ----------------
with tab2:
    st.write(f"### MSE: {mse:.2f}")
    st.info("Observe how error changes when you move the line.")

# ---------------- TAB 3 ----------------
with tab3:
    if st.button("Train Model"):
        m_final, b_final, history = gradient_descent(X, y, m, b, lr, epochs)

        st.write(f"Final m: {m_final:.2f}, b: {b_final:.2f}")

        fig2, ax2 = plt.subplots()
        ax2.plot(history)
        ax2.set_title("Loss vs Iterations")
        st.pyplot(fig2)

# ---------------- TAB 4 ----------------
with tab4:
    m_vals = np.linspace(-5, 5, 30)
    b_vals = np.linspace(-5, 5, 30)

    M, B = np.meshgrid(m_vals, b_vals)
    Z = np.zeros_like(M)

    for i in range(len(m_vals)):
        for j in range(len(b_vals)):
            y_pred_temp = M[i, j] * X + B[i, j]
            Z[i, j] = compute_mse(y, y_pred_temp)

    surface = go.Figure(data=[go.Surface(z=Z, x=M, y=B)])
    st.plotly_chart(surface)

# Info
st.success("Try changing slope/intercept and observe how MSE reacts!")