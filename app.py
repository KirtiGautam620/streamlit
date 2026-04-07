import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Linear Regression Visualizer")
st.write("Understand how Linear Regression learns!")


# Sidebar
st.sidebar.header("Controls")
dataset_mode = st.sidebar.selectbox("Dataset", ["Noisy", "Clean", "Outliers Heavy"])
noise = st.sidebar.slider("Noise Level", 0.0, 20.0, 5.0 if dataset_mode != "Clean" else 0.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
epochs = st.sidebar.slider("Epochs", 50, 500, 100)

st.sidebar.subheader("Manual Fit")
m = st.sidebar.slider("Slope", -10.0, 10.0, 1.0)
b = st.sidebar.slider("Intercept", -10.0, 10.0, 0.0)

st.sidebar.subheader("GD Init")
m_init = st.sidebar.slider("Init m", -5.0, 5.0, 1.0)
b_init = st.sidebar.slider("Init b", -5.0, 5.0, 0.0)
if st.sidebar.button("Reset"):
    st.rerun()
add_outliers = st.sidebar.checkbox("Add Outliers", dataset_mode == "Outliers Heavy")

# Data
def generate_data(n=50, noise=5, mode="Noisy"):
    X = np.linspace(0, 10, n)
    y = 2 * X + 3 + np.random.randn(n) * noise
    if mode == "Clean":
        y = 2 * X + 3 + np.random.randn(n) * 0.5
    elif mode == "Outliers Heavy":
        y = 2 * X + 3 + np.random.randn(n) * noise
        y[:8] += np.random.randn(8) * 30  # heavier outliers
    return X, y

X, y = generate_data(noise=noise, mode=dataset_mode)

# Add outliers BEFORE plotting
if add_outliers:
    y[:5] += np.random.randn(5) * 50

# Prediction
y_pred = m * X + b

# Move def here (was missing early call)
def compute_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

mse = compute_mse(y, y_pred)

# KPIs for emphasis (now after vars defined)
true_m, true_b = 2.0, 3.0  # Ground truth
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MSE", f"{mse:.2f}")
with col2:
    st.metric("Slope m", f"{m:.2f}")
with col3:
    st.metric("Intercept b", f"{b:.2f}")
with col4:
    st.metric("Points", len(X))


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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Line Fit", "Error", "Gradient Descent", "Loss Surface", 
    "Learning Rate", "Noise & Outliers"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.write("Line Fitting: y = mx + b. Residuals show fit quality.")
    
    residuals = y - y_pred
    mean_err = np.mean(np.abs(residuals))
    
    fig1, ax1 = plt.subplots(figsize=(10,6))
    scatter = ax1.scatter(X, y, c=np.abs(residuals), cmap='Reds', s=60, label="Data")
    
    for i in range(len(X)):
        res = residuals[i]
        color = 'red' if abs(res) > mean_err else 'orange'
        lw = 3 + abs(res)/10
        ax1.plot([X[i], X[i]], [y[i], y_pred[i]], color=color, linestyle='-', linewidth=lw, alpha=0.7)
    
    ax1.plot(X, y_pred, color="blue", linewidth=3, label="Model")
    ax1.legend()
    plt.colorbar(scatter, ax=ax1)
    st.pyplot(fig1)

# ---------------- TAB 2 ----------------
with tab2:
    st.write("MSE = mean((y - y_pred)^2)")
    st.write(f"Current MSE: {mse:.2f}")
    
    residuals = y - y_pred
    col_a, col_b = st.columns(2)
    with col_a:
        fig_res, ax_res = plt.subplots()
        ax_res.scatter(X, residuals, c=residuals, cmap='RdBu', alpha=0.6)
        ax_res.axhline(0, color='black', ls='--')
        ax_res.set_xlabel('X')
        ax_res.set_ylabel('Residuals')
        st.pyplot(fig_res)
    
    with col_b:
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(residuals, bins=15, edgecolor='black', alpha=0.7)
        ax_hist.axvline(np.mean(residuals), color='red', ls='--')
        ax_hist.legend()
        st.pyplot(fig_hist)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown("### **Concept: Gradient Descent**")
    st.info("**θ = θ - α ∇J** - Iterative update towards minimum.")
    
    m_gd, b_gd = m_init, b_init
    history_mse, history_m, history_b = [], [], []
    
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        y_pred_gd = m_gd * X + b_gd
        curr_mse = compute_mse(y, y_pred_gd)
        
        dm = (-2/len(X)) * np.sum(X * (y - y_pred_gd))
        db = (-2/len(X)) * np.sum(y - y_pred_gd)
        
        m_gd -= lr * dm
        b_gd -= lr * db
        
        history_mse.append(curr_mse)
        history_m.append(m_gd)
        history_b.append(b_gd)
        
        progress_bar.progress((epoch + 1) / epochs)
    
    col31, col32 = st.columns(2)
    with col31:
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history_mse)
        ax_loss.set_title("Loss Convergence")
        ax_loss.set_xlabel("Epoch")
        st.pyplot(fig_loss)
    
    with col32:
        fig_params, ax_params = plt.subplots()
        ax_params.plot(history_m, label="m")
        ax_params.plot(history_b, label="b")
        ax_params.legend()
        ax_params.set_title("Parameter Evolution")
        st.pyplot(fig_params)
    
    st.success(f"**Final**: m={m_gd:.3f}, b={b_gd:.3f} | MSE={history_mse[-1]:.2f} | **Takeaway**: Watch convergence speed by LR!")

# ---------------- TAB 4 ----------------
with tab4:
    st.markdown("### **Concept: Loss Landscape**")
    st.info("J(m,b) surface shows optimization geometry.")
    
    m_vals = np.linspace(-5, 5, 30)
    b_vals = np.linspace(-5, 5, 30)
    M, B = np.meshgrid(m_vals, b_vals)
    Z = np.zeros_like(M)
    
    for i in range(len(m_vals)):
        for j in range(len(b_vals)):
            y_pred_temp = M[i, j] * X + B[i, j]
            Z[i, j] = compute_mse(y, y_pred_temp)
    
    fig_surface = go.Figure()
    fig_surface.add_trace(go.Surface(z=Z, x=M, y=B, colorscale='Viridis', showscale=True))
    
    # Contour projection
    fig_surface.add_trace(go.Contour(z=Z, x=m_vals, y=b_vals, colorscale='Viridis', 
                                    contours=dict(showlines=False), connectgaps=True, showscale=False))
    
    # Current position marker
    fig_surface.add_trace(go.Scatter3d(x=[m], y=[b], z=[mse], mode='markers+text', 
                                     marker=dict(size=8, color='red'), text=['Current'], 
                                     textposition="top center"))
    
    fig_surface.update_layout(title="Loss Surface J(m,b)", scene=dict(zaxis_title="MSE"))
    st.plotly_chart(fig_surface)
    
    st.success("**Takeaway**: Smooth bowl = easy optimization. Current dot shows your (m,b)!")

# ---------------- TAB 5 ----------------
with tab5:
    st.markdown("### **⚡ Learning Rate Experiments**")
    st.info("Small α=slow, large=unstable/diverge.")
    
    lr_rates = [0.001, 0.01, 0.1, lr]  # include current
    fig_lr = plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for test_lr in lr_rates:
        test_history = []
        tm, tb = m_init, b_init
        for e in range(epochs):
            ty_pred = tm * X + tb
            tdm = (-2/len(X)) * np.sum(X * (y - ty_pred))
            tdb = (-2/len(X)) * np.sum(y - ty_pred)
            tm -= test_lr * tdm
            tb -= test_lr * tdb
            test_history.append(compute_mse(y, ty_pred))
        plt.plot(test_history, label=f"α={test_lr}")
    plt.title("Loss Convergence Comparison")
    # plt.yscale('log')  # Removed to fix overflow
    plt.legend()
    plt.xlabel("Epoch")
    
    plt.subplot(1,2,2)
    final_mses = []
    for test_lr in lr_rates:
        tm, tb = m_init, b_init
        for e in range(epochs):
            ty_pred = tm * X + tb
            tdm = (-2/len(X)) * np.sum(X * (y - ty_pred))
            tdb = (-2/len(X)) * np.sum(y - ty_pred)
            tm -= test_lr * tdm
            tb -= test_lr * tdb
        final_mses.append(compute_mse(y, tm * X + tb))
    plt.bar([str(r) for r in lr_rates], final_mses)
    plt.title("Final MSE by LR")
    plt.xticks(rotation=45)
    st.pyplot(fig_lr)
    st.success("**Takeaway**: Balance speed (fast drop) & stability (no diverge)!")

# ---------------- TAB 6 ----------------
with tab6:
    st.markdown("### **🔊 Noise & Robustness**")
    st.info("Outliers heavily influence due to squared loss.")
    
    X_clean, y_clean = generate_data(noise=0.1, mode="Clean")
    X_noisy, y_noisy = generate_data(noise=noise, mode="Noisy")
    X_out, y_out = generate_data(noise=noise, mode="Outliers Heavy")
    
    col61, col62, col63 = st.columns(3)
    
    with col61:
        st.subheader("Clean Data")
        fig_clean = plt.figure(figsize=(5,4))
        plt.scatter(X_clean, y_clean, alpha=0.6)
        plt.plot(X_clean, true_m * X_clean + true_b, 'g--', lw=2, label="True Line")
        plt.title("Clean (low noise)")
        plt.legend()
        st.pyplot(fig_clean)
    
    with col62:
        st.subheader("Noisy")
        fig_noisy = plt.figure(figsize=(5,4))
        plt.scatter(X_noisy, y_noisy, alpha=0.6, color='orange')
        plt.plot(X_noisy, true_m * X_noisy + true_b, 'g--', lw=2)
        plt.title(f"Noisy (σ={noise:.1f})")
        st.pyplot(fig_noisy)
    
    with col63:
        st.subheader("Outliers")
        fig_out = plt.figure(figsize=(5,4))
        plt.scatter(X_out, y_out, alpha=0.6, color='red')
        plt.plot(X_out, true_m * X_out + true_b, 'g--', lw=2)
        plt.title("Outliers Heavy")
        st.pyplot(fig_out)
    
    st.success("**Takeaway**: Outliers shift line! Consider robust regression.")



