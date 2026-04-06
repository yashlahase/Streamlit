import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time

# Page config
st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Linear Regression Interactive Visualizer</h1>', unsafe_allow_html=True)

# Helper functions
@st.cache_data
def generate_data(n_points, noise_level, outliers, seed=42):
    """Generate synthetic dataset"""
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_points)
    true_slope, true_intercept = 2, 1
    y_true = true_slope * X + true_intercept
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    
    # Add outliers
    if outliers > 0:
        outlier_indices = np.random.choice(n_points, min(outliers, n_points//4), replace=False)
        y[outlier_indices] += np.random.normal(0, noise_level * 5, len(outlier_indices))
    
    return X, y, true_slope, true_intercept

def compute_mse(X, y, slope, intercept):
    """Compute Mean Squared Error"""
    y_pred = slope * X + intercept
    return np.mean((y - y_pred) ** 2)

def compute_gradients(X, y, slope, intercept):
    """Compute gradients for gradient descent"""
    n = len(X)
    y_pred = slope * X + intercept
    dw = -2/n * np.sum(X * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    return dw, db

def gradient_descent_step(X, y, slope, intercept, learning_rate):
    """Perform one step of gradient descent"""
    dw, db = compute_gradients(X, y, slope, intercept)
    new_slope = slope - learning_rate * dw
    new_intercept = intercept - learning_rate * db
    return new_slope, new_intercept

# Sidebar controls
st.sidebar.markdown("## 🎛️ Controls")

# Dataset parameters
st.sidebar.markdown("### Dataset Parameters")
n_points = st.sidebar.slider("Number of points", 20, 200, 50)
noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.5, 0.1)
outliers = st.sidebar.slider("Number of outliers", 0, 10, 2)
seed = st.sidebar.slider("Random seed", 1, 100, 42)

# Model parameters
st.sidebar.markdown("### Model Parameters")
manual_slope = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0, 0.1)
manual_intercept = st.sidebar.slider("Intercept (b)", -5.0, 5.0, 0.0, 0.1)

# Gradient descent parameters
st.sidebar.markdown("### Gradient Descent")
learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.5, 0.1, 0.001)
max_iterations = st.sidebar.slider("Max iterations", 10, 200, 100)

# Generate data
X, y, true_slope, true_intercept = generate_data(n_points, noise_level, outliers, seed)

# Initialize session state for gradient descent
if 'gd_history' not in st.session_state:
    st.session_state.gd_history = []
    st.session_state.gd_slope = 0.0
    st.session_state.gd_intercept = 0.0
    st.session_state.gd_running = False

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Line Fit", 
    "📏 Error Visualization", 
    "🗻 Loss Landscape", 
    "⚡ Gradient Descent", 
    "🎯 Learning Rate Effects",
    "🔍 Noise & Outliers"
])

with tab1:
    st.markdown('<h2 class="section-header">Data Distribution & Line Fitting</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create scatter plot with adjustable line
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            name='Data Points',
            marker=dict(size=8, color='blue', opacity=0.7)
        ))
        
        # Add true line
        y_true_line = true_slope * X + true_intercept
        fig.add_trace(go.Scatter(
            x=X, y=y_true_line,
            mode='lines',
            name='True Line',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add manual line
        y_manual = manual_slope * X + manual_intercept
        fig.add_trace(go.Scatter(
            x=X, y=y_manual,
            mode='lines',
            name='Your Line',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Interactive Line Fitting",
            xaxis_title="X",
            yaxis_title="y",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Model Equation")
        st.latex(f"y = {manual_slope:.2f}x + {manual_intercept:.2f}")
        
        st.markdown("### 🎯 True Parameters")
        st.info(f"True slope: {true_slope:.2f}")
        st.info(f"True intercept: {true_intercept:.2f}")
        
        current_mse = compute_mse(X, y, manual_slope, manual_intercept)
        optimal_mse = compute_mse(X, y, true_slope, true_intercept)
        
        st.markdown("### 📊 Performance")
        st.metric("Current MSE", f"{current_mse:.3f}")
        st.metric("Optimal MSE", f"{optimal_mse:.3f}")
        
        improvement = ((current_mse - optimal_mse) / optimal_mse) * 100
        if improvement > 0:
            st.error(f"Your line is {improvement:.1f}% worse than optimal")
        else:
            st.success("Perfect fit!")

with tab2:
    st.markdown('<h2 class="section-header">Error Visualization</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create residuals plot
        y_pred = manual_slope * X + manual_intercept
        residuals = y - y_pred
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Data with Residuals', 'Residuals Distribution'),
                           vertical_spacing=0.15)
        
        # Top plot: data with residuals
        fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data', 
                                marker=dict(color='blue', size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Fitted Line',
                                line=dict(color='red', width=2)), row=1, col=1)
        
        # Add residual lines
        for i in range(len(X)):
            fig.add_trace(go.Scatter(x=[X[i], X[i]], y=[y[i], y_pred[i]], 
                                   mode='lines', line=dict(color='gray', width=1),
                                   showlegend=False), row=1, col=1)
        
        # Bottom plot: residuals histogram
        fig.add_trace(go.Histogram(x=residuals, name='Residuals', 
                                  marker_color='lightblue', opacity=0.7), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Error Metrics")
        
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        st.metric("MSE", f"{mse:.3f}")
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("MAE", f"{mae:.3f}")
        
        st.markdown("### 📈 Residual Statistics")
        st.write(f"Mean: {np.mean(residuals):.3f}")
        st.write(f"Std: {np.std(residuals):.3f}")
        st.write(f"Min: {np.min(residuals):.3f}")
        st.write(f"Max: {np.max(residuals):.3f}")
        
        # Squared errors visualization
        squared_errors = residuals**2
        st.markdown("### 🔥 Largest Errors")
        worst_indices = np.argsort(squared_errors)[-3:]
        for i, idx in enumerate(worst_indices[::-1]):
            st.write(f"{i+1}. Point {idx}: Error = {residuals[idx]:.3f}")

with tab3:
    st.markdown('<h2 class="section-header">Loss Surface Visualization</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create loss surface
        slope_range = np.linspace(-1, 5, 50)
        intercept_range = np.linspace(-3, 4, 50)
        S, I = np.meshgrid(slope_range, intercept_range)
        
        # Compute loss for each combination
        Z = np.zeros_like(S)
        for i in range(len(slope_range)):
            for j in range(len(intercept_range)):
                Z[j, i] = compute_mse(X, y, slope_range[i], intercept_range[j])
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=Z, x=S, y=I, colorscale='Viridis')])
        
        # Add current point
        current_loss = compute_mse(X, y, manual_slope, manual_intercept)
        fig.add_trace(go.Scatter3d(
            x=[manual_slope], y=[manual_intercept], z=[current_loss],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Current Position'
        ))
        
        # Add optimal point
        optimal_loss = compute_mse(X, y, true_slope, true_intercept)
        fig.add_trace(go.Scatter3d(
            x=[true_slope], y=[true_intercept], z=[optimal_loss],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Optimal Position'
        ))
        
        fig.update_layout(
            title='Loss Surface',
            scene=dict(
                xaxis_title='Slope',
                yaxis_title='Intercept',
                zaxis_title='MSE Loss'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Contour View")
        
        # Create contour plot
        fig_contour = go.Figure(data=go.Contour(z=Z, x=slope_range, y=intercept_range,
                                               colorscale='Viridis'))
        
        # Add current point
        fig_contour.add_trace(go.Scatter(
            x=[manual_slope], y=[manual_intercept],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Current'
        ))
        
        # Add optimal point
        fig_contour.add_trace(go.Scatter(
            x=[true_slope], y=[true_intercept],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Optimal'
        ))
        
        fig_contour.update_layout(
            title='Loss Contours',
            xaxis_title='Slope',
            yaxis_title='Intercept',
            height=400
        )
        
        st.plotly_chart(fig_contour, use_container_width=True)
        
        st.markdown("### 🎯 Navigation")
        st.info("Red X: Your current position")
        st.success("Green ⭐: Optimal position")
        st.write("Darker regions = Lower loss")

with tab4:
    st.markdown('<h2 class="section-header">Gradient Descent Animation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gradient descent controls
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🚀 Start Gradient Descent"):
                st.session_state.gd_slope = np.random.uniform(-2, 4)
                st.session_state.gd_intercept = np.random.uniform(-2, 3)
                st.session_state.gd_history = [(st.session_state.gd_slope, st.session_state.gd_intercept)]
                st.session_state.gd_running = True
        
        with col_b:
            if st.button("⏸️ Reset"):
                st.session_state.gd_history = []
                st.session_state.gd_running = False
        
        with col_c:
            animate = st.checkbox("Auto-animate", value=False)
        
        # Run gradient descent
        if st.session_state.gd_running and len(st.session_state.gd_history) < max_iterations:
            current_slope, current_intercept = st.session_state.gd_history[-1]
            new_slope, new_intercept = gradient_descent_step(X, y, current_slope, current_intercept, learning_rate)
            st.session_state.gd_history.append((new_slope, new_intercept))
            
            # Check convergence
            if len(st.session_state.gd_history) > 1:
                prev_loss = compute_mse(X, y, current_slope, current_intercept)
                new_loss = compute_mse(X, y, new_slope, new_intercept)
                if abs(prev_loss - new_loss) < 1e-6:
                    st.session_state.gd_running = False
        
        # Plot gradient descent path
        if st.session_state.gd_history:
            # Create contour plot
            slope_range = np.linspace(-1, 5, 30)
            intercept_range = np.linspace(-3, 4, 30)
            S, I = np.meshgrid(slope_range, intercept_range)
            Z = np.zeros_like(S)
            for i in range(len(slope_range)):
                for j in range(len(intercept_range)):
                    Z[j, i] = compute_mse(X, y, slope_range[i], intercept_range[j])
            
            fig = go.Figure(data=go.Contour(z=Z, x=slope_range, y=intercept_range,
                                           colorscale='Viridis', opacity=0.7))
            
            # Add gradient descent path
            history = np.array(st.session_state.gd_history)
            fig.add_trace(go.Scatter(
                x=history[:, 0], y=history[:, 1],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8, color='red'),
                name='GD Path'
            ))
            
            # Add start and end points
            if len(history) > 0:
                fig.add_trace(go.Scatter(
                    x=[history[0, 0]], y=[history[0, 1]],
                    mode='markers',
                    marker=dict(size=15, color='orange', symbol='circle'),
                    name='Start'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[history[-1, 0]], y=[history[-1, 1]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='x'),
                    name='Current'
                ))
            
            # Add optimal point
            fig.add_trace(go.Scatter(
                x=[true_slope], y=[true_intercept],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                name='Optimal'
            ))
            
            fig.update_layout(
                title='Gradient Descent Path',
                xaxis_title='Slope',
                yaxis_title='Intercept',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Auto-refresh for animation
            if animate and st.session_state.gd_running:
                time.sleep(0.1)
                st.rerun()
    
    with col2:
        if st.session_state.gd_history:
            st.markdown("### 📊 Progress")
            
            current_slope, current_intercept = st.session_state.gd_history[-1]
            current_loss = compute_mse(X, y, current_slope, current_intercept)
            
            st.metric("Iteration", len(st.session_state.gd_history))
            st.metric("Current Loss", f"{current_loss:.4f}")
            st.metric("Current Slope", f"{current_slope:.3f}")
            st.metric("Current Intercept", f"{current_intercept:.3f}")
            
            # Loss over iterations
            losses = [compute_mse(X, y, s, i) for s, i in st.session_state.gd_history]
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(len(losses))), y=losses,
                mode='lines+markers',
                name='Loss',
                line=dict(color='blue', width=2)
            ))
            
            fig_loss.update_layout(
                title='Loss vs Iteration',
                xaxis_title='Iteration',
                yaxis_title='MSE Loss',
                height=300
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Convergence info
            if len(losses) > 1:
                improvement = losses[0] - losses[-1]
                st.success(f"Loss reduced by {improvement:.4f}")
        else:
            st.info("Click 'Start Gradient Descent' to begin optimization")

with tab5:
    st.markdown('<h2 class="section-header">Learning Rate Effects</h2>', unsafe_allow_html=True)
    
    # Compare different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.3]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig = go.Figure()
    
    for lr, color in zip(learning_rates, colors):
        # Run gradient descent for each learning rate
        slope, intercept = 0.0, 0.0
        losses = []
        
        for _ in range(50):
            loss = compute_mse(X, y, slope, intercept)
            losses.append(loss)
            slope, intercept = gradient_descent_step(X, y, slope, intercept, lr)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(losses))), y=losses,
            mode='lines',
            name=f'LR = {lr}',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title='Learning Rate Comparison',
        xaxis_title='Iteration',
        yaxis_title='MSE Loss',
        height=500,
        yaxis_type='log'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🐌 Small Learning Rate (0.001)")
        st.write("✅ Stable convergence")
        st.write("❌ Very slow progress")
        st.write("💡 Safe but inefficient")
    
    with col2:
        st.markdown("### 🚀 Large Learning Rate (0.3)")
        st.write("✅ Fast initial progress")
        st.write("❌ May overshoot minimum")
        st.write("💡 Risk of divergence")

with tab6:
    st.markdown('<h2 class="section-header">Noise & Outliers Impact</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Current Dataset")
        
        # Fit line to current data
        coeffs = np.polyfit(X, y, 1)
        fitted_slope, fitted_intercept = coeffs[0], coeffs[1]
        
        fig = go.Figure()
        
        # Identify outliers (points with large residuals)
        y_pred = fitted_slope * X + fitted_intercept
        residuals = np.abs(y - y_pred)
        outlier_threshold = np.percentile(residuals, 75) + 1.5 * (np.percentile(residuals, 75) - np.percentile(residuals, 25))
        is_outlier = residuals > outlier_threshold
        
        # Plot normal points
        fig.add_trace(go.Scatter(
            x=X[~is_outlier], y=y[~is_outlier],
            mode='markers',
            name='Normal Points',
            marker=dict(size=8, color='blue', opacity=0.7)
        ))
        
        # Plot outliers
        if np.any(is_outlier):
            fig.add_trace(go.Scatter(
                x=X[is_outlier], y=y[is_outlier],
                mode='markers',
                name='Outliers',
                marker=dict(size=10, color='red', symbol='x')
            ))
        
        # Add fitted line
        fig.add_trace(go.Scatter(
            x=X, y=y_pred,
            mode='lines',
            name='Fitted Line',
            line=dict(color='green', width=2)
        ))
        
        # Add true line
        y_true_line = true_slope * X + true_intercept
        fig.add_trace(go.Scatter(
            x=X, y=y_true_line,
            mode='lines',
            name='True Line',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Impact of Noise and Outliers',
            xaxis_title='X',
            yaxis_title='y',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Robustness Analysis")
        
        # Compare fits with and without outliers
        if np.any(is_outlier):
            # Fit without outliers
            X_clean = X[~is_outlier]
            y_clean = y[~is_outlier]
            coeffs_clean = np.polyfit(X_clean, y_clean, 1)
            clean_slope, clean_intercept = coeffs_clean[0], coeffs_clean[1]
            
            st.markdown("#### With Outliers:")
            st.write(f"Slope: {fitted_slope:.3f}")
            st.write(f"Intercept: {fitted_intercept:.3f}")
            st.write(f"MSE: {compute_mse(X, y, fitted_slope, fitted_intercept):.3f}")
            
            st.markdown("#### Without Outliers:")
            st.write(f"Slope: {clean_slope:.3f}")
            st.write(f"Intercept: {clean_intercept:.3f}")
            st.write(f"MSE: {compute_mse(X_clean, y_clean, clean_slope, clean_intercept):.3f}")
            
            st.markdown("#### True Parameters:")
            st.write(f"Slope: {true_slope:.3f}")
            st.write(f"Intercept: {true_intercept:.3f}")
            
            # Calculate bias
            slope_bias = abs(fitted_slope - true_slope)
            intercept_bias = abs(fitted_intercept - true_intercept)
            
            st.markdown("#### Impact:")
            st.error(f"Slope bias: {slope_bias:.3f}")
            st.error(f"Intercept bias: {intercept_bias:.3f}")
            st.warning(f"Outliers detected: {np.sum(is_outlier)}")
        else:
            st.success("No significant outliers detected!")
            st.write(f"Current noise level: {noise_level:.2f}")
            st.write(f"Fitted slope: {fitted_slope:.3f}")
            st.write(f"Fitted intercept: {fitted_intercept:.3f}")

# Footer
st.markdown("---")
st.markdown("### 🎓 Key Takeaways")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Linear Regression Basics**")
    st.write("• Models linear relationships: y = mx + b")
    st.write("• Minimizes Mean Squared Error (MSE)")
    st.write("• Sensitive to outliers")

with col2:
    st.markdown("**Gradient Descent**")
    st.write("• Iteratively optimizes parameters")
    st.write("• Learning rate controls step size")
    st.write("• Converges to global minimum")

with col3:
    st.markdown("**Data Quality Matters**")
    st.write("• Noise affects model accuracy")
    st.write("• Outliers can bias results")
    st.write("• More data generally helps")