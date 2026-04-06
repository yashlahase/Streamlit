# Linear Regression Interactive Visualizer

An interactive Streamlit application that visualizes how Linear Regression learns, fits data, minimizes error, and converges to optimal solutions.

## Features

### 🔍 Core Visualizations
- **Data Distribution & Line Fitting**: Interactive sliders to adjust slope and intercept
- **Error Visualization**: Real-time MSE calculation with residual plots
- **Loss Landscape**: 3D surface and contour plots of the parameter space
- **Gradient Descent**: Animated optimization process with path tracking
- **Learning Rate Effects**: Comparison of different learning rates
- **Noise & Outliers**: Impact analysis on model robustness

### 🎛️ Interactive Controls
- Dataset parameters (size, noise, outliers)
- Manual parameter adjustment
- Gradient descent settings
- Real-time updates and animations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run linear_regression_visualizer.py
```

## Usage

1. **Explore Data & Line Fit**: Use the sidebar sliders to manually adjust the line and see how it fits the data
2. **Visualize Errors**: See residuals and error metrics in real-time
3. **Understand Loss Surface**: Explore the 3D loss landscape and contour plots
4. **Watch Gradient Descent**: Start the optimization and watch the algorithm find the optimal parameters
5. **Compare Learning Rates**: See how different learning rates affect convergence
6. **Analyze Robustness**: Add noise and outliers to see their impact

## Key Learning Concepts

- **Linear Relationship**: y = mx + b
- **Mean Squared Error**: MSE = Σ(y - ŷ)²/n
- **Gradient Descent**: Iterative optimization algorithm
- **Learning Rate**: Controls optimization step size
- **Overfitting vs Underfitting**: Balance between bias and variance

## Educational Value

This visualizer helps students understand:
- How linear regression finds the best-fit line
- The role of the loss function in optimization
- How gradient descent navigates the parameter space
- The impact of hyperparameters like learning rate
- Real-world challenges like noise and outliers