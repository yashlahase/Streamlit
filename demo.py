"""
Linear Regression Visualizer - Demo Script
This script demonstrates the core concepts implemented in the Streamlit app.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib not available - skipping visualization save")

def generate_sample_data(n_points=50, noise=0.5, seed=42):
    """Generate sample data for demonstration"""
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_points)
    true_slope, true_intercept = 2, 1
    y_true = true_slope * X + true_intercept
    noise_vals = np.random.normal(0, noise, n_points)
    y = y_true + noise_vals
    return X, y, true_slope, true_intercept

def compute_mse(X, y, slope, intercept):
    """Compute Mean Squared Error"""
    y_pred = slope * X + intercept
    return np.mean((y - y_pred) ** 2)

def gradient_descent_demo(X, y, learning_rate=0.1, max_iterations=100):
    """Demonstrate gradient descent optimization"""
    # Initialize parameters randomly
    slope = np.random.uniform(-2, 4)
    intercept = np.random.uniform(-2, 3)
    
    history = []
    
    for i in range(max_iterations):
        # Compute predictions and loss
        y_pred = slope * X + intercept
        loss = np.mean((y - y_pred) ** 2)
        history.append((slope, intercept, loss))
        
        # Compute gradients
        n = len(X)
        dw = -2/n * np.sum(X * (y - y_pred))
        db = -2/n * np.sum(y - y_pred)
        
        # Update parameters
        slope = slope - learning_rate * dw
        intercept = intercept - learning_rate * db
        
        # Check convergence
        if i > 0 and abs(history[-1][2] - history[-2][2]) < 1e-6:
            print(f"Converged after {i+1} iterations")
            break
    
    return history

def main():
    """Run the demo"""
    print("🔍 Linear Regression Visualizer - Core Concepts Demo")
    print("=" * 60)
    
    # Generate sample data
    X, y, true_slope, true_intercept = generate_sample_data()
    print(f"📊 Generated {len(X)} data points")
    print(f"🎯 True parameters: slope={true_slope}, intercept={true_intercept}")
    
    # Demonstrate MSE calculation
    test_slope, test_intercept = 1.5, 0.5
    mse = compute_mse(X, y, test_slope, test_intercept)
    optimal_mse = compute_mse(X, y, true_slope, true_intercept)
    print(f"📏 MSE with test parameters ({test_slope}, {test_intercept}): {mse:.4f}")
    print(f"📏 MSE with optimal parameters: {optimal_mse:.4f}")
    
    # Run gradient descent
    print("\n⚡ Running Gradient Descent...")
    history = gradient_descent_demo(X, y, learning_rate=0.1)
    
    final_slope, final_intercept, final_loss = history[-1]
    print(f"🎯 Final parameters: slope={final_slope:.3f}, intercept={final_intercept:.3f}")
    print(f"📉 Final loss: {final_loss:.4f}")
    print(f"🎉 Error reduction: {((history[0][2] - final_loss) / history[0][2] * 100):.1f}%")
    
    # Create a simple visualization
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Data and fitted line
        plt.subplot(1, 3, 1)
        plt.scatter(X, y, alpha=0.7, color='blue', label='Data')
        y_true_line = true_slope * X + true_intercept
        y_fitted = final_slope * X + final_intercept
        plt.plot(X, y_true_line, 'g--', label='True Line', linewidth=2)
        plt.plot(X, y_fitted, 'r-', label='Fitted Line', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Data & Line Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Loss over iterations
        plt.subplot(1, 3, 2)
        losses = [h[2] for h in history]
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('Gradient Descent Convergence')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Parameter trajectory
        plt.subplot(1, 3, 3)
        slopes = [h[0] for h in history]
        intercepts = [h[1] for h in history]
        plt.plot(slopes, intercepts, 'ro-', markersize=3, alpha=0.7, label='GD Path')
        plt.plot(slopes[0], intercepts[0], 'go', markersize=10, label='Start')
        plt.plot(slopes[-1], intercepts[-1], 'ro', markersize=10, label='End')
        plt.plot(true_slope, true_intercept, 'g*', markersize=15, label='True')
        plt.xlabel('Slope')
        plt.ylabel('Intercept')
        plt.title('Parameter Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('linear_regression_demo.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'linear_regression_demo.png'")
    else:
        print(f"\n📊 Visualization skipped (matplotlib not available)")
    
    print("\n🚀 To run the full interactive visualizer:")
    print("   streamlit run linear_regression_visualizer.py")
    print("   or")
    print("   ./run_app.sh")

if __name__ == "__main__":
    main()