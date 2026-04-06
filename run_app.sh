#!/bin/bash

# Linear Regression Visualizer Launcher
echo "🚀 Starting Linear Regression Visualizer..."
echo "📊 This interactive app will help you understand how Linear Regression works!"
echo ""

# Activate virtual environment and run the simplified app
source env/bin/activate
streamlit run linear_regression_simple.py

echo ""
echo "✅ Thanks for using the Linear Regression Visualizer!"