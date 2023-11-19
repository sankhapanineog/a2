# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Initialize parameters (weights and biases)
def initialize_parameters(input_size):
    parameters = {
        'W': np.random.randn(1, input_size),
        'b': np.zeros((1, 1))
    }
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W, X) + b
    A = sigmoid(Z)

    return A

# Train neural network
def train_neural_network(X, Y, learning_rate, num_iterations):
    np.random.seed(42)
    input_size = X.shape[0]
    parameters = initialize_parameters(input_size)

    # List to store cost during training for visualization
    cost_history = []

    for i in range(num_iterations):
        A = forward_propagation(X, parameters)
        cost = mean_squared_error(Y, A)

        dZ = A - Y
        dW = np.dot(dZ, X.T)
        db = np.sum(dZ)

        parameters['W'] -= learning_rate * dW
        parameters['b'] -= learning_rate * db

        cost_history.append(cost)  # Append the cost to the history list

        if i % 100 == 0:
            st.write(f"Cost after iteration {i}: {cost}")

    # Plot the cost during training
    st.subheader("Cost During Training")
    iterations = np.arange(0, num_iterations, 100)  # Adjusted to match iteration steps
    fig_cost = px.line(x=iterations, y=cost_history[::100], labels={'x': 'Iteration', 'y': 'Cost'})
    st.plotly_chart(fig_cost)

    return parameters

# Make predictions
def predict(X, parameters):
    A = forward_propagation(X, parameters)
    return A

# Generate random time-series data for three days
def generate_random_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-03 23:59:00", freq='T')
    values = np.random.normal(loc=0, scale=1, size=len(timestamps))
    data = pd.DataFrame({'timestamp': timestamps, 'value': values})
    return data

# Streamlit app
def main():
    st.title("Group 7: Neural Network Asset Health Prediction App")

    # Option to Generate Random Data or Upload CSV
    data_option = st.radio("Choose Data Source:", ("Generate Random Data", "Upload CSV"))

    if data_option == "Generate Random Data":
        # Generate random data for three days
        data = generate_random_data()
    else:
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or choose to generate random data.")
            return

    # Sidebar - Neural Network Configuration
    st.sidebar.header("Neural Network Configuration")
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=5000, value=1000, step=100)
    threshold = st.sidebar.slider("Threshold for Health Prediction", min_value=0.1, max_value=0.9, value=0.5, step=0.1)

    # Prepare data
    if 'health_label' in data.columns:
        # Use actual labels from the data
        Y = (data['health_label'] == 'Healthy').astype(int).values.reshape(1, -1)
    else:
        # If 'health_label' is not present, create placeholder labels
        Y = np.zeros((1, len(data)))

    X = data['value'].values.reshape(1, -1)

    # Train neural network
    parameters = train_neural_network(X, Y, learning_rate, num_iterations)

    # Make predictions for the original data
    predictions_original = predict(X, parameters)

    # Label data using threshold
    data['predicted_health_label'] = np.where(predictions_original.flatten() > threshold, 'Healthy', 'Unhealthy')

    # Plot original data with predicted health labels
    st.subheader("Data Plot with Predicted Health Labels")
    fig = px.line(data, x='timestamp', y='value', color='predicted_health_label', labels={'value': 'Data'})
    st.plotly_chart(fig)

    # Performance Matrix and Explanations
    st.subheader("Performance Matrix and Advantages of AI-based Asset Health Forecasting")

    # Confusion Matrix, Accuracy, and Classification Report
    y_true = Y.flatten()
    y_pred = predictions_original.flatten() > threshold
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)

    st.write(f"Confusion Matrix:\n{cm}")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Classification Report:\n{classification_rep}")

    # Explain the advantages of AI-based asset health forecasting
    st.subheader("Advantages of AI-based Asset Health Forecasting:")
    st.write("1. **Early Detection:** AI models can detect subtle patterns indicative of asset degradation before"
             " visible signs appear, allowing for early intervention and maintenance.")

    st.write("2. **Data-Driven Insights:** AI algorithms analyze large datasets, providing data-driven insights into"
             " asset performance and health based on historical patterns and real-time data.")

    st.write("3. **Predictive Maintenance:** AI can predict when equipment is likely to fail, enabling proactive"
             " maintenance schedules and minimizing downtime.")

    st.write("4. **Cost Savings:** Predictive maintenance and early detection of issues lead to cost savings by"
             " preventing major breakdowns and reducing unplanned downtime.")

    st.write("5. **Continuous Improvement:** AI models can be continually trained and improved with new data, adapting"
             " to changing conditions and improving accuracy over time.")

if __name__ == "__main__":
    main()
