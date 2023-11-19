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

    for i in range(num_iterations):
        A = forward_propagation(X, parameters)
        cost = mean_squared_error(Y, A)

        dZ = A - Y
        dW = np.dot(dZ, X.T)
        db = np.sum(dZ)

        parameters['W'] -= learning_rate * dW
        parameters['b'] -= learning_rate * db

        if i % 100 == 0:
            st.write(f"Cost after iteration {i}: {cost}")

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
    X = data['value'].values.reshape(1, -1)
    Y = np.zeros((1, len(X)))  # Initialize Y with zeros

    # Set Y for the last portion of the data as a placeholder for future health label
    Y[0, -100:] = 1  # Placeholder for the last 100 values, assuming the data has 4320 values

    # Train neural network
    parameters = train_neural_network(X, Y, learning_rate, num_iterations)

    # Make predictions for the original data
    predictions_original = predict(X, parameters)

    # Label data using threshold
    data['health_label'] = np.where(predictions_original.flatten() > threshold, 'Healthy', 'Unhealthy')

    # Plot original data with health labels
    st.subheader("Data Plot with Health Labels")
    fig = px.line(data, x='timestamp', y='value', color='health_label', labels={'value': 'Data'})
    st.plotly_chart(fig)

    # Performance Matrix and Explanations
    st.subheader("Performance Matrix")
    y_true = Y.flatten()
    y_pred = predictions_original.flatten() > threshold
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)

    st.write(f"Confusion Matrix:\n{cm}")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Classification Report:\n{classification_rep}")

if __name__ == "__main__":
    main()
