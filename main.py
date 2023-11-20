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
    st.title("Neural Network based Asset Health Prediction ")

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
    y_pred = (predictions_original.flatten() > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)

    # Display the Confusion Matrix
    st.markdown("### Confusion Matrix")
    st.write(pd.DataFrame(cm, columns=['Predicted'], index=['Actual']))

    # Display Accuracy and Classification Report
    st.markdown(f"*Accuracy:* {accuracy:.2%}")
    st.markdown("### Classification Report")
    st.write(classification_rep)

    # Explain the advantages of AI-based asset health forecasting
    st.subheader(" ")
    st.write(" *Early Detection:* AI models can detect subtle patterns indicative of asset degradation before"
             " visible signs appear, allowing for early intervention and maintenance.")

    st.write("Embark on a transformative journey towards proactive asset management with our cutting-edge Neural Network-based Asset Health Prediction software. Secure your spot in our early registration and be among the pioneers leveraging the power of artificial intelligence for asset forecasting. ")

    st.write(" For collaboration mail us at neogsankhapani@gmail.com ")

    st.write(" ")

    st.write(" .")

if __name__ == "__main__":
    main()
