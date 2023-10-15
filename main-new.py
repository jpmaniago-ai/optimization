# 1. setup vscode -> https://code.visualstudio.com/docs/python/python-tutorial
# 2. Open the Command Palette (Ctrl+Shift+P), start typing the Python: Create Environment command to search, and then select the command.
#    The command presents a list of environment types, Venv or Conda. For this example, select Venv.
# 2. intall using terminal
#    python -m pip install streamlit
#    python -m pip install scipy
#    python -m pip install matplotlib
#    python -m pip install scikit-learn
#    python -m pip install tensorflow
#    python -m pip install pydot==1.2.3 
#    python -m pip install graphviz
#    python -m pip install ann_visualizer
# 3. Download GraphViz ->>> This is used to generate the Neural Network Architecture Image and PDF
#    a. Download link ->>> https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/9.0.0/windows_10_msbuild_Release_graphviz-9.0.0-win32.zip
#    b. Then unzip in ->> C:\Graphviz\bin 
#    c. Add C:\Graphviz\bin to System Path  ->> Ganito kung paano https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/
# 4. To run this program, using the terminal type the command below. It will run the program and open a browser
#    streamlit run main-new.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import altair as alt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz
import os
import base64

# Function to train the selected regression model
def train_regression_model(data, selected_model):
    X = data[['QuantitySold', 'UnitCost', 'Profit', 'Year', 'Month']]
    y = data['UnitPrice']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_model == 'Multiple Regression':
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    return model

# Function to display the model's performance
def display_model_performance(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Function to create scatter plots and interpretation text
def create_scatter_plots(df, feature_name, y_name):
    chart = alt.Chart(df).mark_circle().encode(
        x=feature_name,
        y=alt.Y(y_name, title="Unit Price"),
        tooltip=[feature_name, y_name]
    ).properties(
        width=300,
        height=200
    )

    # Calculate the correlation coefficient
    correlation = df[feature_name].corr(df[y_name])

    # Generate dynamic interpretation text
    if correlation > 0.7:
        interpretation_text = f"The strong positive correlation coefficient ({correlation:.2f}) indicates that as {feature_name} increases, Unit Price tends to increase significantly."
    elif correlation > 0.3:
        interpretation_text = f"The moderate positive correlation coefficient ({correlation:.2f}) suggests that there is a positive relationship between {feature_name} and Unit Price."
    elif correlation < -0.7:
        interpretation_text = f"The strong negative correlation coefficient ({correlation:.2f}) indicates that as {feature_name} increases, Unit Price tends to decrease significantly."
    elif correlation < -0.3:
        interpretation_text = f"The moderate negative correlation coefficient ({correlation:.2f}) suggests a negative relationship between {feature_name} and Unit Price."
    else:
        interpretation_text = f"The correlation coefficient is close to zero ({correlation:.2f}), suggesting a weak or no linear relationship between {feature_name} and Unit Price."

    return chart, interpretation_text

# Function to train the neural network and make predictions
def train_and_predict(X_train, y_train, X_test, y_test, X_new):
    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_new_scaled = scaler.transform(X_new)
    
    # Create a neural network model
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output layer with 1 neuron for UnitPrice

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    #model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Plot the training history
    st.subheader("Training History")
    plot_training_history(history)

    # Interpret training history
    interpret_training_history(history)

    # Visualize the model
    #st.subheader("Neural Network Architecture")
    #st.image(plot_model_architecture(model))

    # Visualize the model
    st.subheader("Neural Network Architecture using ann_viz")
    # Generate PDF using ann_viz
    ann_viz(model, title="Neural Network Architecture", view=False, filename="ann_viz_network.pdf")
    
    # Convert PDF to Graphviz format
    graphviz_data = ann_viz(model, title="Neural Network Architecture", view=True, format="pdf", filename="ann_viz_network")
    
    # Display the Graphviz data using st.graphviz_chart
    st.graphviz_chart(graphviz_data)

    # Make predictions
    predictions = model.predict(X_new_scaled)

    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test)
    st.subheader("Model Evaluation on Test Data")
    st.write(f"Test Loss: {loss}")

    return predictions

#def plot_model_architecture(model):
#    plot_model(model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)
#    return 'model_visualization.png'

def plot_training_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model')
    plt.ylabel('Loss (mean squared error)')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    st.pyplot(plt)

# Interpret training history data
def interpret_training_history(history):
    st.subheader("Interpretation of Training History")
    st.write("The training history plot shows how the model's loss function changes over training epochs.")
    st.write("Here's what you can interpret:")
    st.write("1. **Training Loss (Blue Line):** The loss on the training data decreases as the model learns from the data.")
    st.write("2. **Validation Loss (Orange Line):** The loss on a separate validation dataset. It helps you check for overfitting; if the validation loss starts increasing, it might be overfitting.")
    st.write("3. **Epochs:** Each epoch represents one complete pass through the training data.")
    st.write("4. **Lower Loss:** Lower loss values indicate a better model fit, but be cautious of overfitting if the validation loss starts to increase.")


# Streamlit UI
st.title('Optimization Models')
st.write('Select a model.')

# Use st.sidebar for the left pane
st.sidebar.header('Model Selection')
selected_model = st.sidebar.selectbox('Select a Model', ['Multiple Regression', 'Neural Network'])

st.sidebar.header('File Upload')
historical_data = st.sidebar.file_uploader('Upload Historical Data (CSV)', type=['csv'])
new_data = st.sidebar.file_uploader('Upload New Data (CSV)', type=['csv'])

if historical_data is not None and new_data is not None:
    # Read and store the historical data
    historical_df = pd.read_csv(historical_data)

    # Read and store the new data
    new_df = pd.read_csv(new_data)

    # Display historical data table
    st.subheader('Historical Data')
    st.write(historical_df)

    # Display historical data table
    st.subheader('New Data')
    st.write(new_df)

    if (selected_model == 'Multiple Regression'):
        # Train the selected model
        model = train_regression_model(historical_df, selected_model)

        # Predict unit price for new data
        new_data_X = new_df[['QuantitySold', 'UnitCost', 'Profit', 'Year', 'Month']]
        new_df['Predicted_UnitPrice'] = model.predict(new_data_X)

        # Model performance
        y_test = historical_df['UnitPrice']
        y_pred = model.predict(historical_df[['QuantitySold', 'UnitCost', 'Profit', 'Year', 'Month']])
        mse, r2 = display_model_performance(y_test, y_pred)

        # Display historical data table
        #st.subheader('Historical Data')
        #st.write(historical_df)

        # Display scatter plots for feature relationships with unit price
        st.subheader('Feature Relationships with Unit Price')
        scatter_plots = []

        for feature in historical_df.columns[:-2]:  # Exclude 'UnitPrice' and 'Predicted_UnitPrice
            scatter_plot, interpretation_text = create_scatter_plots(historical_df, feature, 'UnitPrice')
            st.altair_chart(scatter_plot, use_container_width=True)
            st.write(interpretation_text)

        # Display feature coefficients in a table with computation, formula, and result
        st.subheader('Feature Coefficients')
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Create a DataFrame to display coefficients, computation, formula, and result
        coef_data = {
            'Feature': historical_df.columns[:-1],  # Exclude 'UnitPrice'
            'Coefficient': coefficients,
            'Computation': coefficients * historical_df.mean()[:-1],
        }
        
        coef_data['Formula'] = [
            f"{round(coef, 2)} * {feature}" for coef, feature in zip(coefficients, historical_df.columns[:-1])
        ]
        
        coef_data['Result'] = [
            coef * historical_df.mean()[feature] for coef, feature in zip(coefficients, historical_df.columns[:-1])
        ]
        
        coef_df = pd.DataFrame(coef_data)
        st.write(coef_df)

        # Explanation of why coefficients are important
        st.subheader('Why Coefficients are Important')
        st.write('In regression analysis, coefficients represent the impact of each feature on the target variable (Unit Price).')
        st.write('The coefficients provide insights into the direction and strength of the relationship between features and the target variable.')
        st.write('For example, a positive coefficient indicates a positive relationship, where an increase in the feature leads to an increase in Unit Price, and vice versa.')
        st.write('Understanding these coefficients helps in making data-driven decisions and predictions.')

        # Display model performance
        st.subheader('Model Performance')
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R^2) Score: {r2:.2f}")
        st.write("Interpretation:")
        st.write("Mean Squared Error (MSE) measures the average squared difference between the actual and predicted values. A lower MSE indicates a better fit.")
        st.write("R-squared (R^2) Score measures the proportion of the variance in the dependent variable (Unit Price) that is predictable from the independent variables. An R-squared score closer to 1 indicates a better model fit.")

        # Display computation of Predicted Unit Price
        st.subheader('Computation of Predicted Unit Price')
        st.write("The predicted Unit Price for the new data is computed using the following formula:")
        st.latex('Predicted\_UnitPrice = \\beta_0 + \\beta_1 * QuantitySold + \\beta_2 * UnitCost + \\beta_3 * Profit + \\beta_4 * Year + \\beta_5 * Month')
        st.write('Where:')
        st.write('- $\\beta_0$ is the intercept')
        st.write('- $\\beta_1, \\beta_2, \\beta_3, \\beta_4, \\beta_5$ are the coefficients for the respective features')
        st.write("The new data is used as input for the model to calculate the Predicted Unit Price.")

        # Display model equation
        st.subheader('Model Equation')
        formula = f"Prediction = {intercept:.2f} + " + " + ".join([f"{round(coef, 2)} * {feature}" for coef, feature in zip(coefficients, historical_df.columns[:-1])])
        st.write(formula)

        # Display new data and predictions
        st.subheader('New Data and Predictions')
        st.write("New Data:")
        st.write(new_df)

        st.write("Prediction for New Data:")
        st.write(new_df[['Predicted_UnitPrice']])

    elif (selected_model == 'Neural Network'):
        # Explanation
        # Explanation of why coefficients are important
        st.subheader('Regression using Neural Network')
        st.write('Create a neural network with a single input layer, one output layer, and no hidden layers to function as a regression model.')

        # Prepare data for training and prediction
        X = historical_df[['QuantitySold', 'UnitCost', 'Profit', 'Year', 'Month']]
        y = historical_df[['UnitPrice']]
        X_new = new_df[['QuantitySold', 'UnitCost', 'Profit', 'Year', 'Month']]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and make predictions
        predictions = train_and_predict(X_train, y_train, X_test, y_test, X_new)

        # Display the predictions
        st.subheader("Prediction for New Data:")
        st.write(predictions)
