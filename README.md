# Multivariate Time-Series Forecasting Using LSTM

A deep learning-based multivariate time-series forecasting project using Long Short-Term Memory (LSTM) neural networks. The project uses the Jena Climate Dataset to learn temporal dependencies among multiple weather variables and forecast future observations.

The project includes data preprocessing, feature scaling, sequential data generation, LSTM model development, time-series cross-validation, hyperparameter optimization, and evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Project Overview

Time-series forecasting aims to predict future observations based on patterns and dependencies present in historical data.

In this project, an LSTM-based neural network is developed for multivariate time-series forecasting, where multiple weather variables are used simultaneously as input.

The model learns temporal patterns from historical observations and predicts the values of multiple weather variables at the next time step.

The project also incorporates time-series cross-validation and hyperparameter optimization to improve model evaluation and identify a suitable LSTM configuration.

---

## Objectives

The main objectives of this project are:

- Develop an LSTM-based model for multivariate time-series forecasting.
- Capture temporal dependencies among multiple weather variables.
- Preprocess and normalize time-series data.
- Generate sequential input-output samples using a sliding window.
- Evaluate the model using MAE and RMSE.
- Apply time-series cross-validation.
- Perform hyperparameter optimization using Keras Tuner.
- Compare actual and predicted values through visualization.
- Save the trained model for future use.

---

## Dataset

This project uses the Jena Climate Dataset, which contains weather measurements recorded at regular time intervals.

The dataset contains multiple atmospheric and weather-related variables collected over several years.

### Selected Features

- Temperature (T)
- Atmospheric Pressure (p)
- Relative Humidity (rh)
- Maximum Vapor Pressure (VPmax)
- Actual Vapor Pressure (VPact)
- Air Density (rho)
- Wind Speed (wv)
- Maximum Wind Speed (max. wv)

---

## Project Workflow

The complete workflow of the project is:

Jena Climate Dataset
        ↓
Data Loading
        ↓
Data Inspection and Cleaning
        ↓
Feature Selection
        ↓
Chronological Train/Test Split
        ↓
Feature Scaling
        ↓
Sequence Generation
        ↓
LSTM Model Development
        ↓
Model Training
        ↓
Time-Series Cross-Validation
        ↓
Hyperparameter Optimization
        ↓
Final Model Training
        ↓
MAE and RMSE Evaluation
        ↓
Actual vs Predicted Visualization
        ↓
Model Saving

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Convert the date-time column into a proper datetime format.
3. Sort observations chronologically.
4. Select relevant weather variables.
5. Divide the dataset into training and testing sets while preserving temporal order.
6. Apply Min-Max scaling to normalize the features.
7. Convert the continuous time-series into supervised learning sequences.

### Avoiding Data Leakage

The scaler is fitted only on the training data and subsequently applied to the testing data.

This prevents information from the future test set from influencing the training process.

---

## Sequence Generation

LSTM models require sequential input data.

A sliding-window approach is used to convert the time-series data into sequences.

The model uses the previous 24 time steps to predict the next time step.

### Input

24 previous time steps × 8 weather features

### Output

Next time step × 8 weather features

Therefore, the LSTM learns temporal relationships across multiple weather variables simultaneously.

---

## LSTM Model

Long Short-Term Memory (LSTM) is a type of recurrent neural network designed to learn dependencies in sequential data.

The model architecture consists of:

Input Sequence
        ↓
LSTM Layer
        ↓
Dropout Layer
        ↓
LSTM Layer
        ↓
Dropout Layer
        ↓
Dense Output Layer
        ↓
Predicted Weather Variables

The LSTM layers learn temporal patterns, while dropout is used to reduce overfitting.

---

## Model Configuration

The initial LSTM model uses:

- LSTM units: 64 and 32
- Dropout rate: 0.2
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Batch size: 64
- Maximum epochs: 30
- Early stopping based on validation loss

Early stopping is used to stop training when the validation performance stops improving.

---

## Time-Series Cross-Validation

Traditional random K-fold cross-validation is not appropriate for time-series data because it can mix past and future observations.

Instead, TimeSeriesSplit is used.

The data is divided chronologically into multiple training and validation folds.

Conceptually:

Fold 1:
Training → Validation

Fold 2:
Training → Training → Validation

Fold 3:
Training → Training → Training → Validation

This approach better represents the real-world forecasting scenario where future observations are predicted using past observations.

---

## Hyperparameter Optimization

Hyperparameter optimization is performed using Keras Tuner.

The following hyperparameters are optimized:

- Number of LSTM units
- Dropout rate
- Learning rate

The search space includes:

- LSTM units: 32 to 128
- Dropout rate: 0.1 to 0.4
- Learning rate: 0.001, 0.0005, and 0.0001

The objective of the hyperparameter search is to minimize validation loss.

The best-performing configuration is then used to train the final model.

---

## Evaluation Metrics

Two primary metrics are used to evaluate forecasting performance.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

MAE = (1/n) × Σ |yᵢ - ŷᵢ|

A lower MAE indicates better prediction accuracy.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]

RMSE gives greater importance to larger prediction errors.

A lower RMSE indicates better model performance.

---

## Results

The final model is evaluated on the unseen test dataset using MAE and RMSE.

### Overall Performance

The actual performance values generated by the final model will be added here after completing model training and hyperparameter optimization.

| Metric | Value |
|--------|-------|
| MAE | To be updated |
| RMSE | To be updated |

The notebook also calculates MAE and RMSE separately for each weather variable.

---

## Visualization

The project generates several visualizations, including:

### 1. Weather Variable Trends

Visualization of selected weather variables over time to understand temporal patterns.

### 2. Training and Validation Loss

Training and validation loss curves are plotted to analyze model learning and identify potential overfitting.

### 3. Actual vs Predicted Values

The final model predictions are compared with actual observations to visually evaluate forecasting performance.

An actual-versus-predicted temperature plot is also generated as an example of the forecasting results.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Keras Tuner
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

---

## Repository Structure

multivariate-time-series-forecasting-lstm/
│
├── LSTM_Forecasting.ipynb
├── README.md
└── .gitignore

---

## How to Run the Project

### Using Google Colab

1. Open LSTM_Forecasting.ipynb in Google Colab.
2. Install the required Python libraries.
3. Run the notebook cells sequentially.
4. The Jena Climate Dataset will be downloaded automatically.
5. Perform data preprocessing and sequence generation.
6. Train the LSTM model.
7. Perform time-series cross-validation.
8. Run hyperparameter optimization.
9. Train the final model using the best hyperparameters.
10. Evaluate the model using MAE and RMSE.
11. Visualize actual and predicted values.
12. Save the trained model.

### Running Locally

Clone the repository:

git clone https://github.com/YOUR_USERNAME/multivariate-time-series-forecasting-lstm.git

Move into the project directory:

cd multivariate-time-series-forecasting-lstm

Install the required dependencies:

pip install numpy pandas matplotlib scikit-learn tensorflow keras-tuner

Open the notebook using Jupyter Notebook or JupyterLab and run LSTM_Forecasting.ipynb.

---

## Model Saving

The trained LSTM model can be saved in Keras format for future use.

The saved model can be loaded later without retraining the entire network.

---

## Key Outcomes

- Developed an LSTM-based model for multivariate time-series forecasting.
- Captured temporal dependencies using sequential input data.
- Processed multiple weather variables simultaneously.
- Applied chronological train-test splitting to preserve temporal structure.
- Used Min-Max scaling for feature normalization.
- Generated sequential data using a sliding-window approach.
- Applied time-series cross-validation.
- Performed hyperparameter optimization using Keras Tuner.
- Evaluated forecasting performance using MAE and RMSE.
- Compared actual and predicted weather observations.
- Saved the trained LSTM model for future use.

---

## Key Learning

This project demonstrates how recurrent neural networks, particularly LSTMs, can be applied to multivariate time-series forecasting problems.

It also highlights the importance of preserving temporal order during data splitting and validation, as well as the role of hyperparameter optimization in improving deep learning model performance.

---

## Future Improvements

Possible extensions of this project include:

- Forecasting multiple future time steps instead of only the next time step.
- Comparing LSTM with GRU and traditional time-series forecasting models.
- Implementing stacked and bidirectional LSTM architectures.
- Performing more extensive hyperparameter optimization.
- Adding attention mechanisms.
- Using walk-forward validation for more realistic forecasting evaluation.
- Deploying the trained model as a web application or API.
- Incorporating additional weather variables.

---

## Author

Pulkit

M.Sc. Mathematics
Indian Institute of Technology Kharagpur

---

## Project Highlights

Multivariate Time-Series Forecasting
+
LSTM Deep Learning
+
Time-Series Cross-Validation
+
Hyperparameter Optimization
+
MAE / RMSE Evaluation

---

## License

This project is intended for educational, academic, and portfolio purposes.
