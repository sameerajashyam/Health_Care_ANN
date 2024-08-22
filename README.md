
# Healthcare Data Classification Using Artificial Neural Network (ANN)

## Overview

This project involves building a classification model using an Artificial Neural Network (ANN) to predict healthcare test results based on various patient features such as age, gender, medical condition, blood type, medication, etc.

The model classifies the test results into three categories:
- **Normal**
- **Inconclusive**
- **Abnormal**

## Project Structure

- **healthcare_dataset.csv**: The raw dataset containing patient information.
- **processed_healthcare_data.csv**: The preprocessed dataset ready for model training (not generated in the provided script).
- **model.h5**: The saved ANN model.
- **label_encoder_gender.pkl**: Pickle file for the label encoder used for the `Gender` feature.
- **medical_condition_encoder.pkl**: Pickle file for the one-hot encoder used for the `Medical Condition` feature.
- **admission_type_encoder.pkl**: Pickle file for the one-hot encoder used for the `Admission Type` feature.
- **blood_type_encoder.pkl**: Pickle file for the one-hot encoder used for the `Blood Type` feature.
- **medication_encoder.pkl**: Pickle file for the one-hot encoder used for the `Medication` feature.
- **test_results_encoder.pkl**: Pickle file for the one-hot encoder used for the `Test Results` feature.
- **scaler.pkl**: Pickle file for the StandardScaler used for feature scaling.
- **logs/**: Directory containing TensorBoard logs.

## Data Preprocessing

1. **Feature Selection**: 
   - The dataset includes many irrelevant features such as `Name`, `Date of Admission`, `Doctor`, `Hospital`, etc. These features were dropped.
   
2. **Label Encoding**: 
   - The `Gender` feature was label encoded.

3. **One-Hot Encoding**:
   - Categorical features like `Medical Condition`, `Admission Type`, `Blood Type`, and `Medication` were one-hot encoded to convert them into a numerical format suitable for model training.

4. **Feature Scaling**:
   - The features were scaled using `StandardScaler` to normalize the input data.

## Model Development

1. **Architecture**:
   - The model is a Sequential ANN with the following layers:
     - Input Layer: Corresponding to the number of features after encoding and scaling.
     - 1st Hidden Layer: 64 neurons with ReLU activation.
     - 2nd Hidden Layer: 32 neurons with ReLU activation.
     - 3rd Hidden Layer: 16 neurons with ReLU activation (optional).
     - Output Layer: 3 neurons with Softmax activation (for multi-class classification).

2. **Compilation**:
   - The model was compiled using the Adam optimizer with a learning rate of 0.01 and categorical cross-entropy loss function, suitable for multi-class classification.

3. **Training**:
   - The model was trained with early stopping and TensorBoard callbacks to monitor performance and prevent overfitting.
   - The model was trained for 100 epochs with early stopping based on validation loss with patience set to 10 epochs.

## Usage

1. **Training**:
   - To train the model, you can run the provided Python script. It reads the data, preprocesses it, builds the ANN model, and trains it using the preprocessed data.

2. **Prediction**:
   - After training, the model is saved as `model.h5`. This model can be loaded for making predictions on new data.

3. **Evaluation**:
   - The model can be evaluated on test data, and TensorBoard logs can be used to visualize training metrics.

## Dependencies

- Python 3.x
- Numpy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Pickle

You can install the required packages using:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib pickle-mixin
```

## How to Run

1. **Preprocess the Data**: 
   - Run the script to preprocess the data and encode categorical variables.
   
2. **Train the Model**:
   - The model will be trained, and a TensorBoard log will be generated for performance tracking.

3. **Check the Results**:
   - The final trained model will be saved, and you can use it for predictions.
