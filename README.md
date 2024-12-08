# Concrete Strength Prediction using Machine Learning

This project demonstrates a complete machine learning pipeline for predicting the compressive strength of concrete based on several features. The pipeline includes data ingestion, preprocessing, model training, and deployment through a **Streamlit** web application. The app allows users to input concrete properties via sliders and receive a predicted concrete strength.

This **README** provides step-by-step instructions for setting up and running the project, including details for each stage of the pipeline: ingestion, processing, training, and deployment.

---

## Table of Contents

- [Project Setup](#project-setup)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data Ingestion](#data-ingestion)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [How to Train a New Model](#how-to-train-a-new-model)
- [License](#license)

---

## Project Setup

1. **Clone the Repository**:
   Start by cloning the project repository to your local machine.
   ```bash
   git clone https://dagshub.com/Agam-Patel-DS/concrete-strength-prediction.git
   cd concrete-strength-prediction
   ```

2. **Install Dependencies**:
   Make sure you have `pip` installed, then create a virtual environment and install the required dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows

   pip install -r requirements.txt
   pip install -v .
   ```

---

## Project Structure
You can find the Project structure in `template.py`.

## Dependencies

This project requires the following Python packages:

- `streamlit`: For creating the web application.
- `joblib`: To load and save the pre-trained model.
- `scikit-learn`: For machine learning and model training.
- `pandas` and `numpy`: For data manipulation.
- `matplotlib`: For visualizations (optional).

You can install all dependencies at once with:

```bash
pip install -r requirements.txt
```

---

## Data Ingestion

The first step is to load the raw data into the system. We can use a CSV file or any other suitable format for the data. Ingestion can be handled by reading the data into a dataframe or loading from a database, I used a **Kaggle Dataset** for this project.

---

## Data Preprocessing

After ingestion, the next step is to clean and preprocess the data. This typically includes tasks such as:

- **Handling Missing Values**: Filling or removing missing data points.
- **Feature Scaling**: Standardizing or normalizing numerical features.
- **Feature Engineering**: Adding, removing, or transforming features to make the model training more effective.
- **Splitting Data**: Dividing the dataset into training and testing sets.

Preprocessing will be done in the `_data_preprocessing.py` script. The resulting cleaned and split data is used for model training.

---

## Model Training

Once the data is preprocessed, the next step is to train the machine learning model. You can choose any suitable regression model, such as:

- **Random Forest Regressor**
- **Gradient Boosting**

The training process includes:
- **Model Initialization**: Selecting a regression model.
- **Model Training**: Fitting the model to the training data.
- **Hyperparameter Tuning**: Optionally tuning the hyperparameters of the model for better performance.
- **Model Evaluation**: Evaluating the model's performance using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or R-squared.

Once the model is trained, it is saved as a `.pkl` file for later use. The model file is placed in the `models/` folder.

---

## Deployment

After training the model, you can deploy it using a **Streamlit** app, which provides a simple and interactive user interface.

### Key Features of the Streamlit App:
- **Model Loading**: The app loads the pre-trained model saved in the `models/` folder.
- **User Input**: Users can provide concrete properties (such as cement, slag, water content, etc.) through sliders.
- **Prediction Output**: After the user inputs the data, the app displays the predicted concrete strength.

To run the app:
1. Ensure the trained model is saved in the `models/` folder.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using:
   ```bash
   streamlit run app.py
   ```
4. The app will be available at `http://localhost:8501`.

---

## How to Train a New Model

If you want to train a new model, follow these steps:

1. **Prepare the Data**: 
   Ensure your dataset is cleaned and preprocessed. The dataset should include columns such as:
   - Cement (kg/m^3)
   - Blast Furnace Slag (kg/m^3)
   - Fly Ash (kg/m^3)
   - Water (kg/m^3)
   - Superplasticizer (kg/m^3)
   - Coarse Aggregate (kg/m^3)
   - Fine Aggregate (kg/m^3)
   - Age (days)

2. **Train the Model**:
   Use any regression model, such as `RandomForestRegressor`, `LinearRegression`, or `XGBoost`. Training is done in the `training.py` script. You can adjust model parameters, add hyperparameter tuning, and evaluate the model performance in training_config.py file.

3. **Save the Model**:
   Once the model is trained, save it in the `models/` folder using `joblib.dump(model, "models/model.pkl")`.

4. **Deploy the Updated Model**:
   After saving the updated model, ensure that the path in the Streamlit app (`app.py`) points to the newly saved model. Run the Streamlit app again to make predictions using the updated model.

---

## License

This project is licensed under the MIT License.