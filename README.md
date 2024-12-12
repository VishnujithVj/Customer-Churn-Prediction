# Customer-Churn-Prediction

## Project Overview
Customer churn prediction aims to identify customers who are likely to stop using a product or service. This project leverages machine learning to analyze customer data and predict churn probabilities, enabling businesses to take proactive steps to retain customers.

## Features
- **Data Preprocessing**: Handles missing values, encoding categorical variables, and feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizations to understand trends and correlations.
- **Model Training**: Includes Logistic Regression, Random Forest, and XGBoost models.
- **Evaluation**: Metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **Prediction**: Outputs churn probability for new customer data.

## Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Tools**: Jupyter Notebook


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or add the dataset to the `data/` folder.

## Usage
1. **Data Preprocessing**:
   Run the preprocessing script to clean and prepare the data:
   ```bash
   python scripts/preprocess.py
   ```

2. **EDA**:
   Open and run the `eda.ipynb` notebook to visualize data trends.

3. **Model Training**:
   Train models using the `model_training.ipynb` notebook or script:
   ```bash
   python scripts/train_model.py
   ```

4. **Prediction**:
   Use the `predict.py` script to predict churn for new data:
   ```bash
   python scripts/predict.py --input new_data.csv
   ```


## Results
The final accuracy and classification report of the best-performing model are as follows:

### Model: Random Forest
- **Accuracy**: 0.9570
- **Classification Report**:
  ```plaintext
               precision    recall  f1-score   support

           0       0.96      0.95      0.95       551
           1       0.96      0.96      0.96       611

    accuracy                           0.96      1162
   macro avg       0.96      0.96      0.96      1162
weighted avg       0.96      0.96      0.96      1162
  ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements.

## License
This project is licensed under the MIT License.
