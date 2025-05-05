# CODSOFT - Machine Learning Internship Projects

Welcome to **CODSOFT**, a repository showcasing my work for the Machine Learning internship. Here, I will upload solutions for multiple tasks related to machine learning, showcasing my skills in different areas like text classification, fraud detection, and churn prediction.

## Completed Tasks

### Task 1: SMS Spam Detection
- **Description**: This project builds an AI model that classifies SMS messages as either **spam** or **ham** (legitimate).
- **Techniques Used**: TF-IDF for text vectorization, Naive Bayes and Logistic Regression for classification.
- **Dataset**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Model Evaluation**: The model was evaluated using accuracy, confusion matrix, and visualization techniques.
  
   **Subfolder**: [task1_sms_spam_detection](task1_sms_spam_detection/)

### Task 2: Credit Card Fraud Detection
- **Description**: A machine learning model that detects fraudulent credit card transactions based on a dataset containing transaction information.
- **Techniques Used**: Logistic Regression, Decision Trees, and Random Forest for classification.
- **Dataset**: [Credit Card Fraud Detection Dataset]([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/kartik2112/fraud-detection))
- **Model Evaluation**: Accuracy, Precision, Recall, and F1-score were used to evaluate model performance.
  
   **Subfolder**: [task2_credit_card_fraud_detection](task2_credit_card_fraud_detection/)

### Task 3: Customer Churn Prediction
- **Description**: The model predicts customer churn for a subscription-based service using historical data, including usage behavior and demographics.
- **Techniques Used**: Logistic Regression, Random Forests, and Gradient Boosting.
- **Dataset**: [Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Model Evaluation**: The model’s performance was evaluated using accuracy and various classification metrics.
  
   **Subfolder**: [task3_customer_churn_prediction](task3_customer_churn_prediction/)

## How to Use This Repository

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/CODSOFT.git
    cd CODSOFT
    ```

2. **Install necessary dependencies**:

    The required libraries for each task are listed in the respective subfolders. You can install dependencies using the `requirements.txt` files or directly with `pip`:

    ```bash
    pip install -r task1_sms_spam_detection/requirements.txt
    pip install -r task2_credit_card_fraud_detection/requirements.txt
    pip install -r task3_customer_churn_prediction/requirements.txt
    ```

3. **Run the scripts**:
   
    To run a task, go to the corresponding subfolder and execute the script:

    - For Task 1: SMS Spam Detection
      ```bash
      python task1_sms_spam_detection/sms_spam_classification.py
      ```
    - For Task 2: Credit Card Fraud Detection
      ```bash
      python task2_credit_card_fraud_detection/credit_card_fraud_detection.py
      ```
    - For Task 3: Customer Churn Prediction
      ```bash
      python task3_customer_churn_prediction/customer_churn_prediction.py
      ```

## License

MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

* [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* [Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Thanks to various online resources for tutorials and support.

