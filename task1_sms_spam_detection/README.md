# SMS Spam Detection

This project aims to build a machine learning model that classifies SMS messages as either **spam** or **ham** (legitimate). The model is trained using a dataset of SMS messages and evaluates text classification techniques to differentiate between spam and non-spam messages.

## Dataset

The **SMS Spam Collection Dataset** contains 5,572 SMS messages, classified as either **spam** or **ham** (legitimate). The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

- **label**: Type of message (spam or ham)
- **message**: The content of the SMS message

## Techniques Used

### Text Preprocessing:
- **Stopword Removal**: Filtering out common words that do not add value to the classification.
- **Text Normalization**: Lowercasing, punctuation removal, and stemming.
  
### Feature Extraction:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used for vectorizing the text data into numerical features that the model can interpret.

### Models:
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **Logistic Regression**: A linear model for binary classification.
- **Support Vector Machines (SVM)**: A supervised learning model for classification.

### Model Evaluation:
- **Accuracy**: Measures the proportion of correct predictions.
- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, and false negatives.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
````

2. **Install required libraries**:

   The dependencies for this project are listed in `requirements.txt`. Install them by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:

   To train and evaluate the SMS Spam Detection model, execute the following script:

   ```bash
   python sms_spam_classification.py
   ```

## Visualizations

* **Spam vs Ham Distribution**: A bar chart that shows the distribution of spam and ham messages.
* **Word Clouds**: Displaying the most frequent words in both spam and ham messages.
* **Confusion Matrix**: Visual representation of classification results for model performance.

## License

MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

* [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* Thanks to the creators of the dataset for making this project possible.

```

---

### Key Sections in This README:

- **Dataset**: Information about the SMS Spam Collection dataset and where to find it.
- **Techniques Used**: Overview of text preprocessing, feature extraction, and the machine learning models used.
- **How to Run**: Steps to clone the repository, install dependencies, and run the model.
- **Visualizations**: Description of the types of visualizations used for model evaluation.
- **License and Acknowledgements**: Proper attribution and the license under which the code is shared.

This README should serve as a clear guide for someone wanting to understand or replicate your SMS Spam Detection task. Let me know if you need any further adjustments!
```

