
## 📊 Emotion Classification Using LinearSVC  
**(NLP Pipeline with NLTK + Scikit-learn)**

This project applies Natural Language Processing and Machine Learning to classify text statements into emotional categories like **Anxiety**, **Depression**, **Relief**, and more. It uses a combination of **NLTK** for preprocessing and **scikit-learn** for modeling and evaluation.

---

### 📁 Dataset

This project uses the dataset:  
🔗 [Sentiment Analysis for Mental Health – Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)

#### 💬 About the Dataset:
- Contains thousands of labeled statements collected from mental health-related sources.
- Each entry has a `statement` (text) and a `status` (emotion category).
- The dataset focuses on **mental health sentiment** and reflects a wide range of human emotions like:
  - `Anxiety`
  - `Depression`
  - `Loneliness`
  - `Optimism`
  - `Gratitude`
  - `Relief`

This makes the dataset ideal for building models that help understand emotional expression in real-world mental health contexts.

---

### 🧠 What This Code Does

1. Prepares and cleans text using **NLTK** tokenization and stopwords.
2. Converts text into numerical features using **CountVectorizer**.
3. Trains a **Linear Support Vector Classifier (LinearSVC)**.
4. Evaluates the model using accuracy, precision, recall, F1-score.
5. Uses **RandomizedSearchCV** to optimize the `C` hyperparameter.
6. Outputs the best model and its performance metrics.

---

### 📈 What This Code Tells Us About the Data

- There are **learnable patterns in the language** used to describe emotional states.
- Even a simple model like `LinearSVC` achieves **~75% accuracy**, showing that:
  - Words and phrases strongly correlate with specific emotions.
  - Emotional text can be **quantified and predicted** with solid performance.
- The model generalizes well across multiple emotional labels, especially after tuning `C` to `0.1`.

In other words:  
> The language people use when expressing mental health concerns contains enough signal for a machine learning model to **recognize and classify emotions with meaningful accuracy**.

---

### ⚙️ Pipeline Breakdown

#### 1. **NLTK Setup**
```python
nltk.download('punkt')
nltk.download('stopwords')
```

#### 2. **Data Preparation**
```python
data = pd.read_csv('Combined Data.csv')
X = data['statement'].fillna("").astype(str)
y = data['status']
```

#### 3. **Vectorization**
```python
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
```

#### 4. **Training LinearSVC**
```python
clf = LinearSVC()
clf.fit(X_train_features, y_train)
```

#### 5. **Evaluation**
```python
accuracy_score(y_test, y_pred)
precision_score(...)
confusion_matrix(...)
```

#### 6. **Hyperparameter Tuning**
```python
RandomizedSearchCV(..., param_distributions={'C': [...]})
```

---

### ✅ Model Results

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 74.5%     |
| Precision    | 74.1%     |
| Recall       | 74.5%     |
| F1 Score     | 74.2%     |
| Best C Value | 0.1       |
| Best CV Score| 75.2%     |

---

---

### 🛠️ Getting Started

To run the code:
1. Download the dataset from Kaggle.
2. Install required libraries:
```bash
pip install nltk scikit-learn pandas
```
3. Upload the CSV file and run the notebook.

---

