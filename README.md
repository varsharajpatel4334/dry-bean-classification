# Dry Bean Classification Using Machine Learning

### a. Problem Statement

The objective of this project is to classify dry beans into their respective varieties using machine learning. The classification is based on physical and shape-related measurements extracted from bean images. Seven different bean varieties are considered in this problem. Accurate classification can help in automating the identification and sorting of beans based on measurable characteristics.

### b. Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Source Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Rows:** 13,611
- **Features:** 16
- **Target:** `Class`
- **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
- **Missing Values:** No missing values are reported in the dataset.
- **Problem Type:** Multi-class Classification

The dataset contains images of 13,611 dry bean grains. From these images, 16 numerical features were extracted, including 12 dimensional and 4 shape-related features. These measurements are used to identify the seven different bean varieties.

For preprocessing, duplicate records were removed before model training. The dataset was then divided into 80% training data and 20% testing data using stratified train-test splitting. StandardScaler was used for Logistic Regression, KNN and SVM because these algorithms are sensitive to feature scales.

### c. GitHub Repository

GitHub Repository: https://github.com/varsharajpatel4334/dry-bean-classification

### d. Models Used

Five classification algorithms were trained and evaluated on the same test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8952 | 0.9752 | 0.8950 | 0.8952 | 0.8949 | 0.8733 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

#### Observations

| Model | Observation |
|---|---|
| **Logistic Regression** | Logistic Regression performed very well with an accuracy of 91.95%. Standardizing the features helped because the dataset contains measurements having different numerical scales. It also achieved the highest AUC of 0.9935, showing very good ability to distinguish between the bean classes. |
| **Decision Tree** | Decision Tree achieved an accuracy of 89.52%. Its performance was lower than Logistic Regression, KNN and SVM. Restricting the tree depth helped control overfitting, but it may also have limited the model's ability to learn some complex relationships between the bean characteristics. |
| **KNN** | KNN achieved an accuracy of 91.55% and performed close to Logistic Regression and SVM. Feature scaling was especially important for KNN because the algorithm classifies observations based on distances between data points. |
| **Naive Bayes** | Naive Bayes achieved the lowest accuracy of 76.30%. One possible reason is that Gaussian Naive Bayes assumes conditional independence among features within a class. In this dataset, physical properties such as area, perimeter and diameter can be related to each other, making this assumption less suitable. |
| **SVM** | SVM gave the best overall performance with an accuracy of 92.10%, F1-score of 0.9211 and MCC of 0.9045. Scaling the features helped SVM work effectively, and the model was able to produce good decision boundaries between the seven bean varieties. |

### Overall Result

Among the five models evaluated, **SVM produced the best overall classification performance**, achieving an accuracy of 92.10% and MCC of 0.9045. Logistic Regression was a very close second and achieved the highest AUC value of 0.9935. KNN also performed well, while Decision Tree showed moderately lower performance. Naive Bayes produced the lowest overall classification performance on this dataset.

### Live App

Streamlit App:
https://dry-bean-classification-kpwwz9wkxytdrw3zgbm6o5.streamlit.app

The Streamlit application allows the user to:
- Upload the test dataset in CSV format.
- Select any of the five trained classification models.
- Select the target column.
- View Accuracy, AUC, Precision, Recall, F1-score and MCC.
- View the confusion matrix and classification report.
- Compare the performance of all five models on the same test dataset.