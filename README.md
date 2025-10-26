# 🚀 Hazardous Asteroid Classification System

### 🛰️ CBIT Hacktoberfest 2025 — Project: Data-Driven Classification of Hazardous Asteroids

This project leverages **Machine Learning** to classify asteroids as **Hazardous** or **Non-Hazardous** based on their physical and orbital parameters.  
It uses a **data-driven approach** to predict potential threats to Earth by analyzing attributes like velocity, distance, and eccentricity.

---

## 🌌 Overview

The goal of this project is to **build an intelligent classifier** that can identify potentially hazardous asteroids using NASA-like dataset parameters.

We trained, compared, and evaluated multiple ML models, visualized results, and exported the **best-performing model** for real-time predictions on unseen data.

---

## 🧠 Features

✅ Trains and evaluates multiple ML models:
- Logistic Regression  
- Decision Tree  
- Random Forest 🌲  
- Gradient Boosting  
- Support Vector Machine (SVM)

✅ Performs:
- Exploratory Data Analysis (EDA)  
- Feature Engineering (Velocity Risk, Proximity Risk, etc.)  
- Model Comparison and Visualization  
- ROC and Confusion Matrix plots  
- Automated Saving of the Best Model  

✅ Includes:
- **`predict_newData.py`** for command-line predictions using trained model  
- **Pre-saved artifacts** (`.pkl` files, plots, metrics)

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 🐍 |
| **Libraries** | pandas, numpy, matplotlib, seaborn, scikit-learn, joblib |
| **ML Models** | RandomForest, GradientBoosting, SVM, etc. |
| **Dataset** | `dataset.csv` (24,000+ asteroid records) |
| **Visualization** | Seaborn & Matplotlib |

---

## 🧪 Dataset

- **File:** `dataset.csv`
- **Size:** ~24,000 records  
- **Target column:** `Hazardous`
- **Features include:**
  - Relative Velocity (km/s)
  - Miss Distance (Astronomical Units)
  - Eccentricity
  - Semi-Major Axis
  - Perihelion Distance
  - Aphelion Distance
  - Inclination
  - and more...

---

## ⚙️ Project Workflow

1️⃣ **Data Loading & Exploration**  
   - Read dataset  
   - Check structure, data types, and missing values  

2️⃣ **Data Preprocessing**  
   - Handle missing data  
   - Convert categorical targets to binary (0/1)  

3️⃣ **Feature Engineering**  
   - Velocity Risk  
   - Proximity Risk  
   - Combined Risk Score  
   - Eccentricity-based Risk  

4️⃣ **Model Training & Evaluation**  
   - Train multiple classifiers  
   - Evaluate using Accuracy, Precision, Recall, F1, ROC-AUC  
   - Select best-performing model  

5️⃣ **Visualization & Export**  
   - Save comparison plots, ROC curves, confusion matrices  
   - Export model and scaler using `joblib`

---

## 📊 Model Performance Summary

The following table shows the performance metrics of all trained models on the asteroid classification dataset:

The table below summarizes the performance of all trained models on the asteroid classification dataset:

| 🧠 Model              | 🎯 Accuracy | 🎯 Precision | 🔁 Recall | ⚖️ F1-Score | 📈 ROC-AUC | ⚠️ False Negatives |
|:----------------------|:-----------:|:------------:|:---------:|:------------:|:----------:|:------------------:|
| 🟩 **Random Forest**  | **0.8401**  | **0.5625**   | 0.0612    | 0.1104       | **0.7135** | 138                |
| ⚙️ SVM                | 0.8379      | 0.0000       | 0.0000    | 0.0000       | 0.5969     | 147                |
| 📉 Logistic Regression| 0.8346      | 0.2857       | 0.0136    | 0.0260       | 0.5827     | 145                |
| 🚀 Gradient Boosting  | 0.8346      | 0.3846       | 0.0340    | 0.0625       | 0.6892     | 142                |
| 🌲 Decision Tree      | 0.7872      | 0.2788       | **0.1973**| **0.2311**   | 0.5748     | **118**            |

---


### 🏆 Insights

- **Best Overall Model:** `Random Forest`  
  Achieved the **highest accuracy (84%)** and **best ROC-AUC (0.71)** — making it the most reliable classifier for this dataset.

- **Decision Tree** achieved the **highest recall (0.1973)**, meaning it identified more hazardous asteroids, though with lower precision.

- **SVM** and **Logistic Regression** struggled due to nonlinear feature relationships and data imbalance.

---

### 🪐 Summary

> **Random Forest Classifier** was chosen as the **final deployed model**, offering a strong balance between accuracy, robustness, and interpretability.

---

## 💾 Generated Files

| File | Description |
|------|-------------|
| `best_asteroid_classifier.pkl` | Trained ML model |
| `feature_scaler.pkl` | StandardScaler used for feature normalization |
| `feature_names.pkl` | List of model input features |
| `confusion_matrix.png` | Visualization of model predictions |
| `roc_curve.png` | ROC curve showing model performance |
| `model_comparison.png` | Model comparison chart |
| `feature_importance.png` | Feature importance ranking |
| `model_comparison_results.csv` | Detailed model metrics |
| `feature_importance_results.csv` | Ranked feature importances |

---

## 💻 Predict on New Data

You can make predictions directly from the terminal using:

```bash
python predict_newData.py
 
```
 
 
 
 
 
 
# HTF25-Team-028

## GitHub submission guide

In this Readme, you will find a guide on how to fork this Repository, add files to it, and make a pull request to contribute your changes.

<details open>
<summary><h3>1. Login to your GitHub Account</h3></summary>
<br>
<p>Go to <a href="https://github.com">github.com</a> to log in.</p>
<ul>
   <li>Open the <a href="https://github.com/cbitosc/HTF25-Team-028">current repo</a> in a new tab.</li>
   <li>Perform all operations in the newly opened tab, and follow the current tab for instructions.</li>
</ul>
</details>

<details open>
<summary><h3>2. Fork the Repository</h3></summary>
<br>
<p align="center">
  <img src="fork.jpeg" alt="Fork the Repository" height="300">
</p>
<ul>
 <li>In the newly opened tab, on the top-right corner, click on <b>Fork</b>.</li>
 <li>Enter the <b>Repository Name</b> as <b>HTF25-Team-028</b>.</li>
 <li>Then click <b>Create Fork</b>, leaving all other fields as default.</li>
 <li>After a few moments, you can view your forked repo.</li>
</ul>
</details>

<details open>
<summary><h3>3. Clone your Repository</h3></summary>
<br>
<ul>
 <li>Click on <b>Code</b> and from the dropdown menu copy your <b>web URL</b> of your forked repository.</li>
 <li>Now open terminal on your local machine.</li>
 <li>Run this command to clone the repo:</li>
<pre><code>git clone https://github.com/your-username/HTF25-Team-028.git</code></pre>
</ul>
</details>

<details open>
<summary><h3>4. Adding files to the Repository</h3></summary>
<br>
<ul>
 <li>While doing it for the first time, create a new branch for your changes:</li>
<pre><code>git checkout -b branch-name</code></pre>
 <li>Add your files or make modifications to existing files.</li>
 <li>Stage your changes:</li>
<pre><code>git add .</code></pre>
 <li>Commit your changes:</li>
<pre><code>git commit -m "Descriptive commit message"</code></pre>
 <li>Push your branch to your fork:</li>
<pre><code>git push origin branch-name</code></pre>
</ul>
</details>

<details open>
<summary><h3>5. Create a Pull Request</h3></summary>
<br>
<ul>
 <li>Click on the <b>Contribute</b> button in your fork and choose <b>Open Pull Request</b>.</li>
 <li>Leave all fields as default, then click <b>Create Pull Request</b>.</li>
 <li>Wait a few moments; your PR is now submitted.</li>
</ul>
</details>

## Thanks for participating!
