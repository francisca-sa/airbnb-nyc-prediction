### Airbnb NYC 2019: Data Analysis & Predictive Modeling 🗽📊

[![Python](https://img.shields.io/badge/Python-3.13.1-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3.2-blue.svg)](https://www.r-project.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)

#### 📌 About the Project
This repository contains a complete Data Science project developed under the "Paradigmas de Aprendizagem Automática" (Machine Learning Paradigms) course as part of the Data Science Bachelor's program at the University of Minho. Following the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology, we analyzed an Airbnb New York City dataset to extract actionable insights for hosts and investors.

The project tackles three main machine learning challenges:
1. **Regression:** Predicting the price per night of a property.
2. **Classification:** Predicting the success rate of a listing (Low, Medium, High) based on annual availability.
3. **Clustering:** Segmenting properties into distinct market profiles.

#### ⚙️ Data Pipeline & Feature Engineering
Starting with raw data from the [New York City Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), the pipeline included:
* **Data Cleaning & Preparation:** Handling missing values, removing extreme outliers using statistical methods (Percentiles and Z-scores), and encoding categorical variables.
* **Data Integration:** Enriched the dataset by integrating external geographical data, including [crime rates](https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Incident-Level-Data-Ye/mv4k-y93f/about_data) (minor, moderate, serious) per neighborhood and proximity to tourist attractions (number of attractions within a 2km radius).

#### 🚀 Models & Results

#### 1. Price Prediction (Regression)
* **Goal:** Estimate the optimal price per night.
* **Models Tested:** Decision Trees vs. Neural Networks (evaluated via Holdout 70/30 and 5-fold CV).
* **Best Model:** Neural Network (Scenario 5).
* **Performance:** Achieved a highly stable Normalized Mean Absolute Error (NMAE) of ~2.06%.

#### 2. Success Rating (Classification)
* **Goal:** Classify the expected success/occupancy of a listing (High, Medium, Low).
* **Models Tested:** Decision Trees vs. Neural Networks (evaluated via 10-fold CV).
* **Best Model:** Decision Tree (Scenario 10).
* **Performance:** Reached an accuracy of 76.3% and an F1-Score of 0.758, showing strong capability in identifying interaction patterns between neighborhoods, prices, and minimum stay requirements.

#### 3. Market Segmentation (Clustering)
* **Goal:** Identify underlying patterns in the Airbnb market.
* **Method:** K-Means algorithm evaluated with the Elbow Method and Silhouette Score.
* **Result:** Segmented the properties into 2 distinct clusters (Economic vs. Premium) with an excellent Silhouette Score of ~0.85.

#### 💻 Web Interfaces (Deployment)
To bridge the gap between technical modeling and end-user applicability, we deployed the models using Flask. Three distinct, user-friendly web interfaces were built:

<img width="300" height="350" alt="Screenshot do Price Predictor" src="https://github.com/user-attachments/assets/bd5980c5-6113-4344-b714-3675fdd92103" /> <img width="300" height="300" alt="Screenshot do Success Rater" src="https://github.com/user-attachments/assets/8aee9029-3d0e-44be-bfa8-9cb9513f862c" /> <img width="300" height="250" alt="Screenshot do Profile Analyzer" src="https://github.com/user-attachments/assets/935ab5c3-c529-4341-977d-aaa2084ae0d5" />

1. **Price Predictor:** Interface to estimate the optimal listing price (Regression Model).
2. **Success Rater:** Interface to evaluate the potential occupancy performance (Classification Model).
3. **Profile Analyzer:** Interface to automatically categorize a property (Clustering Model).

### How the App Works under the Hood
To ensure the web interfaces function seamlessly, the `app/` directory includes dedicated Python scripts (`regressao.py`, `classificacao.py`, `clustering.py`). These files are imported by the main Flask applications to automatically train and load the chosen models (with the best-performing scenarios) whenever the user interacts with the web interfaces.

#### 📁 Repository Structure
```text
airbnb-nyc-prediction/
│
├── data/
│   ├── AB_NYC_2019.csv                   # Original Kaggle dataset
│   ├── Airbnb_NYC_2019_VF.csv            # Cleaned and enriched dataset
|   └── crimes.csv
│
├── notebooks/                            # Jupyter Notebooks and R scripts
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation_init.ipynb
│   ├── 03_outlier_analysis.ipynb
│   ├── 04_data_preparation_final.ipynb
│   ├── 05_modeling_classification.ipynb
│   ├── 06_modeling_clustering.ipynb
│   └── 07_modeling_regression.ipynb
│
├── app/                                  # Flask web interfaces deployment
│   ├── app.py                            # Main app for Price Prediction
│   ├── app_c.py                          # Main app for Success Classification
│   ├── app_k.py                          # Main app for Property Clustering
│   ├── classificacao.py                  # Support script for classification
│   ├── clustering.py                     # Support script for clustering
│   ├── regressao.py                      # Support script for regression
│   ├── label_encoder_classificacao.joblib# Saved encoder
│   ├── limite_preco.joblib               # Saved price limits
│   ├── modelo.joblib                     # Saved regression model
│   ├── modelo_classificacao.joblib       # Saved classification model
│   └── templates/        
│       ├── index.html                    # Interface for regression
│       ├── index_c.html                  # Interface for classification
│       └── index_k.html                  # Interface for clustering
│
├── docs/                                 # Project reports
│   └── relatorio.pdf                     # Report 
│
└── README.md

````
#### Authors
* Dean Kuijf
* Francisca Machado | [GitHub](https://github.com/francisca-sa/)
* Gabriela Durães

_Project developed for the Data Science Bachelor's program at the University of Minho (2024/2025)._
