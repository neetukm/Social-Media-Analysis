
# Social Media Analysis

Analyze Instagram influencer data and predict engagement using interactive dashboards and machine learning.

![Dashboard Screenshot](./Screenshot%202025-06-02%20113522.png)

## Overview

This project provides tools and dashboards to explore Instagram influencer data, visualize key trends, and predict engagement levels using a trained machine learning model. The app is built on Streamlit and uses Jupyter Notebooks for data analysis and model training.


## Table of Contents
- [Go to the Dashboard](https://social-media-analysis1.streamlit.app/)
-  [Project Overview Dashboard](https://social-media-analysis1.streamlit.app/)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](http://localhost:8502/)
- [License](#license)
## Features

- **Interactive Dashboard:** Filter influencers by category, country, and follower range.
- **Visual Analytics:** View top influencer categories, countries, and follower/engagement patterns.
- **Engagement Prediction:** Enter influencer details to predict engagement level with a logistic regression model.
- **Jupyter Notebooks:** Data cleaning, EDA, and model training workflows.
- **Downloadable Data:** Export filtered influencer data as CSV.

## File Structure

- `main.py` – Streamlit app for dashboard and prediction UI.
- `notebooks/` – Jupyter Notebooks for data analysis and modeling.
  - `Social_media_analysis.ipynb` – EDA and insights.
  - `model_train.ipynb` – Model training.
  - CSV datasets for input and cleaned data.
- `logistic_regression_model.pkl` – Pre-trained model for engagement prediction.
- `Instagramfinall.pbix` – Power BI dashboard.
- `image/` – (Directory for visual assets.)
- `Screenshot 2025-06-02 113522.png` – Sample dashboard screenshot.

## Dataset
The dataset includes Instagram influencers' profiles, engagement metrics, audience countries, and categories. It is available in the `notebooks/` directory.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/neetukm/Social-Media-Analysis.git
    cd Social-Media-Analysis
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

## Usage

- **Explore Data:** Use the dashboard filters to analyze influencer segments.
- **Predict Engagement:** Fill influencer details and get engagement predictions.
- **Jupyter Notebooks:** Open and run notebooks in `notebooks/` for analysis and model retraining.

## Results

The app provides visual insights on:
- Top influencer categories and countries.
- Correlations between followers, engagement, and authenticity.
- Machine learning model predictions with confidence scores.


![Screenshot 2025-06-01 163133](https://github.com/user-attachments/assets/9371bfd7-686b-4c86-97fa-87c928be18f7)


## License

MIT License.

---

**Author:** [neetukm](https://github.com/neetukm)
