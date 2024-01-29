
# SMS Spam Detection Script
## Introduction
This Python script focuses on SMS spam detection using a dataset containing labeled messages as spam or ham. The script uses natural language processing techniques, feature engineering, and a classification model for spam prediction.
## Requirements
Python 3.x
Required Python libraries: pandas, numpy, matplotlib, seaborn, nltk, scikit-learn
## Usage
Install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```
Download NLTK resources:
```bash
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
Run the script:
```bash
python your_script_name.py
```
## Script Overview
The script is divided into several sections:

## Data Loading and Exploration
Reads the SMS dataset from "SMSSpamCollection" and sets column names.
Converts the 'label' column to numeric values (0 for 'ham' and 1 for 'spam').
Visualizes the class distribution of spam and ham messages.
## Feature Engineering
Calculates word count for each message.
Identifies messages containing currency symbols and numbers.
Creates new features based on the above observations

# SMS Spam Detection

This repository contains code for detecting spam SMS messages using machine learning, specifically Multinomial Naive Bayes and Decision Tree classifiers.

## Dataset

The dataset used in this project is the "SMSSpamCollection" dataset. It consists of SMS messages labeled as either "ham" (non-spam) or "spam".

## Overview

The project includes the following steps:

1. Data Exploration: Understanding the dataset, handling imbalanced data, and visualizing key statistics.

2. Feature Engineering: Creating new features such as word count, currency symbols, and numbers in messages.

3. Text Preprocessing: Cleaning and lemmatizing the text data to prepare it for machine learning models.

4. TF-IDF Vectorization: Converting the text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

5. Model Training: Implementing and training two classifiers - Multinomial Naive Bayes and Decision Tree.

6. Evaluation: Assessing the performance of the models using cross-validation, F1 scores, and confusion matrices.

7. Prediction: Creating a function to predict whether a given SMS message is spam or ham.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```
## Predicting New Messages
```bash
sample_message = 'Free entry in 2 a wkly comp...'
if predict_spam(sample_message):
    print('Gotcha! This is a spam message.')
else:
    print('This is a ham (normal) message.')
```
Feel free to experiment with your own messages!

## Results
The project achieves a certain level of accuracy in detecting spam SMS messages. Refer to the classification reports and confusion matrices in the notebook for detailed results
