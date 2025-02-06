# Modern ML Course: Final Project Overview

**Context**
Lending Club is a peer-to-peer lending platform that allows borrowers to obtain loans directly from individual and institutional investors. The company, which was founded in 2007, acts as a middleman, connecting borrowers with lenders, and charging fees for its services. Loans issued through Lending Club can be used for a variety of purposes, including debt consolidation, home improvement, and small business expenses. The platform is available to borrowers and investors in the United States.

**About the dataset**
The Lending Club loan dataset is a collection of information on loans issued through the Lending Club platform. The dataset includes information on the borrower, such as credit score, income, and employment history, as well as details about the loan, including the loan amount, interest rate, and loan status. The dataset also includes information on the loan's purpose and the borrower's credit history.

---630545

## Objective

You will build a binary classification model to predict the probability of default (PD) for loan applications, using the Lending Club dataset. This is structured as a Kaggle-style competition where:

1. You have access to a **training dataset** (covering loans from 2007 to 2017), including both features and the target.  
2. You have access to a **test dataset** (covering loans from 2018) with features but no public target (the true outcomes for the test set are hidden).

Download Link: https://drive.google.com/drive/folders/1NVQtamDRuGuAIZCRfidvnlSPH1-xLGlG?usp=sharing

Your task is to:

1. **Train** a model on the provided data.  
2. **Submit** predicted probabilities of default for the test set in parquet format.
   1. Put your team predictions [here](https://drive.google.com/drive/folders/1qxjAQ22yebmX-nohalfO0m28_oFinPTH?usp=sharing) with the following format: team_name.parquet
3. **Propose** a credit approval policy based on expected losses.
4. Use all the course knowledge to build the best model possible
---

## Dataset

- **Train Dataset**: 2007–2017 data (public target).  
- **Test Dataset**: 2018 data (hidden target).

You can download both datasets from the provided Google Drive link: [dataset](https://drive.google.com/drive/folders/1NVQtamDRuGuAIZCRfidvnlSPH1-xLGlG?usp=sharing)
A data dictionary is available here:  
[Model Catalog (Variable Descriptions)](https://docs.google.com/spreadsheets/d/14FaRVNdObbYPskGK5UF_MmNW9d3WMjF3biu9R0V2zzw)

---

## Model Requirements

1. **Feature Selection**  
   - Use only variables relevant to the loan application (the dataset contains many columns with future information—exclude those).  
   - You may create additional features through feature engineering.  
   - You may also incorporate external features (e.g., macroeconomic indicators, city/region demographics).

2. **Training & Validation**  
   - Since this will work like a Kaggle competition, you will not have access to the true labels for the test set.  
   - Develop a validation strategy (train/validation split or K-fold) using the training data to tune and evaluate your model.

3. **Predictions & Output**  
   - Generate **probabilities** of default as your final output (in parquet format).  
   - The evaluation metric is **log loss**.

---

## Loss Calculation & Credit Policy

Beyond predicting PD, you must propose a policy to decide which customers to approve or reject. This policy should be based on **expected losses**, defined as:

$$
\text{Losses}(x) = \text{PD}(x) \times \text{EAD}(x) \times \text{LGD}(x).
$$

- **PD** (Probability of Default) will come from your ML model.  
- **EAD** (Exposure at Default) and **LGD** (Loss Given Default) can be estimated from an out-of-sample calibration (or validation) set within the training data. They can be:
  - Modeled using regression or other approaches.  
  - Estimated as averages or heuristic values.

Below is a Python-like pseudocode snippet demonstrating how you can approach EAD and LGD calculation within a calibration dataset (`calibration_df`):

```python
# Total amount owed on the loan (principal + interest)
calibration_df['principal'] = calibration_df.loan_amnt * (1 + calibration_df.int_rate)

# Exposure at Default (EAD): portion of the total owed that remains unpaid
calibration_df['ead'] = calibration_df['principal'] - calibration_df.total_pymnt
calibration_df['ead_ratio'] = calibration_df['ead'] / calibration_df['principal']

# Loss Given Default (LGD): 1 - recoveries ratio
calibration_df['lgd_ratio'] = 1 - (calibration_df['recoveries'] / calibration_df['ead'])

# Final Losses
calibration_df['losses'] = (
    calibration_df['principal'] 
    - calibration_df.total_pymnt 
    + calibration_df['recoveries']
)
```

This youtube lesson could help on this step: [Criando políticas de crédito com um modelo de Machine Learning](https://www.youtube.com/watch?v=651SAEG7Lkw&t=1s)