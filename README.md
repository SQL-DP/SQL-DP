# SQL-DP
## Introduction
This repository contains the implementations for difficulty prediction task for Structured Query Language (SQL) programming problems.

The main objective of the project is to predict the difficulty of each given SQL programming problem based on its stem and answer (i.e., SQL code).

## SQL-DP Framework
In SQL-DP, semantic features of problem stems and structure features of problem answers in the form of SQL codes are both computed at first, using the NLP and the neural network techniques. Then, these features are used as the input to train a difficulty prediction model with the statistic error rates in tests as the training labels, where the whole modeling do not introduce any experts, some as knowledge labeling. Finally, with the trained model, we can automatic predict the difficulty of each SQL programming problem.

![image](https://github.com/SQL-DP/SQL-DP/blob/main/images/framework.jpg)

As shown in the picture above, SQL-DP framwork includes two modules: Feature Extraction (FE) module and Difficulty Prediction (DP) module. Besides, the Feature Extraction Module includes SQL Text Semantic Fearure Extraction (SQL-TSFE) module and SQL Code Struture Feature Extraction (SQL-CSFE) module.

The SQL-TSFE modul uses NLP technologies (e.g., Word2vec, TF-IDF) to extract text semantic feature from the stem of SQL programming problem.

And The SQL-CSFE module uses TBCNN model to extract the code structure feature from the answer of SQL programming problem.

In the DP module, some machine learning algorithms use the above features to predict the difficulty of SQL programming problems.

## Requirements
### In SQL-TSFE module


### In SQL-CSFE module


### In DP Module


## 

## Data

## Run
