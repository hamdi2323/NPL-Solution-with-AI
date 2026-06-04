"""
===============================================================
Python script to demonstrate a document classification and topic modelling
===============================================================
This program demonstrate two natural language process tasks i.e. Document Classification
(Supervised Learning)
Part 1: Document Classification (Supervised Learning)
        -TF-IDF Vectorization
        -Train/Test split
        -Multinomial Naive Bayes Classification
        -Accuracy Evaluation
        -Classification Report
        -Interaction Predictions

Part 2: Topic Modelling (Unsupervised Learning)
        -TF-IDF Vectorization
        -Latent Dirichlet Allocation (LDA)
        -Topic Discovery
        -Topic Interpretation
        -Dominant Topic Assignment
Dataset location: files/articles.json, files/topics.json

Requirements:
    pip install scikit-learn
Author: Xamdi Salaad
Date: 04-06-2026
"""
#------------------------------------------------------------
#0.Import required modules
#-----------------------------------------------------------
import json,re,sys
import numpy as np
import pandas as pd

from Pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import warnings
#Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")
