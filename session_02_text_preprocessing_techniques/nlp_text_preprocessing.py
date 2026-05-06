"""
Natural Language Processing pipeline for resturant reviews

The preprocessing pipeline includes:
1.Lowercasing
2.Slang and abbreviations normalisation
3.Contraction expansion
4.Repeated character normalisation
5.Emoji removal
6.Punctuation cleaning
7.Tokenisation
8.Stopword removal
9.Optional lemmatisation

Author: Xamdi Salaad
Date: 06-05-2026
"""
#--------------------------------------------------------
# 0.Import the required modules
# --------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List

#--------------------------------------------------------
# 1.Download the required data
# --------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

#--------------------------------------------------------
# 1.Reviews
# --------------------------------------------------------
REVIEWS = [
    "The food was 2die4! Best burgers in town 😋🍔",
    "Service was 10/10, loved the vibe too! Totally recommend this place 🔥👌",
    "2 hrs wait for a table... not worth it. Food was ok, but not great.",
    "Tbh, 4 the price, I expected way better quality. Disappointed.",
    "Great place for brunch! Had the eggs benedict, 1 of my faves 💯🍳",
    "The ambience was gr8! Luvd it :) !!!",
    "Had a blast at this place!! Will come again soon 😋",
    "Food was OK, but could be better, meh...",
    "This pizza was absolutely amazing, best I’ve had!!",
    "Service was horrible... never coming back!",
    "I loved the pasta! But the portion was so small :(",
    "The dessert was soooooo good!! 😍",
    "So disappointed, the steak was overcooked...",
    "Great experience, but the music was a little loud tbh.",
    "Good food, but they forgot my drink. :(",
    "Superb food! Totally worth the price! Will return!",
    "Was okay, nothing special. Meh.",
    "The chicken was so dry, I couldn't finish it :( too bad!",
    "Fantastic, I can't wait to visit again! :) :)",
    "Not worth the price, won't be coming back :( 😔"
]
#--------------------------------------------------------
# 3.Normalisation rules
# --------------------------------------------------------
SLANG_DICT = {
    r'\b2die4\b': 'to die for',
        r'\bgr8\b': 'great',
        r'\bluvd\b': 'loved',
        r'\bfaves\b': 'favourites',
        r'\btbh\b': 'to be honest',
        r'\bthx\b': 'thanks',
        r'\bplz\b': 'please',
        r'\bu\b': 'you',
        r'\b4\b': 'for',
        r'\b2\b': 'to',
        r'\b1\b': 'one',
        r'\b10/10\b': 'perfect',
        r'\bok\b': 'okay',
        r'\bmeh\b': 'mediocre',
        r'\btotally\s*recommend\b': 'highly recommend',
        r'\bambiance\b': 'ambience',  # US to UK spelling
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}