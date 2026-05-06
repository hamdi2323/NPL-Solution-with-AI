# python script to demonstrate stemming with visualization

# --------------------------------------------------------
# 0.Import the required modules
# --------------------------------------------------------

import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# --------------------------------------------------
# 1.Download the required data
# ---------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --------------------------------------------------
# 2.Sample Text to be stemmed
# ---------------------------------------------------
TEXT = """
The researchers were studying the running patterns of various animals.
They observed that faster runners consistently outperformed slower ones.
The studies showed interesting running behaviours.
"""
# initialize the stemmer
stemmer = SnowballStemmer('english')


# --------------------------------------------------
# 3.Text Preparation Function
# ---------------------------------------------------
def preprocess_text(text: str) -> list:
    """

    Tokenize and clean the input text.
    this function converts text to lowercase, remove punctuations, and
    returns: list of cleaned tokens
        :param text: input text
        :return: cleaned tokens
    """

    tokens = word_tokenize(text.lower())

    clean_tokens = [
        re.sub('[^a-z]', ' ', token)
        for token in tokens
    ]
    return [token for token in clean_tokens if token]


# --------------------------------------------------
# 4.Stemming Function
# ---------------------------------------------------
def apply_stemming(tokens: list) -> list:
    """
    Apply stemming to a list of tokens
    :param tokens: list of word tokens
    :return: list of stemmed word tokens
    """
    return [stemmer.stem(token) for token in tokens]


# --------------------------------------------------
# 5.Visualisation Function
# ---------------------------------------------------
def plot_frequencies(original: list, stemmed: list) -> None:
    original_counts = Counter(original)
    stemmed_counts = Counter(stemmed)

    # select top items for clarity
    top_original = dict(original_counts.most_common(5))
    top_stemmed = dict(stemmed_counts.most_common(5))

    # plot original word frequencies
    plt.figure(figsize=(12, 8))
    plt.bar(top_original.keys(), top_original.values())
    plt.title('Top Original words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    # plot stemmed word frequencies
    plt.figure(figsize=(12, 8))
    plt.bar(top_stemmed.keys(), top_stemmed.values())
    plt.title('Top Stemmed words')
    plt.xlabel('Stems')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()


# --------------------------------------------------
# 6.Main Execution Function
# ---------------------------------------------------
def main():
    """
    Main function to execute the stemming demonstration
    """
    print("=" * 50)
    print("STEMMING DEMONSTRATION")
    print("=" * 50)

    # Display original text
    print("\n1. Original Text:")
    print("-" * 30)
    print(TEXT.strip())

    # Preprocess the text
    print("\n2. Preprocessing text (tokenization & cleaning)...")
    clean_tokens = preprocess_text(TEXT)
    print(f"   Cleaned tokens: {clean_tokens}")

    # Apply stemming
    print("\n3. Applying stemming...")
    stemmed_tokens = apply_stemming(clean_tokens)
    print(f"   Stemmed tokens: {stemmed_tokens}")

    # Compare original vs stemmed
    print("\n4. Comparison (Original → Stemmed):")
    print("-" * 30)
    comparison = list(zip(clean_tokens, stemmed_tokens))
    unique_pairs = list(dict.fromkeys(comparison))  # Remove duplicates for cleaner output
    for orig, stem in unique_pairs[:10]:  # Show first 10 unique pairs
        print(f"   '{orig}' → '{stem}'")

    # Show frequency analysis
    print("\n5. Frequency Analysis:")
    print("-" * 30)
    original_counts = Counter(clean_tokens)
    stemmed_counts = Counter(stemmed_tokens)

    print("   Most common original words:")
    for word, count in original_counts.most_common(5):
        print(f"     - '{word}': {count} time(s)")

    print("\n   Most common stems:")
    for stem, count in stemmed_counts.most_common(5):
        print(f"     - '{stem}': {count} time(s)")

    # Generate visualization
    print("\n6. Generating visualization...")
    plot_frequencies(clean_tokens, stemmed_tokens)

    # Explanation of stemming
    print("\n7. Explanation:")
    print("-" * 30)
    print("   Stemming reduces words to their root form (stem).")
    print("   Example: 'running', 'runs', 'ran' → 'run'")
    print("   This helps group similar words together for analysis.")

    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)


# --------------------------------------------------
# 7.Run the script by executing the Main func
# ---------------------------------------------------
if __name__ == "__main__":
    main()