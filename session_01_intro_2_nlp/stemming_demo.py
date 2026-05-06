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
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
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
    Converts text to lowercase, removes punctuations,
    and returns list of cleaned tokens.
    :param text: input text
    :return: cleaned tokens
    """
    tokens = word_tokenize(text.lower())
    clean_tokens = [
        re.sub('[^a-z]', ' ', token)
        for token in tokens
    ]
    # Filter out empty strings and whitespace-only tokens
    return [token.strip() for token in clean_tokens if token.strip()]


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

    # Select top items for clarity
    top_original = dict(original_counts.most_common(5))
    top_stemmed = dict(stemmed_counts.most_common(5))

    plt.figure(figsize=(14, 6))

    # Plot original word frequencies
    plt.subplot(1, 2, 1)
    plt.bar(list(top_original.keys()), list(top_original.values()))
    plt.title('Top 5 Original Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    # Plot stemmed word frequencies
    plt.subplot(1, 2, 2)
    plt.bar(list(top_stemmed.keys()), list(top_stemmed.values()))
    plt.title('Top 5 Stemmed Words')
    plt.xlabel('Stemmed Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('stemming_demo.png')
    plt.show()


# --------------------------------------------------
# 6.Main Execution Function
# ---------------------------------------------------
def main():
    """Main function to demonstrate stemming process with visualization."""
    print("Original text:")
    print(TEXT)
    print("\n" + "=" * 60)

    # Preprocess the text
    original_tokens = preprocess_text(TEXT)
    print(f"Original tokens ({len(original_tokens)}): {original_tokens}")
    print(f"Original top frequencies: {Counter(original_tokens).most_common(5)}")

    # Apply stemming
    stemmed_tokens = apply_stemming(original_tokens)
    print(f"\nStemmed tokens ({len(stemmed_tokens)}): {stemmed_tokens}")
    print(f"Stemmed top frequencies: {Counter(stemmed_tokens).most_common(5)}")

    print("\n" + "=" * 60)
    print("Generating visualization...")

    # Create visualization
    plot_frequencies(original_tokens, stemmed_tokens)
    print("Visualization saved as 'stemming_demo.png'")


# --------------------------------------------------
# 7.Run the script by executing the Main func
# ---------------------------------------------------
if __name__ == "__main__":
    main()