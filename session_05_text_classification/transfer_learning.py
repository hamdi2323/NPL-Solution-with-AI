"""
=============================================================
Python script to demonstrate transfer learning for text
classification using spaCy and transformers
==============================================================
This example demonstrates TRANSFER LEARNING in NLP using:
  - spaCy
  - Hugging Face Transformers
  - spaCy-Transformers

Instead of training a language model from scratch, we:
  1. Load a PRE-TRAINED transformer model (DistilBERT)
  2. Attach a text classification head
  3. Fine-tune it on a small custom dataset

Task:
  Binary sentiment classification:
    - POSITIVE
    - NEGATIVE

Task pipeline:
      1. Dependency checkinh
      2. Building a transformer-based pipeline
      3. Preparing training examples
      4. Fine-tuing the classifier
      5. Running inference on unseen text
      6. Saving and loading the trained model

Requirments:
    pip install -U transformers torch spacy-transformers

Author: Xamdi Salaad
Date: 14-05-2026
"""
#--------------------------------------------------------
# 0.Import the required modules
# --------------------------------------------------------
from __future__ import annotations
import random, sys
from pathlib import Path
from typing import Any

#--------------------------------------------------------
# 1. Dependency checks
# --------------------------------------------------------
def check_import(module_name: str, install_hint : str) -> Any:

    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"\n[ERROR] Missing dependency:  {module_name}"
              f"\nInstall using: \n{install_hint}")
        sys.exit(1)

#core library
spacy = check_import("spacy", "pip install spacy")
check_import("spacy_transformers", "pip install spacy_transformers torch")

from spacy.training import Example

#--------------------------------------------------------
# 2. Training data
# --------------------------------------------------------
TRAIN_DATA = [
    (
        "I absolutely love this product",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "This movie was fantastic",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "The service was excellent",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "I hate this item",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "This was a terrible experience",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "The food tasted awful",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
]

#--------------------------------------------------------
# 3. Build spaCy pipeline with TRANSFORMER
# --------------------------------------------------------
print("\n[INFO] Creating NLP pipeline")

#create blank English pipeline
nlp = spacy.blank("en")

#Add transformer component (Transfer learning happens here using a pretrained Hugging Face model)
# Note: The correct factory name is "transformer", not "transformers"
nlp.add_pipe(
    "transformer",
    config={
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": "distilbert-base-uncased",
        }
    }
)

#Add text classifier (Modern spaCy versions auto-configure the architecture)
# Note: Use nlp.add_pipe() instead of nlp.create_pipe()
textcat = nlp.add_pipe("textcat", last=True)

#Add labels
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

print(f"\n[INFO] Pipeline components: {nlp.pipe_names}\n")

#--------------------------------------------------------
# 4.Initialise training
# --------------------------------------------------------
print("\n[INFO] Initialising model")
optimiser = nlp.initialize()

#--------------------------------------------------------
# 5.Training loop
# --------------------------------------------------------
print("\n[INFO] Starting fine-tuning")
EPOCHS = 10

for epoch in range(EPOCHS):
    random.shuffle(TRAIN_DATA)

    # Initialize losses as a dictionary, not a tuple
    losses = {}
    examples = []

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Update the model with all examples for this epoch
    nlp.update(
        examples,
        drop=0.2,
        losses=losses,
        sgd=optimiser,
    )

    print(f"Epoch {epoch + 1:02d} | Loss: {losses.get('textcat', 0):.4f}")

#--------------------------------------------------------
#  6.Save model
# --------------------------------------------------------
output_dir = Path("../files/sentiment_transformer_model")

#create a directory
output_dir.mkdir(parents=True, exist_ok=True)

nlp.to_disk(output_dir)

#display save location
print(f"\n[INFO] Saving model to:\n {output_dir.absolute()}")

#--------------------------------------------------------
# 7.Inference /prediction
# --------------------------------------------------------
print("\n[INFO] Inference Demo...\n")

TEST_TEXT = [
    "I really enjoyed this book",
    "The customer support was horrible",
    "Amazing performance by the actors",
    "This app is frustrating and buggy",
]

for text in TEST_TEXT:
    doc = nlp(text)  # Process the text through the full pipeline

    # Access the textcat component's predictions
    positive_score = doc.cats["POSITIVE"]
    negative_score = doc.cats["NEGATIVE"]

    predicted = max(doc.cats, key=doc.cats.get)
    print("\n" + "-" * 55)
    print(f"TEXT: {text}")
    print(f"Predicted: {predicted}")
    print(f"POSITIVE: {positive_score:.4f}")
    print(f"NEGATIVE: {negative_score:.4f}")

#--------------------------------------------------------
# 8. Optional: Reload saved model
# --------------------------------------------------------
print("\n" + "-" * 55)
print("[INFO] Model Reload Demo")
print("=" * 55)
loaded_nlp = spacy.load(output_dir)
reloaded_doc = loaded_nlp("I really enjoyed this book")

print(f"\nReloaded doc cats: {reloaded_doc.cats}")
print(f"Predicted sentiment: {max(reloaded_doc.cats, key=reloaded_doc.cats.get)}")