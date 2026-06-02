"""
===============================================================
Python script to demonstrate transformer-based question
answering system
=============================================================

This program demonstrates how to use transformer-based
question answering system (QA) using information about tourist
desination in Kenya

The system performs the following steps:
    1. Load tousrim data from a JSON file
    2. Convert the data into readable format
    3. Use TF-IDF retrieval to identify the most relevant
    4. Uses a Transformer-based Question Answering model to extract an answer
    5. Display the answer in the console

Dataset Location:
files/kenya_tourism.json

Requriments:
pip instal transformer torch scikit-learn
"""

#--------------------------------------------------------------
#0. Import required modules
#--------------------------------------------------------------
import  json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

#--------------------------------------------------------------
#1. Configuration
#------------------------------------------------------------
# Fixed: Go up one level from session_09_question_answering_system to .app, then into files
DATASET_FILE = Path('../files/kenya_tourism.json')

MODEL_NAME = 'distilbert-base-cased-distilled-squad'

#--------------------------------------------------------------
#2. Data loading functions
#------------------------------------------------------------
def load_dataset(file_path):
    # Convert to absolute path for debugging
    absolute_path = file_path.resolve()
    print(f"Looking for file at: {absolute_path}")

    if not absolute_path.exists():
        print(f"ERROR: File not found at {absolute_path}")
        print(f"Current working directory: {Path.cwd()}")
        raise FileNotFoundError(f"Cannot find {absolute_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def build_contexts(dataset):
    contexts = []
    sites_name = []

    sites = dataset['sites']
    for site in sites:
        name = site['name']
        category = site['category']
        region = site['region']
        description = site['description']
        best_time = site['best_time_to_visit']

        # Fixed: Changed 'highlight' to 'highlights' (plural with 's')
        highlights = " , ".join(site['highlights'])
        activities = " , ".join(site['activities'])

        accessibility = site['accessibility']

        context = f"""
        {name} is a {category} destination located in {region}
        
        Description: {description}
        
        Best time to visit: {best_time}
        
        Highlights: {highlights}
        
        Activities: {activities}
        
        Accessibility: {accessibility}
        """

        context = context.strip()
        contexts.append(context)
        sites_name.append(name)

    return contexts, sites_name
#--------------------------------------------------------------
#3. Retrieval Functions
#------------------------------------------------------------
def create_tfidf_matrix(contexts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contexts)  # Fixed variable name
    return vectorizer, tfidf_matrix  # Fixed return order

def retrieve_best_context(question, vectorizer, tfidf_matrix, site_names, contexts):  # Added contexts parameter
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, tfidf_matrix)  # Fixed variable name
    best_index = similarity_scores.argmax()
    best_context = contexts[best_index]  # Fixed: using contexts parameter
    best_site = site_names[best_index]

    similarity_score = similarity_scores[0][best_index]
    return best_context, best_site, similarity_score
#--------------------------------------------------------------
#4.Question Answering Functions
#------------------------------------------------------------
def load_qa_pipeline():
    qa_pipeline = pipeline("question-answering",model = MODEL_NAME)
    return qa_pipeline

def answer_question(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    return result


#--------------------------------------------------------------
#5.Main function execution
#------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("Kenya Tourism Transformer Question Answering System")
    print("=" * 80)

    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_FILE)
    print("Building tourism contexts...")
    contexts, site_names = build_contexts(dataset)

    print("Creating TF-IDF retrieval index...")
    vectorizer, tfidf_matrix = create_tfidf_matrix(contexts)  # Fixed variable assignment
    print("Loading Transformer QA model...")
    print("Please wait on first execution...\n")
    qa_pipeline = load_qa_pipeline()
    print("System ready.\n Type 'exit' or 'quit' to quit")
    while True:
        print("=" * 80)
        question = input("Kindly ask a Kenyan tourism question: \n")
        if question.lower() == 'exit' or question.lower() == 'quit':
            print("\nExiting Transformer QA System...")
            break

        if len(question.strip()) == 0:
             print("Please enter a valid question")
             continue
        #retrieve relevant context
        best_context, best_site, similarity_score = (
            retrieve_best_context(question, vectorizer, tfidf_matrix, site_names, contexts)  # Added contexts
        )

        #generate answer using transformer
        result = answer_question(question, best_context, qa_pipeline)
        answer = result["answer"]
        confidence = result["score"]  # Fixed: changed from "confidence" to "score"

        #display results
        print("\nMost relevant site:")
        print(best_site)

        print("\nAnswer:")
        print(answer)

        print("\nTransformer Confidence Score:")
        print(f"{confidence:.3f}")
        print(f"\nTF-IDF similarity score: {similarity_score:.3f}")

        print("\nRetrieved context:")
        print(best_context)

        print("=" * 80)
#--------------------------------------------------------------
#6.Run the script by invoking main function
#------------------------------------------------------------
if __name__ == "__main__":
    main()