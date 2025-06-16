import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import numpy as np
import os
import nltk
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from xlm_emo.classifier import EmotionClassifier

# --- Configuration ---

# File Paths
INITIAL_DATASET_PATH = 'GPTarget2024.csv'
FINAL_DATASET_PATH = 'Final_dataset_cleaned_2.csv'  # Output file for the OLS model

# Model IDs
TOPIC_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# Pipeline Settings
TEXT_COLUMN = 'message_text'
EXCLUDED_TOPICS_FOR_EXTRACTION = ["China sanctions", "NATO support", "Renewable energy", "digital privacy"]

# BERTopic Settings
MIN_TOPIC_SIZE_BERT = 15


# --- End Configuration ---


def initialize_topic_pipeline():
    """Initializes and returns the text generation pipeline for topic extraction."""
    print("Initializing LLAMA-2-7B model for topic extraction...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TOPIC_MODEL_ID,
        quantization_config=bnb_config,
        device_map='auto'
    )
    topic_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )
    print("Topic extraction pipeline ready.")
    return topic_pipeline


def initialize_emotion_pipeline(model_id):
    """Initializes and returns a pipeline for emotion classification."""
    print(f"Initializing model for emotion classification: {model_id}...")
    emotion_pipeline = pipeline(
        "text-classification",
        model=model_id,
        return_all_scores=False
    )
    print("Emotion classification pipeline ready.")
    return emotion_pipeline


def extract_persuasive_topics(text, topic_pipeline):
    """
    Uses the topic extraction pipeline to identify persuasive topics in a given text.
    """
    if not isinstance(text, str) or not text.strip():
        return [None, None, None]

    excluded_topics_str = ', '.join(EXCLUDED_TOPICS_FOR_EXTRACTION)
    prompt = (
        f"Read the text provided below. Your task is to identify the persuasive topics that the text uses to influence the reader's perspective.\n"
        f"These topics should reflect the ideas or themes the text emphasizes to persuade the reader, rather than the main content of the text.\n"
        f"An example is a text that tries to persuade the reader to up NATO spending and does this by talking about family safety. In this case I want family safety as a topic\n"
        f"Do not include the following topics {excluded_topics_str}.\n"
        f"Each topic should contain fewer than 3 words. Ensure you only return the topic and nothing else.\n"
        f"The output should be in the following format: Topic 1: XXX Topic 2: XXX Topic 3: XXX\n"
        f"Text: {text}"
    )

    try:
        response = topic_pipeline(prompt)
        generated_text = response[0]['generated_text']
        topics = [None, None, None]
        if "Topic 1:" in generated_text:
            last_topic_section = generated_text.rfind("Topic 1:")
            topic_section = generated_text[last_topic_section:]
            matches = re.findall(r"Topic \d: (.*?)(?=\s*Topic \d:|$)", topic_section)
            for i, match in enumerate(matches[:3]):
                topic = match.strip()
                if topic and topic.lower() != 'xxx':
                    topics[i] = topic
        return topics
    except Exception as e:
        return [None, None, None]


def classify_emotion(text, emotion_pipeline):
    """
    Uses the emotion classification pipeline to identify the dominant emotion in a text.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        cleaned_text = BeautifulSoup(text, "html.parser").get_text()
        if not cleaned_text.strip():
            return "unknown"
        results = emotion_pipeline(cleaned_text)
        return results[0]['label']
    except Exception as e:
        return "unknown"


def run_bertopic(df, text_column):
    """Runs BERTopic on the dataframe and returns keywords."""
    print("\n--- Running BERTopic for Topic Accuracy Metric ---")
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')

    docs = df[text_column].fillna('').tolist()
    cleaned_docs = [BeautifulSoup(doc, 'html.parser').get_text() for doc in
                    tqdm(docs, desc="Cleaning HTML for BERTopic")]

    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE_BERT,
        verbose=False
    )

    print("Fitting BERTopic model...")
    topics, _ = topic_model.fit_transform(cleaned_docs)

    # Create a mapping from topic ID to keywords
    topic_keyword_map = {topic_id: ", ".join([word for word, _ in topic_model.get_topic(topic_id)[:5]]) for topic_id in
                         topic_model.get_topics()}

    # Assign keywords to each document
    bertopic_keywords = [topic_keyword_map.get(topic_id, "[OUTLIER]") for topic_id in topics]
    print("BERTopic analysis complete.")
    return bertopic_keywords


def get_words_from_text(text):
    """Helper function to extract clean words from a string."""
    if not isinstance(text, str): return set()
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def calculate_accuracy_metrics(df):
    """Calculates and prints the final accuracy metrics."""
    print("\n--- Calculating Final Accuracy Metrics ---")

    # 1. Emotion Classifier Agreement
    print("\n1. Emotion Classifier Agreement (DistilRoBERTa vs. XLM-EMO)")
    common_emotions = ['anger', 'sadness', 'fear', 'joy']
    # Filter for rows where both classifiers predicted one of the common emotions
    df_filtered_emotion = df[
        df['predicted_emotion_2'].isin(common_emotions) &
        df['predicted_emotion_xlm'].isin(common_emotions)
        ].copy()

    if not df_filtered_emotion.empty:
        agreement = (df_filtered_emotion['predicted_emotion_2'] == df_filtered_emotion['predicted_emotion_xlm']).mean()
        print(f"   Agreement on common emotions (joy, anger, fear, sadness): {agreement:.2%}")
        print(f"   (Based on {len(df_filtered_emotion)} messages where both models identified a common emotion)")
    else:
        print("   No messages found for common emotion agreement calculation.")

    # 2. Topic Overlap (LLM vs. BERTopic)
    print("\n2. Topic Model Overlap (LLM vs. BERTopic)")
    match_count = 0
    total_rows = 0
    for _, row in df.iterrows():
        total_rows += 1
        llm_words = set()
        for col in ['topic_1', 'topic_2', 'topic_3']:
            llm_words.update(get_words_from_text(row[col]))

        bertopic_words = get_words_from_text(row['bertopic_keywords'])

        if llm_words and bertopic_words and not llm_words.isdisjoint(bertopic_words):
            match_count += 1

    if total_rows > 0:
        overlap_percentage = (match_count / total_rows) * 100
        print(f"   Overlap Percentage: {overlap_percentage:.2f}%")
        print(f"   ({match_count} out of {total_rows} messages showed an overlap)")
    else:
        print("   No rows available to calculate topic overlap.")


def main():
    """Main function to run the complete data preparation pipeline."""
    try:
        df = pd.read_csv(INITIAL_DATASET_PATH)
        print(f"Successfully loaded initial dataset from '{INITIAL_DATASET_PATH}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Initial dataset not found at '{INITIAL_DATASET_PATH}'. Please check the path.")
        return

    # --- Step 1: Topic Extraction (LLM) ---
    topic_pipeline = initialize_topic_pipeline()
    print("\nStarting persuasive topic extraction...")
    tqdm.pandas(desc="Extracting Topics")
    topic_results = df[TEXT_COLUMN].progress_apply(lambda text: extract_persuasive_topics(text, topic_pipeline))
    df[['topic_1', 'topic_2', 'topic_3']] = pd.DataFrame(topic_results.tolist(), index=df.index)
    print("Topic extraction complete.")

    # --- Step 2: Emotion Classification (DistilRoBERTa for OLS) ---
    emotion_pipeline_roberta = initialize_emotion_pipeline(EMOTION_MODEL_ID)
    print("\nStarting emotion classification for OLS model...")
    tqdm.pandas(desc="Classifying Emotions (DistilRoBERTa)")
    df['predicted_emotion_2'] = df[TEXT_COLUMN].progress_apply(
        lambda text: classify_emotion(text, emotion_pipeline_roberta))
    print("Emotion classification for OLS model complete.")

    # --- Step 3: Save the Final Dataset for OLS ---
    try:
        final_df_for_ols = df.copy()
        final_df_for_ols.to_csv(FINAL_DATASET_PATH, index=False)
        print(f"\nPipeline intermediate save complete. Main dataset saved to '{FINAL_DATASET_PATH}'.")
        print("This file is now ready for use with your OLS_model.py script.")
    except Exception as e:
        print(f"Error saving the final dataset: {e}")
        return

    # --- Step 4: Run Additional Models for Accuracy Calculation ---
    df['bertopic_keywords'] = run_bertopic(df, TEXT_COLUMN)
    print("\n--- Running XLM-EMO for Emotion Agreement Metric ---")
    xlm_classifier = EmotionClassifier()
    texts_for_xlm = df[TEXT_COLUMN].dropna()
    print("Predicting emotions with XLM-EMO...")
    xlm_predictions = xlm_classifier.predict(texts_for_xlm.tolist())
    df['predicted_emotion_xlm'] = pd.Series(xlm_predictions, index=texts_for_xlm.index)
    print("XLM-EMO analysis complete.")

    # --- Step 5: Calculate and Display Accuracy Metrics ---
    calculate_accuracy_metrics(df)


if __name__ == '__main__':
    tqdm.pandas()
    main()