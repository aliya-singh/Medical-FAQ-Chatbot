# preprocess_data.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    return text.strip()

def preprocess_medical_data():
    """Load and preprocess the medical FAQ data"""
    
    # Load the original data
    print("Loading data from data/medical_faqs.csv...")
    df = pd.read_csv("data/medical_faqs.csv")
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean the data
    print("\nCleaning data...")
    
    # Clean text columns
    df['Question'] = df['Question'].apply(clean_text)
    df['Answer'] = df['Answer'].apply(clean_text)
    
    # Remove rows with empty questions or answers
    df = df[(df['Question'].str.len() > 10) & (df['Answer'].str.len() > 20)]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Question', 'Answer'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data shape: {df.shape}")
    
    # Save cleaned data
    df.to_csv("data/medical_faq_clean.csv", index=False)
    print("Saved cleaned data to data/medical_faq_clean.csv")
    
    return df

def generate_embeddings(df):
    """Generate embeddings for the answers"""
    
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings for answers...")
    
    # Generate embeddings for answers
    answers = df['Answer'].tolist()
    embeddings = model.encode(answers, 
                             batch_size=32, 
                             show_progress_bar=True,
                             convert_to_numpy=True)
    
    # Save embeddings
    np.save("answer_embeddings.npy", embeddings)
    print(f"Saved {len(embeddings)} embeddings to answer_embeddings.npy")
    
    # Also generate embeddings for questions (optional, for future use)
    print("Generating embeddings for questions...")
    questions = df['Question'].tolist()
    question_embeddings = model.encode(questions, 
                                     batch_size=32, 
                                     show_progress_bar=True,
                                     convert_to_numpy=True)
    
    np.save("question_embeddings.npy", question_embeddings)
    print(f"Saved {len(question_embeddings)} question embeddings")
    
    return embeddings

def main():
    """Main preprocessing pipeline"""
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Preprocess data
    df = preprocess_medical_data()
    
    # Generate embeddings
    embeddings = generate_embeddings(df)
    
    # Show some statistics
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Total questions: {len(df)}")
    print(f"Average question length: {df['Question'].str.len().mean():.1f} characters")
    print(f"Average answer length: {df['Answer'].str.len().mean():.1f} characters")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Show sample data
    print("\nSample entries:")
    for i in range(min(3, len(df))):
        print(f"\nQ{i+1}: {df.iloc[i]['Question'][:100]}...")
        print(f"A{i+1}: {df.iloc[i]['Answer'][:100]}...")

if __name__ == "__main__":
    main()