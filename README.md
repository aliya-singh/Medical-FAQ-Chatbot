🏥 Medical FAQ Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a curated FAQ dataset. This application uses semantic search and text generation to provide informative responses while emphasizing the importance of professional medical consultation.
✨ Features

Semantic Search: Uses sentence transformers to find relevant medical information
Multiple Generation Methods: Choose between simple retrieval or local LLM generation
Confidence Scoring: Shows how well the retrieved information matches your query
Free Operation: No API keys required for basic functionality
Medical Disclaimers: Emphasizes the educational nature of responses
Interactive UI: Clean Streamlit interface with quick question buttons

🚀 Quick Start
Prerequisites

Python 3.8 or higher
pip package manager

Installation

Clone or download the project files

bash   git clone <your-repo-url>
   cd medical-faq-chatbot

Install required packages

bash   pip install streamlit pandas numpy sentence-transformers requests

Set up the data directory structure

   medical-faq-chatbot/
   ├── data/
   │   └── medical_faqs.csv
   ├── app_free.py
   ├── preprocess_data.py
   └── README.md
Data Setup

Place your medical FAQ data

Put your medical_faqs.csv file in the data/ directory
The CSV should have columns: qtype, Question, Answer


Preprocess the data and generate embeddings

bash   python preprocess_data.py
This will:

Clean and normalize the text data
Remove duplicates and empty entries
Generate embeddings for semantic search
Create data/medical_faq_clean.csv and answer_embeddings.npy

Running the Application
bashstreamlit run app_free.py
The app will open in your browser at http://localhost:8501
📊 Data Format
Your medical_faqs.csv should follow this format:
qtypeQuestionAnswersymptomsWhat are the symptoms of diabetes?The signs and symptoms of diabetes are...treatmentHow is high blood pressure treated?High blood pressure can be managed by...
🎯 Usage

Ask Questions: Type your medical question in the text input
View Answers: Get AI-generated responses based on the FAQ database
Check Sources: Expand the "View Sources" section to see retrieved information
Quick Questions: Use the preset buttons for common medical queries
Confidence Metrics: Check the confidence score to gauge answer reliability
