🏥 Medical FAQ Chatbot
A RAG-powered chatbot that answers medical questions using semantic search. For educational purposes only - not medical advice.
🚀 Quick Start
Clone the Repository
git clone https://github.com/aliya-singh/Medical-FAQ-Chatbot.git
cd Medical-FAQ-Chatbot
Install Dependencies
pip install -r requirements.txt
Setup Data & Embeddings
python preprocess_data.py
Run the App
streamlit run app.py
Open http://localhost:8501 in your browser.

📁 Repository Structure
Medical-FAQ-Chatbot/
├── data/medical_faqs.csv        # FAQ dataset
├── app.py                       # Main Streamlit app
├── preprocess_data.py          # Data preprocessing
├── requirements.txt            # Python dependencies
└── README.md

🎯 Features
Free: No API keys required
Semantic Search: Finds relevant medical info
Confidence Scores: Shows answer reliability
Quick Questions: Preset common queries
Source Citations: View original FAQ sources
