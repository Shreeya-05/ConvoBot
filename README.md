# ConvoBot 🤖
# AI-Powered Customer Support Chatbot with Real-Time Analytics

ConvoBot is an AI chatbot built using Python, FastAPI, and the Groq API. It features a live analytics panel that tracks sentiment, intent, and escalation risk in real time.

---

## 🛠 Tech Stack

|       Technology         |            Purpose            |
|--------------------------|-------------------------------|
| Python                   | Core language                 |
| FastAPI                  | Backend web framework         |
| Groq API (Llama 3.3 70B) | AI response generation        |
| Uvicorn                  | ASGI server                   |
| HTML + CSS + JS          | Frontend (embedded in Python) |
| Render                   | Cloud deployment              |

---

## ✨ Features

- 💬 AI-powered customer support chat
- 📊 Live analytics — intent, sentiment, topic, escalation risk
- ⚡ Fast responses using Groq's Llama 3.3 70B model
- 🚀 Deployed on Render via GitHub

---

## 📁 Project Structure

```
convobot/
├── run.py              # Entire app — backend + frontend
├── requirements.txt    # Python dependencies
├── .env                # API key (not pushed to GitHub)
└── .gitignore          # Ignores .env and other files
```

---

## ⚙️ Installation

**1. Clone the repository**
git clone https://github.com/your-username/ConvoBot.git
cd convobot

**2. Install dependencies**
pip install -r requirements.txt

**3. Add your Groq API key**
Create a `.env` file in the root folder:
GROQ_API_KEY=gsk_your_actual_key_here

**4. Run the app**
python run.py
or
python3 run.py

The app will start and open automatically in your browser at:
http://localhost:8001

---

## 🌐 Deployment

This project is deployed on **Render** and connected to GitHub. Any changes pushed to the main branch are automatically redeployed.

Live URL: `https://convobot-n3cf.onrender.com`

---

## 📌 Notes

- The Groq API key is stored in a `.env` file and never hardcoded.
- The frontend is embedded directly in `run.py` — no separate HTML file needed.
- Make sure `.env` is listed in `.gitignore` before pushing to GitHub.

---
## 📄 License

This project is for educational purposes.
