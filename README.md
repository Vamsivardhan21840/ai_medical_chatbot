# 🩺 AI Medical Chatbot: Modern Multimodal Diagnostic Engine

A professional-grade medical AI application built with **FastAPI** and **Groq**, featuring a high-end futuristic UI and a powerful multimodal diagnostic analyzer.

---

## 🚀 Overview
This project transforms basic medical AI interactions into a premium user experience. It allows users to upload medical images (like dermatology photos, scans, etc.) and receive instant, high-fidelity AI-generated diagnostic analysis.

### ✨ Key Features
- **Futuristic UI/UX**: Built with a sleek dark-mode aesthetic, glassmorphism, and smooth Tailwind CSS animations.
- **Multimodal AI Analysis**: Powered by **Llama 3.2 90B Vision**, providing deep clinical context from image inputs.
- **Ultra-Low Latency**: Optimized for Groq's LPU (Language Processing Unit) architecture for near-instant responses.
- **Future-Proofing**: Built-in fallback to **Llama 4 Scout** to ensure 100% uptime.
- **Mobile Responsive**: Fully adaptive design for various devices and screen sizes.

---

## 🛠️ Technology Stack

| Layer | Technology Used |
| :--- | :--- |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), Python 3.10+ |
| **AI Engine** | [Groq API](https://groq.com/) (Llama 3.2 90B Vision) |
| **Frontend** | HTML5, Vanilla JavaScript, [Tailwind CSS](https://tailwindcss.com/) |
| **Styling** | Google Fonts (Montserrat, Inter), FontAwesome, Custom CSS Glassmorphism |
| **Utilities** | Jinja2 Templates, Pillow (PIL), Requests, Python-Dotenv |

---

## ⚙️ Project Setup

### 1. Prerequisites
- Python 3.9 or higher
- A [Groq API Key](https://console.groq.com/keys)

### 2. Installation
Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/Vamsivardhan21840/ai_medical_chatbot.git
cd ai_medical_chatbot

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
# Copy the example env
cp .env.example .env
```
Open `.env` and add your Groq API Key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the Project

Launch the development server:
```bash
python ai_medical_chatbot/app.py
```
Or use Uvicorn directly:
```bash
uvicorn ai_medical_chatbot.app:app --reload
```

Once running, access the application at: **`http://127.0.0.1:8000`**

---

## 📂 Project Structure
```text
├── ai_medical_chatbot/
│   ├── app.py             # Main FastAPI Backend
│   ├── .env               # Secrets (Not pushed to git)
│   └── requirements.txt   # Dependencies
├── static/
│   └── hero.png           # Custom AI Generated Assets
├── templates/
│   └── index.html         # Modern Frontend Implementation
└── README.md              # Project Documentation
```

## 📜 License
Distributable for educational and clinical research purposes.

---
**Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
