# Voice Print Authentication System

## Overview

This project implements a biometric authentication system that verifies user identity based on unique voice characteristics. Instead of traditional passwords, the system uses voice patterns such as pitch, tone, and frequency to authenticate users.

---

## Features

### Secure Authentication
- Voice-based user verification  
- Eliminates dependency on passwords  
- Reduces risk of credential-based attacks  

### Voice Enrollment
- Records user voice samples  
- Extracts distinguishing audio features  
- Stores voice embeddings for future use  

### Voice Matching
- Compares live voice input with stored voice data  
- Uses similarity metrics for authentication  
- Applies threshold-based decision logic  

### Real-Time Processing
- Fast voice capture and verification  
- Immediate authentication results  

---

## Tech Stack

### Backend
- Python  
- FastAPI or Flask  

### Audio Processing
- Librosa  
- SoundDevice or PyAudio  

### Machine Learning
- NumPy  
- Scikit-learn or PyTorch  

### Database
- SQLite or PostgreSQL  

---

## Project Structure

```
voice-auth/
│
├── app.py                 # Main application entry point
├── enroll.py              # Handles voice enrollment
├── authenticate.py        # Handles authentication logic
├── utils.py               # Audio processing utilities
├── models/                # Saved voice embeddings or ML models
├── database/              # Storage for user data
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voice-auth.git
cd voice-auth
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
python app.py
```

---

## How It Works

1. User records a voice sample during enrollment  
2. System extracts features (e.g., MFCCs)  
3. Features are stored as a voice print  
4. During login, a new voice sample is captured  
5. System compares new sample with stored voice print  
6. Access is granted if similarity exceeds threshold  

---

## Use Cases

- Secure login systems  
- Banking and financial authentication  
- Access control systems  
- Smart home security  

---

## Future Enhancements

- Anti-spoofing (replay attack detection)  
- Deep learning-based speaker recognition  
- Mobile and web integration  
- Multi-factor authentication  

---

## Author

Swathilekha V

---

## License

This project is intended for academic and research purposes.
