
# Arabic Pronunciation Checker 

AI-powered Arabic pronunciation evaluation using Wav2Vec2 and Flask.

The system:

- Converts speech → text
- Compares with target sentence
- Returns pronunciation accuracy score

---


---

##  Model Used

jonatasgrosman/wav2vec2-large-xlsr-53-arabic

HuggingFace pretrained ASR model for Arabic speech recognition.

---

##  Installation

### 1️ Clone the project

```bash
git clone https://github.com/Ghadeer52/Arabic-Pronunciation-AI.git
cd pronunciation-checker
```

---

###  Create virtual environment (recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

###  Install requirements

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
python app.py
```

You should see:

```
Arabic Pronunciation Checker API
Model loaded: jonatasgrosman/wav2vec2-large-xlsr-53-arabic
Web interface: http://localhost:5000
```

---

##  Web Interface

Open browser:

http://localhost:5000

Upload audio + enter target sentence → get score.

---

##  API Endpoints

### 1️ Health Check

GET /api/check

Response:

```json
{
  "status": "healthy",
  "model_name": "...",
  "timestamp": "..."
}
```

---

### 2️ Transcription Only

POST /api/transcribe

Form-data:

- audio → WAV file

Response:

```json
{
  "transcription": "...",
  "normalized": "...",
  "duration": 2.3
}
```

---

### 3️ Pronunciation Check

POST /api/check-pronunciation

Form-data:

- audio → WAV file
- target_text → Arabic sentence
- threshold → optional (default 0.75)

Response:

```json
{
  "passed": true,
  "score": 91.4,
  "user_said": "...",
  "feedback": "Excellent!"
}
```

---

##  Audio Requirements

- Format: WAV preferred
- Sample rate: 16 kHz (auto-resampled)
- Clear speech recommended
- Avoid background noise

---

##  Performance Notes

- First run downloads model (~1.2GB)
- GPU speeds up inference (optional)
- CPU works but slower

---

##  Troubleshooting

### Port already in use

Change port in code:

```python
app.run(port=5001)
```

---

### Librosa / soundfile error

#### Ubuntu

```bash
sudo apt-get install libsndfile1
```

#### Mac

```bash
brew install libsndfile
```

---

##  Project Structure

```
Arabic-Pronunciation-AI/
│
├── test_audio/
├── app.py
├── requirements.txt
├── README.md
└── venv/
```
