from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import os
import re
from difflib import SequenceMatcher
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)


print("=" * 70)
print("LOADING PRONUNCIATION ENGINE...")
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()
print("✓ Engine Ready!")
print("=" * 70)



def normalize_arabic(text):
    """Clean Arabic text for fair comparison"""
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ة', 'ه', text)
    return ' '.join(text.replace('ـ', '').split()).strip()

def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)

    duration = librosa.get_duration(y=speech, sr=sr)

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()

    return text, duration




@app.route('/')
def home():
    return render_template_string(HTML_TEST_PAGE)

# 1. Health check for the test script
@app.route('/api/check', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_name": model_name, "timestamp": datetime.now().isoformat()})

# 2. Transcribe for the test script

@app.route('/api/transcribe', methods=['POST'])
def transcribe_only():
    if 'audio' not in request.files: return jsonify({"error": "no audio"}), 400
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        text, duration = transcribe_audio(tmp.name)
    os.unlink(tmp.name)
    return jsonify({"success": True, "transcription": text, "normalized": normalize_arabic(text), "duration": duration})

# 3. Main endpoint (Renamed to 'check' to match your test script)
@app.route('/api/check', methods=['POST'])
@app.route('/api/check-pronunciation', methods=['POST']) 
def check_pronunciation():
    try:
        if 'audio' not in request.files: return jsonify({'success': False, 'error': 'No audio'}), 400
        audio_file = request.files['audio']
        target_text = request.form.get('target_text', '')
        threshold = float(request.form.get('threshold', 0.75))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            user_said, _ = transcribe_audio(tmp.name)
        os.unlink(tmp.name)
        
        norm_user = normalize_arabic(user_said)
        norm_target = normalize_arabic(target_text)
        similarity = SequenceMatcher(None, norm_user, norm_target).ratio()
        
        return jsonify({
            'success': True,
            'passed': similarity >= threshold,
            'score': similarity * 100,
            'user_said': user_said,
            'target': target_text,
            'feedback': "Excellent!" if similarity > 0.9 else "Keep practicing!",
            'exact_match': "Yes" if norm_user == norm_target else "No"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



HTML_TEST_PAGE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Pronunciation Checker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            max-width: 800px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        h1 { color: #333; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        
        .test-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .arabic-text {
            font-size: 32px;
            text-align: center;
            margin: 20px 0;
            font-weight: bold;
            color: #333;
        }
        
        .input-group { margin-bottom: 15px; text-align: right; }
        label { display: block; margin-bottom: 5px; color: #555; font-weight: 600; }
        
        input[type="text"] {
            width: 100%; padding: 12px; border: 2px solid #e0e0e0;
            border-radius: 8px; font-size: 16px; transition: border 0.3s;
        }

        input[type="file"] {
            width: 100%; padding: 10px; border: 2px dashed #667eea;
            border-radius: 8px; cursor: pointer; background: white;
        }
        
        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 15px; border-radius: 50px;
            font-size: 16px; font-weight: 600; cursor: pointer; width: 100%;
            transition: transform 0.2s, box-shadow 0.2s; margin-top: 10px;
        }
        
        .button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102,126,234,0.4); }

        .loading { text-align: center; padding: 20px; display: none; }
        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea;
            border-radius: 50%; width: 40px; height: 40px;
            animation: spin 1s linear infinite; margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .result { margin-top: 20px; padding: 20px; border-radius: 10px; display: none; text-align: right; }
        .result.pass { background: #d4edda; border: 2px solid #28a745; }
        .result.fail { background: #f8d7da; border: 2px solid #dc3545; }
        .result p { margin: 8px 0; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>اختبار النطق العربي</h1>
        <p class="subtitle">Arabic Pronunciation AI</p>
        
        <div class="test-section">
            <div class="arabic-text" id="displayTarget"></div>
            
            <div class="input-group">
                <label>Target Text (النص المطلوب):</label>
                <input type="text" id="targetText" value="" placeholder="اكتب الجملة هنا...">
            </div>
            
            <div class="input-group">
                <label>Upload Audio (تحميل ملف الصوت):</label>
                <input type="file" id="audioFile" accept="audio/*">
            </div>
            
            <button class="button" onclick="check()">Check Pronunciation</button>
        </div>
        
        <div class="loading" id="loader">
            <div class="spinner"></div>
            <p>Processing audio with AI...</p>
        </div>
        
        <div class="result" id="resBox">
            <h3 id="status" style="margin-bottom:10px;"></h3>
            <p><strong>Score:</strong> <span id="score"></span></p>
            <p><strong>You said:</strong> <span id="said"></span></p>
            <p><strong>Target:</strong> <span id="targ"></span></p>
            <p><strong>Feedback:</strong> <span id="feed"></span></p>
            <p><strong>Mode:</strong> whole</p>
            <p><strong>Exact match:</strong> <span id="exact"></span></p>
        </div>
    </div>
    
    <script>
        // Update big display as user types
        document.getElementById('targetText').addEventListener('input', (e) => {
            document.getElementById('displayTarget').textContent = e.target.value;
        });

        async function check() {
            const file = document.getElementById('audioFile').files[0];
            const target = document.getElementById('targetText').value;
            if(!file) return alert("Please upload a file");

            document.getElementById('resBox').style.display = 'none';
            document.getElementById('loader').style.display = 'block';

            const formData = new FormData();
            formData.append('audio', file);
            formData.append('target_text', target);

            try {
                const response = await fetch('/api/check-pronunciation', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                document.getElementById('loader').style.display = 'none';
                const resBox = document.getElementById('resBox');
                resBox.style.display = 'block';
                resBox.className = 'result ' + (data.passed ? 'pass' : 'fail');
                
                document.getElementById('status').textContent = data.passed ? "Correct!" : "Try Again";
                document.getElementById('score').textContent = data.score.toFixed(1) + "%";
                document.getElementById('said').textContent = data.user_said;
                document.getElementById('targ').textContent = data.target;
                document.getElementById('feed').textContent = data.feedback;
                document.getElementById('exact').textContent = data.exact_match;
            } catch (err) {
                alert("Server error");
                document.getElementById('loader').style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Arabic Pronunciation Checker API")
    print("=" * 60)
    print(f"\n✓ Model loaded: {model_name}")
    print(f"\nWeb interface: http://localhost:5000")
    print(f"API endpoint: http://localhost:5000/api/check")
    print(f"\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)