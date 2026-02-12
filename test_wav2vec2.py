"""
TEST SCRIPT: Wav2Vec2-XLSR-Arabic
==================================
Run this to test the general Arabic model

USAGE:
1. Open PowerShell in VS Code
2. Activate venv: .\venv\Scripts\Activate.ps1
3. Run: python test_wav2vec2.py
"""

import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os
from pathlib import Path

# Colors for terminal output (works in PowerShell)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}Pass {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.FAIL}Fail {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.OKBLUE}- {text}{Colors.END}")


class Wav2Vec2Tester:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    
    def load_model(self):
        """Load the Wav2Vec2 model"""
        print_header("LOADING WAV2VEC2-XLSR-ARABIC MODEL")
        
        print_info("Downloading/loading model from Hugging Face...")
        print_info("First time: ~1.2 GB download (cached for future use)")
        
        start = time.time()
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.model.eval()
            
            elapsed = time.time() - start
            print_success(f"Model loaded in {elapsed:.1f} seconds")
            
            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print_info(f"Model parameters: {total_params:,}")
            print_info(f"Model size: ~1.2 GB")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transcribe an audio file"""
        if not os.path.exists(audio_path):
            print_error(f"File not found: {audio_path}")
            return None
        
        print_info(f"Processing: {audio_path}")
        
        try:
            # Load audio at 16kHz
            speech, rate = librosa.load(audio_path, sr=16000)
            duration = len(speech) / 16000
            
            print_info(f"Audio duration: {duration:.2f} seconds")
            print_info(f"Sample rate: {rate} Hz")
            
            # Process with model
            start = time.time()
            
            inputs = self.processor(
                speech, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            elapsed = time.time() - start
            
            print_success(f"Transcription: {transcription}")
            print_info(f"Processing time: {elapsed:.2f} seconds")
            print_info(f"Real-time factor: {elapsed/duration:.2f}x")
            
            return {
                'transcription': transcription,
                'process_time': elapsed,
                'audio_duration': duration,
                'real_time_factor': elapsed/duration
            }
            
        except Exception as e:
            print_error(f"Error processing audio: {e}")
            return None
    
    def benchmark(self):
        """Run performance benchmark"""
        print_header("PERFORMANCE BENCHMARK")
        
        import numpy as np
        
        print_info("Creating test audio (1 second of silence)...")
        test_audio = np.zeros(16000, dtype=np.float32)
        
        print_info("Running 5 inference tests...\n")
        
        times = []
        for i in range(5):
            start = time.time()
            
            inputs = self.processor(test_audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f"  Test {i+1}: {elapsed:.3f}s")
        
        avg = sum(times) / len(times)
        print_success(f"\nAverage inference time: {avg:.3f} seconds")
        print_info(f"Min: {min(times):.3f}s | Max: {max(times):.3f}s")
        
        print("\ On Surface Pro 6:")
        print("   • First request: ~1-2 seconds")
        print("   • Subsequent: ~0.5-1 second")
        print("   • Acceptable for production")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print_header("INTERACTIVE TESTING")
        
        print("Options:")
        print("  1. Test a single audio file")
        print("  2. Test all files in test_audio/ folder")
        print("  3. Run performance benchmark")
        print("  4. Exit")
        
        while True:
            print()
            choice = input("Choice (1-4): ").strip()
            
            if choice == "1":
                path = input("Enter audio file path: ").strip().strip('"')
                self.transcribe_audio(path)
            
            elif choice == "2":
                test_dir = Path("test_audio")
                
                if not test_dir.exists():
                    print_error("test_audio/ folder not found")
                    print_info("Create 'test_audio' folder and add WAV files")
                    continue
                
                wav_files = list(test_dir.glob("*.wav"))
                
                if not wav_files:
                    print_error("No WAV files found in test_audio/")
                    continue
                
                print_info(f"Found {len(wav_files)} audio files\n")
                
                for audio_file in wav_files:
                    print(f"\n{'-' * 70}")
                    self.transcribe_audio(str(audio_file))
            
            elif choice == "3":
                self.benchmark()
            
            elif choice == "4":
                print_success("Goodbye!")
                break
            
            else:
                print_error("Invalid choice")


def main():
    print_header("WAV2VEC2-XLSR-ARABIC TESTER")
    
    # Initialize tester
    tester = Wav2Vec2Tester()
    
    # Load model
    if not tester.load_model():
        print_error("Failed to initialize. Exiting.")
        return
    
    # Start interactive mode
    tester.interactive_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n")
        print_success("Interrupted by user. Goodbye!")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
