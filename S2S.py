import os
import sys
import pyaudio
import json
from vosk import Model, KaldiRecognizer
from transformers import pipeline
import pyttsx3

# Function to load the model based on language
def load_model(language):
    if language == "en":
        model_name = "vosk-model-small-en-us-0.15"
    elif language == "fr":
        model_name = "vosk-model-small-fr-0.22"
    else:
        print("Language not supported.")
        sys.exit(1)

    if not os.path.exists(model_name):
        print(f"Please download the {language.upper()} model from https://alphacephei.com/vosk/models and unpack it in the current directory.")
        sys.exit(1)

    return Model(model_name)

# Initialize translation pipelines
def init_translation_pipelines():
    translations = {}
    translations['en_to_fr'] = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    translations['fr_to_en'] = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return translations

# Function for live text-to-speech conversion
def text_to_speech(text):
    if not text:
        print("No text to convert to speech.")
        return  # Exit if there's no text to speak

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speech rate
    engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)

    # Speak the text directly
    engine.say(text)
    engine.runAndWait()  # Blocks while processing all currently queued commands

# Language selection
print("Select language:")
print("1. English")
print("2. French")
language_choice = input("Enter your choice (1/2): ")

if language_choice == "1":
    language_code = "en"
elif language_choice == "2":
    language_code = "fr"
else:
    print("Invalid choice. Defaulting to English.")
    language_code = "en"

# Initialize the model and recognizer
model = load_model(language_code)
recognizer = KaldiRecognizer(model, 16000)

# Initialize translation pipelines
translation_pipes = init_translation_pipelines()

# Set up audio input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Listening... Press Ctrl+C to stop.")

# Real-time audio processing
try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)  # Prevent buffer overflow
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_json = json.loads(result)
            if 'text' in result_json:
                original_text = result_json['text']
                print("Transcribed Text:", original_text)

                # Translation logic
                if language_code == "en":
                    translated_text = translation_pipes['en_to_fr'](original_text)[0]['translation_text']
                    print(f"Translated to French: {translated_text}")
                elif language_code == "fr":
                    translated_text = translation_pipes['fr_to_en'](original_text)[0]['translation_text']
                    print(f"Translated to English: {translated_text}")

                # Ask user if they want to hear the translated text
                t2s_choice = input("Do you want to hear the translated text? (yes/no): ").strip().lower()
                if t2s_choice in ['yes', 'y']:
                    text_to_speech(translated_text)

                # Ask user if they want to hear the original text
                hear_original_choice = input("Do you want to hear the original text? (yes/no): ").strip().lower()
                if hear_original_choice in ['yes', 'y']:
                    text_to_speech(original_text)

        else:
            partial_result = recognizer.PartialResult()
            partial_result_json = json.loads(partial_result)
            if 'partial' in partial_result_json:
                print("Partial Transcription:", partial_result_json['partial'])

except KeyboardInterrupt:
    print("\nStopped listening.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
