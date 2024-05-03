import numpy as np
import sounddevice as sd
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pyttsx3

class PersonalAssistant:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize the recognizer
        self.recognizer = sr.Recognizer()

    def listen(self):
        # Record audio from the microphone
        print("Listening...")
        duration = 5  # seconds
        fs = 44100  # Sample rate
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait until recording is finished
        audio_data = np.array(myrecording, dtype=np.float32)

        # Use the audio data with the recognizer
        audio = sr.AudioData(audio_data.tobytes(), fs, 2)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

    def respond(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=150)
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(response_text)
        self.engine.say(response_text)
        self.engine.runAndWait()
        return response_text

# Create an instance of the assistant
assistant = PersonalAssistant()

# Example of continuous listening and responding
while True:
    text = assistant.listen()
    if text:
        assistant.respond(text)
