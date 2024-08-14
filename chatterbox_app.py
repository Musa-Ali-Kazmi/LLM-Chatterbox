import tkinter as tk
from tkinter import scrolledtext, simpledialog, ttk
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob
import datetime
import os
import pyttsx3
import threading
import time

# Initialize models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
dialogpt_model.to(device)

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Ensure we have at least two voices
if len(voices) < 2:
    raise ValueError("Not enough voices available. Please install additional voice packs.")

# Function to generate response from GPT-2
def generate_gpt2_response(prompt, max_new_tokens=50):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    output = gpt2_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=gpt2_tokenizer.eos_token_id)
    response = gpt2_tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Function to generate response from DialoGPT
def generate_dialogpt_response(prompt, max_new_tokens=50):
    input_ids = dialogpt_tokenizer.encode(prompt + dialogpt_tokenizer.eos_token, return_tensors="pt").to(device)
    output = dialogpt_model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2)
    response = dialogpt_tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "positive"
    elif sentiment < -0.1:
        return "negative"
    else:
        return "neutral"

# Function to speak text
def speak_text(text, voice_index):
    engine.setProperty('voice', voices[voice_index].id)
    engine.say(text)
    engine.runAndWait()

# Main application class
class ChatbotApp:
    def __init__(self, master):
        self.master = master
        master.title("LLM Chatbot")
        master.geometry("900x700")
        master.configure(bg="#2C3E50")

        # Create and pack widgets
        self.chat_frame = tk.Frame(master, bg="#34495E", padx=10, pady=10)
        self.chat_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=80, height=25, font=("Arial", 12), bg="#ECF0F1", fg="#2C3E50")
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.sentiment_frame = tk.Frame(master, bg="#2C3E50", padx=10, pady=5)
        self.sentiment_frame.pack(fill=tk.X)

        self.sentiment_label = tk.Label(self.sentiment_frame, text="Sentiment: Neutral", font=("Arial", 14, "bold"), bg="#F1C40F", fg="#2C3E50", padx=10, pady=5)
        self.sentiment_label.pack()

        self.control_frame = tk.Frame(master, bg="#2C3E50", padx=10, pady=10)
        self.control_frame.pack(fill=tk.X)

        self.model1_var = tk.StringVar(value="GPT-2")
        self.model2_var = tk.StringVar(value="DialoGPT")

        tk.Label(self.control_frame, text="Model 1:", font=("Arial", 12), bg="#2C3E50", fg="white").grid(row=0, column=0, padx=5)
        self.model1_menu = ttk.Combobox(self.control_frame, textvariable=self.model1_var, values=["GPT-2", "DialoGPT"], state="readonly", font=("Arial", 12))
        self.model1_menu.grid(row=0, column=1, padx=5)

        tk.Label(self.control_frame, text="Model 2:", font=("Arial", 12), bg="#2C3E50", fg="white").grid(row=0, column=2, padx=5)
        self.model2_menu = ttk.Combobox(self.control_frame, textvariable=self.model2_var, values=["GPT-2", "DialoGPT"], state="readonly", font=("Arial", 12))
        self.model2_menu.grid(row=0, column=3, padx=5)

        self.start_button = tk.Button(self.control_frame, text="Start Conversation", command=self.start_conversation, font=("Arial", 12, "bold"), bg="#27AE60", fg="white", padx=10, pady=5)
        self.start_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.exit_button = tk.Button(self.control_frame, text="Exit", command=self.exit_app, font=("Arial", 12, "bold"), bg="#E74C3C", fg="white", padx=10, pady=5)
        self.exit_button.grid(row=1, column=2, columnspan=2, pady=10)

        self.conversation_active = False
        self.log_file = None
        self.conversation_history = []
        self.speech_thread = None

    def start_conversation(self):
        if not self.conversation_active:
            # Prompt user for a starting topic
            topic = simpledialog.askstring("Input", "Enter a word or phrase to start the conversation:")
            if topic:
                self.conversation_active = True
                self.chat_area.delete(1.0, tk.END)
                self.chat_area.insert(tk.END, f"Starting conversation about: {topic}\n\n")
                
                # Create a new log file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"conversation_log_{timestamp}.txt"
                self.log_file = open(log_filename, "w", encoding="utf-8")
                self.log_file.write(f"Conversation Log\nStarting topic: {topic}\n\n")
                
                # Start the conversation with the given topic
                initial_prompt = f"Let's have a conversation about {topic}. Please respond with a question or statement to start the discussion."
                self.conversation_history = [initial_prompt]
                self.continue_conversation(initial_prompt)

    def continue_conversation(self, prompt):
        if self.conversation_active:
            # Model 1 generates a response
            model1_response = self.generate_response(self.model1_var.get(), prompt)
            self.chat_area.insert(tk.END, f"{self.model1_var.get()}: {model1_response}\n\n")
            self.chat_area.see(tk.END)
            self.log_file.write(f"{self.model1_var.get()}: {model1_response}\n\n")
            self.conversation_history.append(model1_response)

            # Speak Model 1's response
            self.speak_and_wait(model1_response, 0)

            # Update sentiment
            sentiment = analyze_sentiment(model1_response)
            self.update_sentiment_display(sentiment)

            # Model 2 responds to Model 1
            model2_response = self.generate_response(self.model2_var.get(), model1_response)
            self.chat_area.insert(tk.END, f"{self.model2_var.get()}: {model2_response}\n\n")
            self.chat_area.see(tk.END)
            self.log_file.write(f"{self.model2_var.get()}: {model2_response}\n\n")
            self.conversation_history.append(model2_response)

            # Speak Model 2's response
            self.speak_and_wait(model2_response, 1)

            # Update sentiment
            sentiment = analyze_sentiment(model2_response)
            self.update_sentiment_display(sentiment)

            # Continue the conversation
            self.master.after(100, self.continue_conversation, model2_response)

    def speak_and_wait(self, text, voice_index):
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join()
        self.speech_thread = threading.Thread(target=speak_text, args=(text, voice_index))
        self.speech_thread.start()
        while self.speech_thread.is_alive():
            self.master.update()
            time.sleep(0.1)

    def generate_response(self, model_name, prompt):
        # Combine conversation history with prompt
        context = " ".join(self.conversation_history[-5:])  # Use last 5 exchanges as context
        full_prompt = f"{context} {prompt}"

        if model_name == "GPT-2":
            return generate_gpt2_response(full_prompt)
        elif model_name == "DialoGPT":
            return generate_dialogpt_response(full_prompt)

    def update_sentiment_display(self, sentiment):
        if sentiment == "positive":
            self.sentiment_label.config(text="Sentiment: Positive", bg="#27AE60")
        elif sentiment == "negative":
            self.sentiment_label.config(text="Sentiment: Negative", bg="#E74C3C")
        else:
            self.sentiment_label.config(text="Sentiment: Neutral", bg="#F1C40F")

    def exit_app(self):
        if self.log_file:
            self.log_file.close()
        self.master.quit()

# Create and run the application
root = tk.Tk()
app = ChatbotApp(root)
root.mainloop()




