# AI Conversational Agent

This project implements an AI conversational agent using two large language models (LLMs) that can engage in a dialogue with each other. The conversation is displayed in a graphical user interface (GUI) and is also spoken aloud using text-to-speech technology.

## Features

- Utilizes two LLMs (GPT-2 and DialoGPT) for generating responses
- GUI for displaying the conversation
- Text-to-speech functionality for audible responses
- Sentiment analysis of the conversation
- Conversation logging
- Memory feature to maintain context

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- TextBlob
- pyttsx3

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Musa-Ali-Kazmi/LLM-Chatterbox.git
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers textblob pyttsx3
   ```

## Usage

Run the main script:

```
python chatterbox_app.py
```

1. Select the models for each agent from the dropdown menus.
2. Click "Start Conversation" and enter a topic when prompted.
3. The conversation will begin, and you can watch and listen as the AI agents converse.
4. To end the program, click the "Exit" button.

## Customization

- To use different pre-trained models, modify the model initialization in the script.
- Adjust the `max_new_tokens` parameter in the response generation functions to control response length.
- Modify the GUI layout and design by changing the Tkinter widget properties.


## Acknowledgments

- This project uses the Transformers library by Hugging Face.
- Text-to-speech functionality is provided by pyttsx3.
- Sentiment analysis is performed using TextBlob.

