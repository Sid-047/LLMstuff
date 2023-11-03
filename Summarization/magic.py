from transformers import pipeline
from colorama import Fore, Style
import torch

input_text = input('inputText Yo: ')
print(Fore.CYAN + Style.BRIGHT + "\nJus' Got the Text Content In !" + Fore.RESET)
min_word_count = int(input('MinWordCount Yo: '))
max_length = min(len(input_text.split(' ')) + 50, 512)

device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=device)
summary = summarizer(input_text, max_length=max_length, min_length=min_word_count)
generated_summary = summary[0]['summary_text']
print(Fore.WHITE + Style.BRIGHT + "\tThe Content is Out Yo!" + Fore.RESET)

print("GeneratedStuff: ", generated_summary)