from transformers import pipeline
from colorama import Fore, Style
import torch

inText = input('inputText Yo: ')
print(Fore.CYAN + Style.BRIGHT + "\nJus' Got the Text Content In !" + Fore.RESET)
minWordCount = int(input('MinWordCount Yo: '))
maxLen = min(len(inText.split(' ')) + 50, 512)

device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=device)
summary = summarizer(inText, maxLen=maxLen, min_length=minWordCount)
genSummary = summary[0]['summary_text']
print(Fore.WHITE + Style.BRIGHT + "\tThe Content is Out Yo!" + Fore.RESET)

print("GeneratedStuff: ", genSummary)