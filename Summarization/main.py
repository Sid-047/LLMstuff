from transformers import pipeline
import pprint

s = str(input("Enter Content: "))
x = int(input("\nMinimal Word Count: "))
device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=device)
ans = summarizer(s, min_length=x)

pprint.pprint(ans)