import time
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

x = input("Enter Text Yo: ")
t1 = time.time()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer(x, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
t2 = time.time()
print(model.config.id2label[predicted_class_id])

print("ExecTime: ", (t2-t1))