import time
import torch
from PIL import Image
from tkinter import filedialog
from transformers import BlipProcessor, BlipForQuestionAnswering

t1 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Select the Directory Yo!")
inImg = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff")])
rawImg = Image.open(inImg).convert('RGB')

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

question = "how many peeps are in this Picture?"
inputs = processor(rawImg, question, return_tensors="pt").to(device)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

t2 = time.time()
print("\nCompleteExecTime: ", (t2-t1))