import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")

imgPath = 'imgSample.png'
raw_image = Image.open(imgPath).convert('RGB')

question = "how many peeps are in this Picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))