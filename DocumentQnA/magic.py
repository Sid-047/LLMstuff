from transformers import pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
).to(device)

nlp(
    "#DocImgFile",
    "Question"
)