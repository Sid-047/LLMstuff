from transformers import pipeline

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

nlp(
    "#DocImgFile",
    "Question"
)