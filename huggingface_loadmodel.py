from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
import numpy as np
model2 = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\Tom\\Google Drive\\data\\final_model")
tokenizer2 = AutoTokenizer.from_pretrained("C:\\Users\\Tom\\Google Drive\\data\\final_model")
labels = torch.tensor([1]).unsqueeze(0)


def classify(text, percentfilter=False):
    inputs = tokenizer2(text, return_tensors="pt", truncation=True)
    a = model2(**inputs, labels=labels)
    t = a[1][0]
    probs = softmax(t.cpu().detach().numpy())
    argmax = np.argmax(probs)
    if (probs[argmax] > 0.9) or not percentfilter:
        return argmax + 1
    else:
        return None

import pandas as pd
testdata =pd.read_csv("data/test.csv")
labels = [classify(testdata.loc[i, "Narrative"]) for i in range(len(testdata))]
testdata["Abteilung"] = labels

#testdata.to_csv("submission.csv", index=False)