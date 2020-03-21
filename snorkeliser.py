import sys
import os
import logging

sys.path.insert(0,os.getcwd())

from ArxivScraper import ArxivScraper
from snorkel.labeling.lf import labeling_function
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel

import pandas as pd

scraper = ArxivScraper(search_terms=["machine","learning","neural","networks"],max_results=10000,write_csv=True)
data = scraper.scrape()

POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
#######################################
''' 
If you get an error with spacy  below such as cannot find "en_core_web_sm", execute this in shell

"python -m spacy download en_core_web_sm"
'''
from snorkel.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)
from snorkel.labeling.lf.nlp import nlp_labeling_function

@labeling_function()
def ai_positive(x):
    return POSITIVE if "artificial intelligence" in x.text.lower() else ABSTAIN

@labeling_function()
def proceedings(x):
    return POSITIVE if "proceeedings" in x.text.lower() else ABSTAIN

@labeling_function()
def generative(x):
    return POSITIVE if "generative" in x.text.lower() else ABSTAIN

@labeling_function()
def has_synthesis(x):
    return POSITIVE if any([s==x.text.lower() for s in ['synthesis','sampling','data generation']]) else ABSTAIN

@nlp_labeling_function()
def has_company(x):
    if any([ent.label_=="ORG" for ent in x.doc.ents]):
        return POSITIVE
    else:
        return ABSTAIN


lfs = [ai_positive,proceedings,generative,has_synthesis,has_company]

applier = PandasLFApplier(lfs=lfs)
processed_data = applier.apply(data)

logging.info("applied labelling functions to scraped data")
print(LFAnalysis(L=processed_data,lfs=lfs).lf_summary())

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=processed_data, n_epochs=500, log_freq=100, seed=123)

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=processed_data)
pred_LM_train = label_model.predict(processed_data)
logging.info("generated noisy labels")
logging.info("writting to DataFrame")

pred_frame = pd.DataFrame(data ={'title':data['title'],'Prediction':preds_train})
print(pred_frame['Prediction'].value_counts(normalize=True))
pred_frame2 = pd.DataFrame(data ={'title':data['title'],'Prediction':pred_LM_train})

filter = pred_frame['Prediction']==1
filter2 = pred_frame2['Prediction']==1
print("Majority Label Voting :")
print(pred_frame.loc[filter])
print("Label Model Voting :" )
print(pred_frame2.loc[filter2])

pred_frame.loc[filter].to_csv("MajoritylModelResults.csv")
pred_frame2.loc[filter2].to_csv("LabelModelResults.csv")
