import sys
import os
import logging

sys.path.insert(0,os.getcwd())

from ArxivScraper import ArxivScraper
from snorkel.labeling.lf import labeling_function
from snorkel.labeling import PandasLFApplier, LFAnalysis

scraper = ArxivScraper(max_results=1000,write_csv=True)
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
    return POSITIVE if any([s==x.text.lower() for s in ['sythesis','sampling','data generation']]) else ABSTAIN

@nlp_labeling_function()
def has_company(x):
    if any([ent.label_=="ORG" for ent in x.doc.ents]):
        return NEGATIVE
    else:
        return ABSTAIN


lfs = [ai_positive,proceedings,generative,has_nuclear,is_social,has_synthesis,has_company]

applier = PandasLFApplier(lfs=lfs)
processed_data = applier.apply(data)

logging.info("applied labelling functions to scraped data")
print(LFAnalysis(L=processed_data,lfs=lfs).lf_summary())

from snorkel.labeling import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=processed_data)
logging.info("generated noisy labels")
logging.info("writting to DataFrame")
import pandas as pd
pred_frame = pd.DataFrame(data ={'title':data['title'],'Prediction':preds_train})
print(pred_frame['Prediction'].value_counts(normalize=True))

filter = pred_frame['Prediction']==1
print(pred_frame.loc[filter])
