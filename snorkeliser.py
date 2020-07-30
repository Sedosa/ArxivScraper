import sys
import os
import logging
import re
import math
sys.path.insert(0,os.getcwd())

from ArxivScraper import ArxivScraper
from snorkel.labeling.lf import labeling_function,LabelingFunction
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import spacy
import pandas as pd

###############################################################
## SCRAPE DATA FROM ARXIV
###############################################################

scraper = ArxivScraper(search_terms=["machine","learning","neural","networks"],max_results=1000,write_csv=True)
data = scraper.scrape()
dev_set= data.iloc[(math.floor(2*len(data)/3)):]
train_set = data.iloc[:(math.floor(2*len(data)/3))]
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

###############################################################
## WRITING LABELLING FUNCTIONS
###############################################################

from snorkel.preprocess.nlp import SpacyPreprocessor

try:
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)
except IOError:
    spacy.cli.download("en_core_web_sm")
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

from snorkel.labeling.lf.nlp import nlp_labeling_function

def keyword_lookup(x, keywords, field, label):
    """Given a list of tuples, look for any of a list of keywords"""
    if field in x and x[field] and any(word.lower() in x[field].lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, field, label=ABSTAIN, separate=False):
    """Given a list of keywords and a label, return a keyword search LabelingFunction"""
    prefix = 'separate_' if separate else ''
    name = f'{prefix}{keywords[0]}_field_{field}'        
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(
            keywords=keywords,
            field=field,
            label=label,
        ),
    )

def make_keyword_lfs(keywords, fields, label=ABSTAIN, separate=False):
    """Given a list of keywords and fields, make one or more LabelingFunctions for the keywords with each field"""
    lfs = []
    for field in fields:
        
        # Optionally make one LF per keyword
        if separate:
            for i, keyword in enumerate(keywords):
                lfs.append(
                    make_keyword_lf(
                        [keyword],
                        field,
                        label=label,
                        separate=separate,
                    )
                )
        # Optionally group keywords in a single LF for each field
        else:
            lfs.append(
                make_keyword_lf(
                    keywords,
                    field,
                    label=label,
                )
            )
    return lfs

#Making a labelling function programmatically taken from -> https://github.com/rjurney/amazon_open_source/blob/master/3_Amazon_Open_Source_Analysis.ipynb
any_ai_lf = make_keyword_lf(keywords=["artificial intelligence","proceedings","generative"],field="text",label=POSITIVE)

@labeling_function()
def ai_positive_lf(x):
    return POSITIVE if "artificial intelligence" in x.text.lower() else ABSTAIN

@labeling_function()
def proceedings_lf(x):
    return POSITIVE if "proceeedings" in x.text.lower() else ABSTAIN

@labeling_function()
def generative_lf(x):
    return POSITIVE if "generative" in x.text.lower() else ABSTAIN

@labeling_function()
def gaussian_lf(x):
    return NEGATIVE if "gaussian" in x.text.lower() else ABSTAIN

@labeling_function()
def synthesis_lf(x):
    return POSITIVE if any([s==x.text.lower() for s in ['synthesis','sampling','data generation']]) else ABSTAIN

@nlp_labeling_function()
def has_company_lf(x):
    if any([ent.label_=="ORG" for ent in x.doc.ents]):
        return POSITIVE
    else:
        return ABSTAIN

@labeling_function()
def has_comparison_lf(x):
    return NEGATIVE if not any([s==x.text.lower() for s in ["compare","comparison","improvement"]]) else ABSTAIN

lfs = [ai_positive_lf,
        proceedings_lf,
        generative_lf,
        synthesis_lf,
        gaussian_lf,
        has_company_lf,
        any_ai_lf,
        has_comparison_lf]

###############################################################
## APPYLING LABELLING FUNCTIONS TO TRAIN & DEV SETS
###############################################################

applier = PandasLFApplier(lfs=lfs)
processed_train_data = applier.apply(data)
processed_dev_data = applier.apply(data)
logging.info("applied labelling functions to scraped data")
print(LFAnalysis(L=processed_train_data,lfs=lfs).lf_summary())

###############################################################
## FITTING THE GENERATIVE MODELAND PREDICTING

###############################################################
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=processed_train_data, n_epochs=500, log_freq=100, seed=123)

pred_LM_train = label_model.predict(processed_dev_data)
logging.info("generated noisy labels")
logging.info("writing to DataFrame")

###############################################################
## OUTPUTS
###############################################################
pred_frame = pd.DataFrame(data ={'title':data['title'],'Prediction':pred_LM_train})
print(pred_frame['Prediction'].value_counts())

filter_ = pred_frame['Prediction']==1
pred_frame = pred_frame.loc[filter_]
print(pred_frame)



print(f"Label Model Voting : {pred_frame2.shape}" )
print(pred_frame2)

pred_frame2.to_csv("LabelModelResults.csv")
