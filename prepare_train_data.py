#!python3
# -*- coding: utf-8 -*-
# pylint: disable=W0312, line-too-long, C0103


import custom_data_cleaner
import json
from sklearn.datasets import fetch_20newsgroups
import re


def mydataextractor(rawtext):
    """ The 20 Newsgroup dataset has large headers. This function tries to remove them. """
    # skip first 6 lines which are headers
    # headers and footers removed by call
    # remove "in article ... write?/wrote :"
    # remove "from: "
    # remove "Date: "
    # remove "Subject: "
    # remove "To: "
    # remove "...@..."
    finalstring = ""
    for line in rawtext.split("\n"):
        lline = line.lower().strip()
        if not lline:
            finalstring += "\n"
            continue
        if "from: " in lline or "date: " in lline or "subject: " in lline or "to: " in lline:
            continue
        if "in article" in lline and ("write" in lline or "wrote" in lline):
            continue
        if lline.startswith(">") or lline.startswith("|>"):
            continue
        finalstring += "\n" + re.sub(".+@.+", "", line)
    return finalstring


cats = None
twentyng = fetch_20newsgroups(categories=cats, subset='all',
                              shuffle=True, random_state=43, remove=('headers', 'footers'))

texts = []  # list of text samples
texts_cleaned = []  # list of text samples
label_id_to_str = {}  # dictionary mapping label name to numeric id
label_str_to_id = {}  # dictionary mapping label name to numeric id
labels_id = []  # list of label ids
labels_str = []  # list of label ids


# transform label-output to integers
data = {}
data["data"] = []
for i in range(len(twentyng.data)):
    label = twentyng.target_names[twentyng.target[i]]
    if label in label_str_to_id:
        label_id = label_str_to_id.get(label)
    else:
        label_id = len(label_str_to_id)
        label_str_to_id[label] = label_id
        label_id_to_str[label_id] = label

    # 20ng has a lot of headers, remove them!
    extracted_text = mydataextractor(twentyng.data[i])
    this_entry = {}
    # this_entry["raw"] = extracted_text
    this_entry["cleaned"] = custom_data_cleaner.clean_raw_text(extracted_text, {"stopwords": True, "stemmer": False})
    this_entry["cleaned_including_stopwords"] = custom_data_cleaner.clean_raw_text(extracted_text, {"stopwords": False, "stemmer": False})
    this_entry["cleaned_stemmed"] = custom_data_cleaner.clean_raw_text(extracted_text, {"stopwords": True, "stemmer": True})
    this_entry["label_id"] = label_id
    this_entry["label_str"] = label

    # print(this_entry)
    # if len(data["data"])>10:
    #     raise

    data["data"].append(this_entry)

data["label_id_to_str"] = label_id_to_str
data["label_str_to_id"] = label_str_to_id

print(data["data"][0])

with open("twentyng_processed.json", "w") as f:
    json.dump(data, f)
