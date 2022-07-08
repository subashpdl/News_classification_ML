#!python3
# -*- coding: utf-8 -*-
# pylint: disable=W0312, line-too-long, C0103

import unicodedata
import re
from nltk.util import ngrams
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords  # Import the stop word list

# load stop words and stemmeers
default_stopwords = set(stopwords.words("english"))
default_stopwords.discard("not") # not is not a stopword
default_stopwords.discard("very")
default_stopwords.discard("further")
default_stopwords.discard("just")
default_stopwords.discard("more")
default_stopwords.discard("most")
default_stopwords.discard("only")
default_stopwords.discard("other")
default_stopwords.discard("too")
default_stemmer = SnowballStemmer("english")

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub("cannot", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def negate_and_stem(text, stemmer=None, stopwords=[], use_negation=False):
    delims = ["?",".",",","!","?",":",";","but"]
    negations = ["not"] # maybe "no" or "never"
    neg = False
    text = text.lower()

    words = text.split()

    result = []

    for word in words:
        # original world is only kept for reference for detecting end of negations!
        cleaned_word = re.sub(r"[^a-z0-9 ]", "", word)
        cleaned_word = re.sub(r"^\d$", "ONEDIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^\d\d$", "TWODIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^\d\d\d$", "THREEDIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^\d\d\d\d$", "FOURDIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^\d\d\d\d\d$", "FIVEDIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^\d+$", "LONGDIGITNUMBER", cleaned_word)
        cleaned_word = re.sub(r"^(?=.*[0-9])(?=.*[a-zA-Z])([a-zA-Z0-9]+)$", "ALPHANUMERICWORD", cleaned_word)

        if not cleaned_word: continue

        if not use_negation:
            if stemmer:
                cleaned_word = stemmer.stem(cleaned_word)
            if cleaned_word not in stopwords:
                result.append(cleaned_word)
        elif cleaned_word in negations:
            # use_negation is active
            neg = not neg
            # start of negation (ignore the negation word itself)
        elif cleaned_word not in stopwords:
            # use of negation is active
            # not a stop word and neither negation word
            if stemmer:
                cleaned_word = stemmer.stem(cleaned_word)

            if neg:
                result.append("neg_" + cleaned_word)
            else:
                result.append(cleaned_word)

        # detect end of negation
        if any(d in word for d in delims):
            neg = False


    return " ".join(result)

def clean_raw_text(rawtext, params):
    """cleans a word. replaces umlauts, accents, removes stop words, creates ngrams, etc.
    returns a string
    """
    stopwords = []
    if params["stopwords"]:
        stopwords = default_stopwords

    stemmer = None
    if params["stemmer"]:
        stemmer = default_stemmer
    
    if "negation" not in params:
        params["negation"] = False

        
    rawtext = str(rawtext)

    rawtext = rawtext.replace("ä", "ae")
    rawtext = rawtext.replace("Ä", "Ae")
    rawtext = rawtext.replace("ü", "ue")
    rawtext = rawtext.replace("Ü", "Ue")
    rawtext = rawtext.replace("ö", "oe")
    rawtext = rawtext.replace("Ö", "Oe")
    rawtext = rawtext.replace("ß", "ss")

    #accents
    rawtext = unicodedata.normalize('NFKD', rawtext).encode('ascii', 'ignore').decode()

    #lower
    rawtext = rawtext.lower()

    #remove things
    rawtext = re.sub(r"[\r\n]", " ", rawtext)
    rawtext = re.sub(r"\s+", " ", rawtext)
    rawtext = rawtext.replace("/", " ")

    #decontract
    rawtext = decontracted(rawtext)

    #negate and stem
    rawtext = negate_and_stem(rawtext, stemmer=stemmer, stopwords=stopwords, use_negation=params["negation"])

    #split into individual words
    words = rawtext.split()

    #ngrams?
    #we add ngrams manually
    useNgrams = False
    if useNgrams:
    	ngram_words = []
    	for word in words:
    		chrs = [c for c in word] #split
    		# ngram_words.append(word) #decide: add the original world or solely use ngrams?

    		ngram_words.extend(["".join(i) for i in ngrams(chrs, 1)])
    		ngram_words.extend(["".join(i) for i in ngrams(chrs, 2)])
    		ngram_words.extend(["".join(i) for i in ngrams(chrs, 3)])
    		ngram_words.extend(["".join(i) for i in ngrams(chrs, 4)])

    		words = ngram_words

    #make a string again
    return( " ".join( words ))
