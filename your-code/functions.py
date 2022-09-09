import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def clean_up(s):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    
    s = s.lower()
    # URLs
    s = re.sub(r'(http[s]?://[a-z0-9]+\.[a-z]+)'," ", s)
    # Special characters
    s = re.sub(r'(\W|\_|[À-ű]+)'," ", s)
    # Digits
    s = re.sub(r'(\d)'," ", s)
    # 2+ white spaces to 1
    s = re.sub(r'(\s+)'," ", s)
    # White spaces at the beginning and end of the sentence
    s = re.sub(r'(^\s|\s$)',"", s)
       
    return s

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    
    return word_tokenize(s)

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    
    stem = PorterStemmer()
    lem = WordNetLemmatizer()
    
    return [lem.lemmatize(stem.stem(word)) for word in l]

stop_words = list(stopwords.words("english"))
stop_words = [clean_up(x) for x in stop_words]

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    
    return [word for word in l if word not in stop_words]

import pickle

filename = "corpus.pkl"
with open(filename, "rb") as file:
    corpus = pickle.load(file)

def find_features(doc):
    
    row = []
    [row.append((item in doc)) for item in corpus]
    
    return row