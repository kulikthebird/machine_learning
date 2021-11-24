#!/usr/bin/env python3

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.catch_warnings()
warnings.filterwarnings("ignore",category=FutureWarning)


import re
import numpy as np
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle


#TODO: do something with these functions, they are needed in pickled objects:
def LemmaTokenizer(articles):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in word_tokenize(articles)]

def filter_function(x):
    return re.sub(r'(_)|(\d[0-9.]*)', ' ', x.lower())



class Classifier():
    def __init__(self, model_file, weights_file, text_transformer_file):
        with open(model_file, 'r') as json_file:
            self._model = model_from_json(json_file.read())
            self._model.load_weights(weights_file)
            print("Loaded model from disk: ", self._type())
            with open(text_transformer_file, 'rb') as f:
                self._text_transformer = pickle.load(f, encoding='latin1')
                return
        raise Exception("Couldn't load model")

    def clasify(self, article):
        raise NotImplementedError()

    def _type(self):
        raise NotImplementedError()

    def get_name_of_the_class(self, c):
        names = ['alt.atheism',
                 'comp.graphics',
                 'comp.os.ms-windows.misc',
                 'comp.sys.ibm.pc.hardware',
                 'comp.sys.mac.hardware',
                 'comp.windows.x',
                 'misc.forsale',
                 'rec.autos',
                 'rec.motorcycles',
                 'rec.sport.baseball',
                 'rec.sport.hockey',
                 'sci.crypt',
                 'sci.electronics',
                 'sci.med',
                 'sci.space',
                 'soc.religion.christian',
                 'talk.politics.guns',
                 'talk.politics.mideast',
                 'talk.politics.misc',
                 'talk.religion.misc']
        return names[c]
        



class RNN(Classifier):
    def __init__(self):
        super().__init__('model/model_rnn.json', 'model/model_rnn.h5', 'model/tokenizer_rnn.pkl')

    def classify(self, article):
        max_review_length = 1250
        text_to_classify = sequence.pad_sequences(self._text_transformer.texts_to_sequences([article]), maxlen=max_review_length)
        result = self._model.predict(text_to_classify)
        return self.get_name_of_the_class(np.argmax(result[0]))

    def _type(self):
        return self.__class__.__name__


class TFIDF(Classifier):
    def __init__(self):
        super().__init__('model/model_tfidf.json', 'model/model_tfidf.h5', 'model/vectorizer_tfidf.pkl')

    def classify(self, article):
        text_to_classify = self._text_transformer.transform([article])
        result = self._model.predict(text_to_classify)
        return self.get_name_of_the_class(np.argmax(result[0]))

    def _type(self):
        return self.__class__.__name__



if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        art = f.read()
        result1 = RNN().classify(art)
        result2 = TFIDF().classify(art)
        print("\n#########################")
        print("#########################")
        print("#########################")
        print("#########################\n")
        print("RNN result: ", result1)
        print("TFIDF result: ", result2)

