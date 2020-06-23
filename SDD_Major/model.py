import re

import nltk
from langdetect import detect
from langdetect import DetectorFactory


class Model:
    def __init__(self):
        self.fileName = None

    def isValid(self, fileName):
        try:
            file = open(fileName, 'r')
            file.close()
            return True
        except IOError:
            return False

    def setFileName(self, fileName):
        if self.isValid(fileName):
            self.fileName = fileName
        else:
            self.fileName = ""

    def getFileName(self):
        return self.fileName

    def readLang(self, text):
        DetectorFactory.seed = 0
        return detect(text)


def get_persons_list(names):
    person_list = []
    for sent in nltk.sent_tokenize(names):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                person_list.insert(0, (chunk.leaves()[0][0]))
    return list(set(person_list))


def text_replace(text):
    rep = {'\n': ' ', '\\': ' ', '-': ' ', '"': ' " ', ',': ' , ', '.': ' . ', '!': ' ! ',
           '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ',
           '(': ' ( ', ')': ' ) ', "s'": "s '"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text


def replace_incorrect(incorrect_words, text, text_original):
    for w in incorrect_words:
        text = text.replace(w, '[MASK]')
        text_original = text_original.replace(w, '[MASK]')
    return text, text_original
