import re
from difflib import SequenceMatcher

import nltk
import torch
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


def predict_word(id_mask, tokenizer, suggested_words, original_text, predicts):
    for i in range(len(id_mask)):
        preds = torch.topk(predicts[0, id_mask[i]], k=90)
        indices = preds.indices.tolist()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        list2 = suggested_words[i]
        sim_max = 0
        predicted_token = ''
        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, word1, word2).ratio()
                if s is not None and s > sim_max:
                    sim_max = s
                    predicted_token = word1
        original_text = original_text.replace('[MASK]', predicted_token, 1)
    return original_text
