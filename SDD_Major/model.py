from difflib import SequenceMatcher

import cv2
import nltk
import numpy as np
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


# Gets a list of common names from NLTK to ignore from spell checks
def get_persons_list(names):
    person_list = []
    for sent in nltk.sent_tokenize(names):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                person_list.insert(0, (chunk.leaves()[0][0]))
    return list(set(person_list))


# Defines 'bounding boxes' which are green boxes that
# highlight text that is obvious for the OCR to pick up on
def bounding_box(boxes, orig, r_h, r_w):
    for (start_x, start_y, end_x, end_y) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        start_x = int(start_x * r_w)
        start_y = int(start_y * r_h)
        end_x = int(end_x * r_w)
        end_y = int(end_y * r_h)

        # draw the bounding box on the image
        cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


# Function to loop through defined columns based on image width to check for text
def column_loop(angles_data, confidences, num_cols, col_rect, scores_data, x_data0, x_data1, x_data2,
                x_data3, y):
    for x in range(0, num_cols):
        # if our score does not have sufficient probability, ignore it
        if scores_data[x] < float(0.5):
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = angles_data[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = x_data0[x] + x_data2[x]
        w = x_data1[x] + x_data3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
        end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
        start_x = int(end_x - w)
        start_y = int(end_y - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        col_rect.append((start_x, start_y, end_x, end_y))
        confidences.append(scores_data[x])


# Replaces unknown or misspelled words with [MASK] for future replacement
def mask_text(incorrect_words, text, text_original):
    for w in incorrect_words:
        text = text.replace(w, '[MASK]')
        text_original = text_original.replace(w, '[MASK]')
    return text, text_original


# extract the scores (probabilities), followed by the geometrical
# data used to derive potential bounding box coordinates that
# surround text
def bounding_coords(confidences, geometry, num_cols, num_rows, rects, scores):
    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # loop over the number of columns
        column_loop(angles_data, confidences, num_cols, rects, scores_data, x_data0, x_data1, x_data2,
                    x_data3, y)


# refine prediction by matching with proposals from SpellChecker
def predict_word(id_mask, tokenizer, suggested_words, original_text, predicts):
    for i in range(len(id_mask)):
        predict = torch.topk(predicts[0, id_mask[i]], k=90)
        indices = predict.indices.tolist()
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
