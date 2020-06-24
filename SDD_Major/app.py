# Handling all imports
import os

os.environ["PYTORCH_JIT"] = "0"
from PyQt5 import QtWidgets
from mainWindow import Ui_MainWindow
from PIL import Image
from urllib.error import HTTPError
from translate import Translator
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import pytesseract, cv2
from pytesseract import image_to_string
import re, time, sys
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from enchant.checker import SpellChecker

# Imports the model, as the project follows the view/model/controller pattern.
from model import Model, get_persons_list, text_replace, replace_incorrect, predict_word


# Function used to extract the scores (probabilities), followed
# by the geometrical data used to derive potential bounding box
# coordinates that surround text
def probability_score(args, confidences, geometry, numCols, numRows, rects, scores):
    for y in range(0, numRows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scores_data[x] < args["min_confidence"]:
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
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])


# Function used to evaluate language tokens
# See http://www.nltk.org for more info
def evaluate_tokens(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    id_mask = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']
    # Create the segments tensors.
    id_segments = [0] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([id_segments])
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    return id_mask, model, segments_tensor, tokenizer, tokens_tensor


class MainWindowUI(Ui_MainWindow):
    # Initialise the model
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.tr = Translator(to_lang="en", from_lang="ja")

    # Use super function to allow inheritance from GUI
    def setup_ui(self, mw):
        super().setup_ui(mw)
        self.splitter.setSizes([300, 0])

    # Function that performs ocr
    def read_img(self):
        image = cv2.imread(self.model.getFileName())

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_print(str(type(gray)))

        image_name = "{}.png".format(os.getpid())
        cv2.imwrite(image_name, gray)

        # run ocr in both languages
        text = pytesseract.image_to_string(Image.open(image_name), config='-l eng+jpn')

        text = self.read_lang(image_name, text)

        os.remove(image_name)
        self.originalTextBrowser.setText(text)

    # Non-deterministic algorithm to read the language present
    def read_lang(self, image_name, text):
        if self.model.readLang(text) == 'en':
            filename = self.model.getFileName()
            text = image_to_string(Image.open(filename))
            text_original = str(text)
            # cleanup text
            text = text_replace(text)

            persons_list = get_persons_list(text)
            ignore_words = persons_list + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
            # using enchant.checker.SpellChecker, identify incorrect words
            spell = SpellChecker("en_US")
            words = text.split()
            incorrect_words = [w for w in words if not spell.check(w) and w not in ignore_words]
            # using enchant.checker.SpellChecker, get suggested replacements
            suggested_words = [spell.suggest(w) for w in incorrect_words]
            # replace incorrect words with [MASK]
            text, text_original = replace_incorrect(incorrect_words, text, text_original)

            id_mask, model, segments_tensor, tokenizer, tokens_tensor = evaluate_tokens(text)
            # Predict all tokens
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensor)

            # Refine prediction by matching with proposals from SpellChecker
            text = predict_word(id_mask, tokenizer, suggested_words, text_original, predictions)
        else:
            text = pytesseract.image_to_string(Image.open(image_name), config='-l jpn')
            text = re.sub(" ", "", text)
        return text

    # Function to detect text in an image using EAST
    def text_detect(self):
        #                                                 ,  ,
        #                                                / \/ \
        #                                               (/ //_ \_
        #      .-._                                      \||  .  \
        #       \  '-._                            _,:__.-"/---\_ \
        #  ______/___  '.    .--------------------'~-'--.)__( , )\ \
        # `'--.___  _\  /    |             Here        ,'    \)|\ `\|
        #      /_.-' _\ \ _:,_          Be Dragons           " ||   (
        #    .'__ _.' \'-/,`-~`                                |/
        #        '. ___.> /=,| Abandon all hope, ye who enter! |
        #         / .-'/_ )  '---------------------------------'
        #         )'  ( /(/
        #              \\ "
        #               '=='
        # This horrible monstrosity takes what should be normal variables as
        # CLI arguments, but for some reason they don't work as normal variables.
        # It's bloated, confusing, and pretty awful by necessity(for the most part).
        # Due to the extremely volatile nature of this code, it should NOT be edited
        # unless absolutely necessary.

        # Defines an argument parser, which handles command line inputs
        ap = argparse.ArgumentParser()
        ap.add_argument("-east", "--east", type=str,
                        help="me", default="frozen_east_text_detection.pb")
        ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                        help="these shouldn't even be needed????")
        ap.add_argument("-w", "--width", type=int, default=320,
                        help="magic, do not touch")
        ap.add_argument("-e", "--height", type=int, default=320,
                        help="without this line, the program breaks")
        args = vars(ap.parse_args())

        # Reads image
        image = cv2.imread(self.model.getFileName())
        orig = image.copy()
        (H, W) = image.shape[:2]

        # Resize image based on given parameters
        (newW, newH) = (args["width"], args["height"])
        r_w = W / float(newW)
        r_h = H / float(newH)

        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # Load EAST
        self.debug_print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(args["east"])

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        end = time.time()

        # Time that the text detection took
        self.debug_print("[INFO] text detection took {:.6f} seconds".format(end - start))
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # Generate a probability score for each word to be a word
        probability_score(args, confidences, geometry, numCols, numRows, rects, scores)

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        for (start_x, start_y, end_x, end_y) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            start_x = int(start_x * r_w)
            start_y = int(start_y * r_h)
            end_x = int(end_x * r_w)
            end_y = int(end_y * r_h)

            # draw the bounding box on the image
            cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow("Text Detection", orig)
        self.debug_print("[INFO] detected language " + self.model.readLang(self.originalTextBrowser.toPlainText()))
        cv2.waitKey(0)

    # Setup a hidden debugging log
    def debug_print(self, msg):
        self.debugTextBrowser.append(msg)

    # Basic function to empty the textboxes
    def refresh_all(self):
        self.lineEdit.setText(self.model.getFileName())

    # Function executes when enter is pressed, but only while the file path textbox is focused
    def returnPressedSlot(self):
        file_name = self.lineEdit.text()
        if self.model.isValid(file_name):
            self.model.setFileName(self.lineEdit.text())
            self.refresh_all()
            self.read_img()
            self.text_detect()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + file_name)
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit.setText("")
            self.refresh_all()
            self.debug_print("Invalid file specified: " + file_name)

    # Function when translate is pressed
    def translateSlot(self):
        text = self.originalTextBrowser.toPlainText()
        if self.model.readLang(text) == 'en':
            self.debug_print("Text already in English.")
            self.translatedTextBrowser.setText(text)
        else:
            text.join(text.split())
            text = re.sub(" ", "", text)
            try:
                tText = self.tr.translate(text)
                self.translatedTextBrowser.setText(tText)
            except HTTPError:
                time.sleep(2)
                try:
                    tText = self.tr.translate(text)
                    self.translatedTextBrowser.setText(tText)
                except HTTPError:
                    self.debug_print("Failed twice.")

    # Function when browse is pressed
    def browseSlot(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Jpeg Files (*.jpg)",
            options=options)
        # This comment is self explanatory
        if file_name:
            self.debug_print("setting file name: " + file_name)
            self.model.setFileName(file_name)
            self.refresh_all()
            try:
                self.read_img()
                try:
                    self.text_detect()
                except:
                    m = QtWidgets.QMessageBox()
                    m.setText("Invalid file name!\n" + file_name)
                    m.setIcon(QtWidgets.QMessageBox.Warning)
                    m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                         | QtWidgets.QMessageBox.Cancel)
                    m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
                    ret = m.exec_()
                    self.lineEdit.setText("")
                    self.refresh_all()
                    self.debug_print("Invalid file specified: " + file_name)
            except:
                m = QtWidgets.QMessageBox()
                m.setText("File contains no readable text!\n" + file_name)
                m.setIcon(QtWidgets.QMessageBox.Warning)
                m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                     | QtWidgets.QMessageBox.Cancel)
                m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
                ret = m.exec_()
                self.lineEdit.setText("")
                self.refresh_all()
                self.debug_print("File contains no readable text: " + file_name)


# Main function, first thing that the program runs
def main():
    # Locate trained data for ocr
    pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'
    # Load the GUI
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    main_window = QtWidgets.QMainWindow()
    ui = MainWindowUI()
    ui.setup_ui(main_window)
    main_window.show()
    sys.exit(app.exec_())


main()
