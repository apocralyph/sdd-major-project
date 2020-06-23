# Handling imports for gui, ocr and google translate modules
import os

os.environ["PYTORCH_JIT"] = "0"
from PyQt5 import QtWidgets
from mainWindow import Ui_MainWindow
from PIL import Image
from urllib.error import HTTPError
from translate import Translator
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract, cv2
from pytesseract import image_to_string
import re
import time
import sys
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from enchant.checker import SpellChecker

# Imports the model, as the project follows the view/model/controller pattern.
from model import Model, get_persons_list, bounding_box, mask_text, bounding_coords, predict_word


# UI class, the main program runs in here
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

        # uses a non-deterministic algorithm to determine the primary language, then reruns ocr to suit
        if self.model.readLang(text) == 'en':
            filename = self.model.getFileName()
            text = image_to_string(Image.open(filename))
            text_original = str(text)
            # dictionary of words used to clean up text for easy translation
            rep = {'\n': ' ', '\\': ' ', '\"': '"', '-': ' ', ',': ' , ', '.': ' . ', '!': ' ! ',
                   '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ',
                   '(': ' ( ', ')': ' ) ', "s'": "s '"}
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

            # make sure people's names don't get counted as incorrect words

            persons_list = get_persons_list(text)
            ignore_words = persons_list + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
            # using enchant.checker.SpellChecker, identify incorrect words
            spell = SpellChecker("en_US")
            words = text.split()
            incorrect_words = [w for w in words if not spell.check(w) and w not in ignore_words]
            # using enchant.checker.SpellChecker, get suggested replacements
            suggested_words = [spell.suggest(w) for w in incorrect_words]
            # replace incorrect words with [MASK]
            text, text_original = mask_text(incorrect_words, text, text_original)

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
            # Predict all tokens
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensor)

            text = predict_word(id_mask, tokenizer, suggested_words, text_original, predictions)
        else:
            text = pytesseract.image_to_string(Image.open(image_name), config='-l jpn')
            text = re.sub(" ", "", text)

        os.remove(image_name)
        self.originalTextBrowser.setText(text)

    # Detects blocks of text in an image to prioritise for OCR. Not very accurate
    # however it picks up small details that tesseract misses.
    # tesseract usually picks up on whatever this function misses.
    def text_detect(self):

        image = cv2.imread(self.model.getFileName())
        orig = image.copy()
        (H, W) = image.shape[:2]

        r_w = W / float(W)
        r_h = H / float(H)

        image = cv2.resize(image, (W, H))
        (H, W) = image.shape[:2]

        # Loads in 'layers' of functions used for text detection and probability calculations
        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # Loads the EAST text detection system
        self.debug_print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        end = time.time()

        # Shows an estimate of how long text detection took
        self.debug_print("[INFO] text detection took {:.6f} seconds".format(end - start))
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        bounding_coords(confidences, geometry, numCols, numRows, rects, scores)

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        bounding_box(boxes, orig, r_h, r_w)

        # Display the original image with green bounding boxes surrounding 'easy' words to detect
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
        # Tests whether or not the given file is valid
        if self.model.isValid(file_name):
            self.model.setFileName(self.lineEdit.text())
            self.refresh_all()
            self.read_img()
            self.text_detect()
        else:
            # Displays a generic error message
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

    # Function triggers when the translate button is pressed
    def translateSlot(self):
        text = self.originalTextBrowser.toPlainText()
        # Checks whether the text is already in English or not
        if self.model.readLang(text) == 'en':
            self.debug_print("Text already in English.")
            self.translatedTextBrowser.setText(text)
        else:
            text.join(text.split())
            text = re.sub(" ", "", text)
            # Attempts to translate twice, just in case of random network drop outs
            try:
                t_text = self.tr.translate(text)
                self.translatedTextBrowser.setText(t_text)
            except HTTPError:
                time.sleep(2)
                try:
                    t_text = self.tr.translate(text)
                    self.translatedTextBrowser.setText(t_text)
                except HTTPError:
                    self.debug_print("Failed twice.")

    # Function triggers when the browse button is pressed
    def browseSlot(self):
        # Opens a file select dialog box
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Jpeg Files (*.jpg)",
            options=options)
        # Checks to see whether the file exists and can be read
        if file_name:
            self.debug_print("setting file name: " + file_name)
            self.model.setFileName(file_name)
            self.refresh_all()
            self.open_img(file_name)

    # Opens the image, and attempts to read it
    def open_img(self, file_name):
        try:
            self.read_img()
            # If the image can be read, attempts to detect any text in it
            try:
                self.text_detect()
            except:
                # File is invalid if there is no readable text, display error message
                m = QtWidgets.QMessageBox()
                m.setText("File contains no readable text!\n" + file_name)
                m.setIcon(QtWidgets.QMessageBox.Warning)
                m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                     | QtWidgets.QMessageBox.Cancel)
                m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
                ret = m.exec_()
                self.lineEdit.setText("")
                self.refresh_all()
                self.debug_print("Invalid file specified: " + file_name)
        # File is invalid if there is no readable text, display error message
        except:
            m = QtWidgets.QMessageBox()
            m.setText("File contains no readable text: " + file_name)
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
