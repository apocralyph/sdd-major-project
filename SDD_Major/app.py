#Handling imports for gui, ocr and google translate modules
import os
os.environ["PYTORCH_JIT"] = "0" 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from mainWindow import Ui_MainWindow
from PIL import Image
from textblob import TextBlob
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
import nltk
from enchant.checker import SpellChecker
from difflib import SequenceMatcher

#Imports the model, as the project follows the view/model/controller pattern.
from model import Model

#UI class
class MainWindowUI(Ui_MainWindow):
	#Initialise the model
	def __init__(self):
		super().__init__()
		self.model = Model()
		self.tr = Translator(to_lang="en", from_lang="ja")

	#Use super function to allow inheritance from GUI
	def setupUi(self, MW):
		super().setupUi(MW)
		self.splitter.setSizes([300,0])

	#Function that performs ocr
	def readImg(self):
		image = cv2.imread(self.model.getFileName())
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.debugPrint(str(type(gray)))

		imageName = "{}.png".format(os.getpid())
		cv2.imwrite(imageName, gray)
		
		#run ocr in both languages
		text = pytesseract.image_to_string(Image.open(imageName), config=('-l eng+jpn'))
		
		#uses a non-deterministic algorithm to determine the primary language, then reruns ocr to suit
		if self.model.readLang(text) == 'en':
			filename = self.model.getFileName()
			text = image_to_string(Image.open(filename))
			textOriginal = str(text)
			# cleanup text
			rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
			        '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
			        '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
			        '(': ' ( ', ')': ' ) ', "s'": "s '"}
			rep = dict((re.escape(k), v) for k, v in rep.items()) 
			pattern = re.compile("|".join(rep.keys()))
			text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
			def getPersonsList(text):
			    personsList=[]
			    for sent in nltk.sent_tokenize(text):
			        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
			            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
			                personsList.insert(0, (chunk.leaves()[0][0]))
			    return list(set(personsList))
			personsList = getPersonsList(text)
			ignoreWords = personsList + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
			# using enchant.checker.SpellChecker, identify incorrect words
			spell = SpellChecker("en_US")
			words = text.split()
			incorrectWords = [w for w in words if not spell.check(w) and w not in ignoreWords]
			# using enchant.checker.SpellChecker, get suggested replacements
			suggestedWords = [spell.suggest(w) for w in incorrectWords]
			# replace incorrect words with [MASK]
			for w in incorrectWords:
			    text = text.replace(w, '[MASK]')
			    textOriginal = textOriginal.replace(w, '[MASK]')
			    
			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			tokenizedText = tokenizer.tokenize(text)
			indexedTokens = tokenizer.convert_tokens_to_ids(tokenizedText)
			IdMask = [i for i, e in enumerate(tokenizedText) if e == '[MASK]']
			# Create the segments tensors.
			IdSegments = [0] * len(tokenizedText)
			# Convert inputs to PyTorch tensors
			tokensTensor = torch.tensor([indexedTokens])
			segmentsTensor = torch.tensor([IdSegments])
			# Load pre-trained model (weights)
			model = BertForMaskedLM.from_pretrained('bert-base-uncased')
			model.eval()
			# Predict all tokens
			with torch.no_grad():
			    predictions = model(tokensTensor, segmentsTensor)
			#refine prediction by matching with proposals from SpellChecker
			def predict_word(textOriginal, predictions, maskids):
			    pred_words=[]
			    for i in range(len(IdMask)):
			        preds = torch.topk(predictions[0, IdMask[i]], k=90) 
			        indices = preds.indices.tolist()
			        list1 = tokenizer.convert_ids_to_tokens(indices)
			        list2 = suggestedWords[i]
			        simmax=0
			        predicted_token=''
			        for word1 in list1:
			            for word2 in list2:
			                s = SequenceMatcher(None, word1, word2).ratio()
			                if s is not None and s > simmax:
			                    simmax = s
			                    predicted_token = word1
			        textOriginal = textOriginal.replace('[MASK]', predicted_token, 1)
			    return textOriginal
			text = predict_word(textOriginal, predictions, IdMask)
		else:
			text = pytesseract.image_to_string(Image.open(imageName), config=('-l jpn'))
			text = re.sub(" ","",text)

			

		os.remove(imageName)
		#text = ''.join(text.split())
		self.originalTextBrowser.setText(text)



	def text_detect(self):
		ap = argparse.ArgumentParser()
		ap.add_argument("-east", "--east", type=str,
			help="path to input EAST text detector", default="frozen_east_text_detection.pb")
		ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
			help="minimum probability required to inspect a region")
		ap.add_argument("-w", "--width", type=int, default=320,
			help="resized image width (should be multiple of 32)")
		ap.add_argument("-e", "--height", type=int, default=320,
			help="resized image height (should be multiple of 32)")
		args = vars(ap.parse_args())

		image = cv2.imread(self.model.getFileName())
		orig = image.copy()
		(H, W) = image.shape[:2]

		(newW, newH) = (args["width"], args["height"])
		rW = W / float(newW)
		rH = H / float(newH)

		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]

		self.debugPrint("[INFO] loading EAST text detector...")
		net = cv2.dnn.readNet(args["east"])

		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		start = time.time()
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		end = time.time()

		self.debugPrint("[INFO] text detection took {:.6f} seconds".format(end - start))
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []

		for y in range(0, numRows):
			# extract the scores (probabilities), followed by the geometrical
			# data used to derive potential bounding box coordinates that
			# surround text
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]

			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have sufficient probability, ignore it
				if scoresData[x] < args["min_confidence"]:
					continue

				# compute the offset factor as our resulting feature maps will
				# be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)

				# extract the rotation angle for the prediction and then
				# compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				# use the geometry volume to derive the width and height of
				# the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]

				# compute both the starting and ending (x, y)-coordinates for
				# the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)

				# add the bounding box coordinates and probability score to
				# our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])

		boxes = non_max_suppression(np.array(rects), probs=confidences)
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# draw the bounding box on the image
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		
		cv2.imshow("Text Detection", orig)
		self.debugPrint("[INFO] detected language " + self.model.readLang(self.originalTextBrowser.toPlainText()))
		cv2.waitKey(0)

	#Setup a hidden debugging log
	def debugPrint(self, msg):
		self.debugTextBrowser.append(msg)

	#Basic function to empty the textboxes
	def refreshAll(self):
		self.lineEdit.setText(self.model.getFileName())

	#Function executes when enter is pressed, but only while the file path textbox is focused
	def returnPressedSlot(self):
		fileName = self.lineEdit.text()
		if self.model.isValid(fileName):
			self.model.setFileName(self.lineEdit.text())
			self.refreshAll()
			self.readImg()
			self.text_detect()
		else:
			m = QtWidgets.QMessageBox()
			m.setText("Invalid file name!\n" + fileName)
			m.setIcon(QtWidgets.QMessageBox.Warning)
			m.setStandardButtons(QtWidgets.QMessageBox.Ok 
								| QtWidgets.QMessageBox.Cancel)
			m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
			ret = m.exec_()
			self.lineEdit.setText("")
			self.refreshAll()
			self.debugPrint("Invalid file specified: " + fileName)

	def translateSlot(self):
		text = self.originalTextBrowser.toPlainText()
		if self.model.readLang(text) == 'en':
			self.debugPrint("Text already in English.")
			self.translatedTextBrowser.setText(text)
		else:
			text.join(text.split())
			text = re.sub(" ","",text)
			try:
				tText = self.tr.translate(text)
				self.translatedTextBrowser.setText(tText)
			except HTTPError:
				time.sleep(2)
				try:
					tText = self.tr.translate(text)
					self.translatedTextBrowser.setText(tText)
				except HTTPError:
					self.debugPrint("Failed twice.")

	def browseSlot(self):
		options = QtWidgets.QFileDialog.Options()
		options |= QtWidgets.QFileDialog.DontUseNativeDialog
		fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
								None,
								"QFileDialog.getOpenFileName()",
								"",
								"All Files (*);;Jpeg Files (*.jpg)",
								options = options)
		if fileName:
			self.debugPrint("setting file name: " + fileName)
			self.model.setFileName(fileName)
			self.refreshAll()
			try:
				self.readImg()
				try:
					self.text_detect()
				except:
					m = QtWidgets.QMessageBox()
					m.setText("Invalid file name!\n" + fileName)
					m.setIcon(QtWidgets.QMessageBox.Warning)
					m.setStandardButtons(QtWidgets.QMessageBox.Ok 
										| QtWidgets.QMessageBox.Cancel)
					m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
					ret = m.exec_()
					self.lineEdit.setText("")
					self.refreshAll()
					self.debugPrint("Invalid file specified: " + fileName)
			except:
				m = QtWidgets.QMessageBox()
				m.setText("File contains no readable text!\n" + fileName)
				m.setIcon(QtWidgets.QMessageBox.Warning)
				m.setStandardButtons(QtWidgets.QMessageBox.Ok 
									| QtWidgets.QMessageBox.Cancel)
				m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
				ret = m.exec_()
				self.lineEdit.setText("")
				self.refreshAll()
				self.debugPrint("File contains no readable text: " + fileName)

			

#Main function, first thing that the program runs
def main():
	#Locate trained data for ocr
	pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'
	#Load the GUI
	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	MainWindow = QtWidgets.QMainWindow()
	ui = MainWindowUI()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())

main()
