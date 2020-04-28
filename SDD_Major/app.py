#Handling imports for gui, ocr and google translate modules
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from mainWindow import Ui_MainWindow
from PIL import Image
from textblob import TextBlob
import pytesseract, argparse, cv2, os, goslate, sys
import re

#Imports the model, as the project follows the view/model/controller pattern.
from model import Model

#UI class
class MainWindowUI(Ui_MainWindow):
	#Initialise the model
	def __init__(self):
		super().__init__()
		self.model = Model()

	#Function that performs ocr
	def readImg(self):
		ap = argparse.ArgumentParser()
		ap.add_argument("-p", "--preprocess", type=str, default="thresh",
			help="type of preprocessing to be done")
		args = vars(ap.parse_args())
		try:
			image = cv2.imread(self.model.getFileName())
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cv2.imshow("Image", gray)
		except:
			m = QtWidgets.QMessageBox()
			m.setText("Invalid file!")
			m.setIcon(QtWidgets.QMessageBox.Warning)
			m.setStandardButtons(QtWidgets.QMessageBox.Ok 
								| QtWidgets.QMessageBox.Cancel)
			m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
			ret = m.exec_()
			self.lineEdit.setText("")
			self.debugPrint("Invalid file specified!")
			return

		#Potential for preprocessing images
		# if args["preprocess"] == "thresh":
		# 	gray = cv2.threshold(gray, 0, 255,
		# 		cv2.THRESH_BINARY / cv2.THRESH_OTSU)[1]
		# elif args["preprocess"] == "blur":
		# 	gray = cv2.medianBlur(gray, 3)

		imageName = "{}.png".format(os.getpid())
		cv2.imwrite(imageName, gray)
		
		#run ocr in both languages
		text = pytesseract.image_to_string(Image.open(imageName), config=('-l eng+jpn'))
		
		#uses a non-deterministic algorithm to determine the primary language, then reruns ocr to suit

		if self.model.readLang(text) == 'en':
		 	text = pytesseract.image_to_string(Image.open(imageName), config=('-l eng'))
		elif self.model.readLang(text) == 'jp':
		 	text = pytesseract.image_to_string(Image.open(imageName), config=('-l jpn'))
		 	text = re.sub("\s","",text)

		os.remove(imageName)
		#text = ''.join(text.split())
		self.originalTextBrowser.setText(text)
		gs = goslate.Goslate()
		translatedText = gs.translate(text, 'en')
		self.translatedTextBrowser.setText(translatedText)
		cv2.imshow("Output", gray)
		cv2.waitKey(0)


	#Use super function to allow inheritance from GUI
	def setupUi(self, MW):
		super().setupUi(MW)
		self.splitter.setSizes([300,0])

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
			self.readImg()

#Main function, first thing that the program runs
def main():
	#Locate trained data for ocr
	pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR\\tesseract.exe'
	gs = goslate.Goslate()
	#Load the GUI
	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	MainWindow = QtWidgets.QMainWindow()
	ui = MainWindowUI()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())

main()