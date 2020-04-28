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

	def readLang(self,text):
		DetectorFactory.seed = 0
		return detect(text)