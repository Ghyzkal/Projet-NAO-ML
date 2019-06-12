import pyttsx3

def conversion(text):
	speech = pyttsx3.init()
	speech.say(text)
	speech.runAndWait()

