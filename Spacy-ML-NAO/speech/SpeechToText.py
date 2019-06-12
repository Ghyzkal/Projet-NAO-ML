import speech_recognition as sr
import TextToSpeech as ts

def conversion():	
	r = sr.Recognizer()
	with sr.Microphone() as source:
		r.adjust_for_ambient_noise(source)
		#print("Comment Puis-je vous aider ? :")
		audio = r.listen(source)
		try:
			text = format(r.recognize_google(audio, language='fr-FR'))
			#ts.conversion("Vous avez choisi "+text)
			return text.lower()
		except:
			return "Navré, je ne vous comprends pas"



