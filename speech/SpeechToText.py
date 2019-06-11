import speech_recognition as sr


def conversion():	
	r = sr.Recognizer()
	with sr.Microphone() as source:
		r.adjust_for_ambient_noise(source)
		print("Comment Puis-je vous aider ? :")
		audio = r.listen(source)
		try:
			text = r.recognize_google(audio, language='fr-FR')
			return format(text)
		except:
			return "Navré, je ne vous comprends pas"



