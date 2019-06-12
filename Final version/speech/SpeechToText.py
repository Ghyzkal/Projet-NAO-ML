import speech_recognition as sr


def conversion():	
	r = sr.Recognizer()
	with sr.Microphone() as source:
		r.adjust_for_ambient_noise(source)
		#print("Comment Puis-je vous aider ? :")
		audio = r.listen(source)
		try:
			return format(r.recognize_google(audio, language='fr-FR')).lower()
		except:
			return "Invalid String"



