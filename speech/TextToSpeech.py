from gtts import gTTS


def conversion(text):
	speech = gTTS(text,lang='fr-FR')
	speech.save('speech.mp3')
