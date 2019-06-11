from gtts import gTTS


def conversion(text):
	speech = gTTS(text,lang='fr-FR', slow)
	speech.save('speech.mp3')
