#Meet NAO: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import warnings
import sys
from speech import SpeechToText as st
from speech import TextToSpeech as ts
warnings.filterwarnings('ignore')

#librairies a importer avec pip avant utilisation ici
import numpy as np
import spacy
import fr_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer #scikit + TfidfVectorizer + cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# Mots d'accueil
GREETING_INPUTS = ("hello", "salut", "coucou", "bonjour", "ca va ?","hey")
GREETING_RESPONSES = ["salut", "bonjour", "quoi de neuf docteur ?", "mes hommages", "hello"]

# Mots d'accueil
THX_INPUTS = ("merci", "merci beaucoup", "super", "gracias", "cimer","bsartek")
THX_RESPONSES = ["avec plaisir", "a vot' service !"]

# Selection des thèmes (documents)
MAP_THEMES ={"orientation":"Vous pouvez poser des questions sur les sujets suivant : la réorientation, les concours et plus \n", "pratique":"Vous pouvez poser des questions sur les sujets suivant : carte étudiante \t\t soutien \t\t ENT \n fonctionnement \t\t notes et cours \t\t stage \n association \t\t et autres services proposés par l'université\n", "erasmus":"Découvrez-en plus sur le programme Erasmus : à qui s'adresser, aides financières et plus \n", "bye":"bye"}
MENU_INPUTS = ("orientation", "pratique", "erasmus")
MENU_RESPONSES = ["data-orientation.txt", "data-pratiques.txt", "data-erasmus.txt"]
#MENU_TEXTS = ("\nQuelle thématique vous intéresse ? "+"\n dites Orientation afin d'en savoir plus sur la réorientation et les concours."+"\n• Dites Pratique pour en savoir plus sur la carte étudiante, l'ENT et  le fonctionnement des notes et des cours"+"\n•dites Erasmus afin d'en savoir plus sur le programme Erasmus")
MENU_TEXTS = ("Menu Orientation Pratique Erasmus")

monFichierDeDonnee = ""
nlp = spacy.load('fr_core_news_md') 
sent_tokens = ""
themeLoaded = 0
stopWords = spacy.lang.fr.stop_words.STOP_WORDS


# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def greeting(sentence):
    """Si l'utilisateur envoie une salutation, répondre par une salutation"""
    for word in sentence.split():
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def remerciement(sentence):
    """Si l'utilisateur envoie une salutation, répondre par une salutation"""
    for word in sentence.split():
        if word in THX_INPUTS:
            return random.choice(THX_RESPONSES)



# Générer une réponse
def response(user_response):
    NAO_response=''
    global sent_tokens
    global stopWords
    sent_tokens.append(user_response)
    #On initialize TfidfVectorizer. 
    #En particulier, on passe à TfIdfVectorizer notre propre fonction de tokenisation et d'analyse syntaxique
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopWords)
    #fit_transform fait 2 choses : créer un dictionnaire de mots provenant du document source et calculer le tfidf pour chaque mot du document source 
    #Ainsi on crée une matrice pour laquelle en ordonnée on a les phrases du document source, en colonne les mots et à leur intersection : le tfidf si le mot apparaît dans la phrase, 0 sinon
    tfidf = vectorizer.fit_transform(sent_tokens)
    #On utilise ici la similarité cosinus pour comparer le document source à la requete de l'utilisateur
    #on compare tfidf[-1] qui est la question de l'utilisateur et tfidf qui est notre base de référence
    vals = cosine_similarity(tfidf[-1], tfidf)
    #on trie le tableau et on récupère l'indice des n plus grands éléments
    #argsort()[][n] permet de trier la matrice afin d'obtenir les n plus grands éléments (plus hauts scores tfidf)
    idx=vals.argsort()[0][-2]
    #on transfère toutes les valeurs dans un array de dimension 1 qu'on va ensuite trier pour ne garder que les 2 plus grandes valeurs
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        NAO_response=NAO_response + "Désolé ! Je n'ai pas compris."
        return NAO_response
    else:
        NAO_response = NAO_response + sent_tokens[idx] #indice de la phrase/réponse qui a le plus au score de correspondance avec la question
        return NAO_response

def greetingOrRemerciement(user_response):
    if (greeting(user_response)):
        return greeting(user_response)
    elif (remerciement(user_response)):
        return remerciement(user_response)


def selectionTheme(sentence):
    """Si l'utilisateur envoie une selection de theme, met à jour le nom du document source pour qu'il pointe vers ce thème"""
    global monFichierDeDonnee
    for word in sentence.split():
        if word in MAP_THEMES: 
            monFichierDeDonnee = "data-"+word+".txt"
            return  MAP_THEMES.get(word)

#Lecture de la source de données      
def get_tokens():
    with open(monFichierDeDonnee, 'r', encoding='utf8', errors ='ignore') as maBDD:
        raw = maBDD.read().lower()
    #remove the punctuation using the character deletion step of translate
    #Tokenisation
    global sent_tokens

    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    return sent_tokens

#Main
def main ():
    nao("Mon nom é NAO. Je répondrai à vos questions sur la MIAGE de Nanterre Université. Pour quitter, dites Bye")
    nao("Pour choisir un thème,dites Menu")
    
    global themeLoaded
    global sent_tokens
    themeLoaded=0

    navigator=Navigator()
    while(True):
        user_response = st.conversion()
        printUser(user_response)

        if(navigator.indirect(user_response)=="Invalid" and themeLoaded==1):
            nao(response(user_response))
            sent_tokens.remove(user_response)


def nao(text):
    print("\n ##NAO## : "+text)
    ts.conversion(text)

def printUser(text):
    if(text=="Invalid String"):
        nao("Navré je n'ai pas compris")
    else :
        print("\n ##You## : "+text)



class Navigator(object):
          def indirect(self,name):
                  # map to key TODO

                   #name ==self.getMethodName(text)
                method=getattr(self,name,lambda :'Invalid')
                return method()
          def bye(self):
                   ts.conversion("Bye bye ! A bientot !")
                   sys.exit()                   
          def greetings(self):
                    #TODO
                    return 'greetings'

          def menu(self):
                    nao(MENU_TEXTS)
                    user_response = st.conversion()
                    printUser(user_response)
                    self.indirect(user_response)

          def erasmus(self):
                    self.doMenu("erasmus")
          def orientation(self):
                    self.doMenu("orientation")
          def pratique(self):
                    self.doMenu("pratique")
                             
          def doMenu(self,choice):
                    theme = selectionTheme(choice)
                    self.loadTheme(theme)
                    
          def loadTheme(self,theme):
                    global sent_tokens
                    nao(theme)
                    sent_tokens = get_tokens()
                    global themeLoaded
                    themeLoaded = 1     
          def response(self,user_response):
                    nao(response(user_response))
                    sent_tokens.remove(user_response)
"""

         #def getMethodName(text)
            liste=[] orientation erasmus menu ..
            if(text in list)
            return text
            else enfonction du text je retourne soit bye soit greetings"""