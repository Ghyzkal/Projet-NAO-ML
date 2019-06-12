#Meet NAO: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
import spacy
import fr_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer #scikit
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from speech import SpeechToText as st
warnings.filterwarnings('ignore')

#Lecture de la source de données
with open('data-MIAGE.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
    #C:\Users\Rayan\AppData\Local\Programs\Python\Python36\Lib\site-packages\spacy\lang\fr

#Tokenisation
nlp = spacy.load('fr_core_news_md') 
doc = nlp(raw) # Passage du texte par le pipeline
word_tokens = [w for w in doc] # converts to list of words
sent_tokens = [s.text for s in doc.sents] #converts to list of sentences 
stopWords = spacy.lang.fr.stop_words.STOP_WORDS

# Preprocessing
def LemTokens(tokens):
    return [(token.lemma_) for token in tokens]
remove_punct = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    texte_sans_ponct = text.lower().translate(remove_punct)
    return LemTokens(nlp(texte_sans_ponct))

# Mots d'accueil
GREETING_INPUTS = ("hello", "salut", "coucou", "bonjour", "ca va ?","hey",)
GREETING_RESPONSES = ["salut", "bonjour", "quoi de neuf docteur ?", "mes hommages", "hello"]

def greeting(sentence):
    """Si l'utilisateur envoie une salutation, répondre par une salutation"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Générer une réponse
def response(user_response):
    NAO_response=''
    sent_tokens.append(user_response)
    #On utilise ici la similrité cosinus pour comparer le document source à la requete de l'utilisateur
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopWords)
    tfidf = vectorizer.fit_transform(sent_tokens)
    #tfidf[-1] est la réponse de l'utilisateur et tfidf notre base de référence
    #on compare donc les deux
    vals = cosine_similarity(tfidf[-1], tfidf)
    #on trie le tableau en sens décroissant
    idx=vals.argsort(kind = 'quicksort')[0][-2]
    #on transfère toutes les valeurs dans un array 1 dimension qu'on va ensuite trier
    flat = vals.flatten()
    flat.sort()
    #on ne garde que les 2 dernières valeurs : 
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        NAO_response=NAO_response+"Désolé ! Je n'ai pas compris."
        return NAO_response
    else:
        NAO_response = NAO_response+sent_tokens[idx]
        return NAO_response

flag=True
print("NAO: Mon nom est NAO. Je répondrai à vos questions sur les ChatBots. Pour quitter, tapez Bye!")
while(flag==True):
    user_response = st.conversion()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='merci' or user_response=='merci beaucoup' ):
            flag=False
            print("NAO: Avec plaisir...")
        else:
            if(greeting(user_response)!=None):
                print("NAO: "+greeting(user_response))
            else:
                print("NAO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("NAO: Bye bye ! A bientot !")  