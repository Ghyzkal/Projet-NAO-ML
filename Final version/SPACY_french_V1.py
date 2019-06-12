#Meet NAO: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import warnings
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
MENU_INPUTS = ("orientation", "pratique", "erasmus")
MENU_RESPONSES = ["data-orientation.txt", "data-pratiques.txt", "data-erasmus.txt"]

monFichierDeDonnee = ""
nlp = spacy.load('fr_core_news_md') 
sent_tokens = ""
tmp = 0
stopWords = spacy.lang.fr.stop_words.STOP_WORDS

"""
# Preprocessing
def LemTokens(tokens):
    return [(token.lemma_) for token in tokens]

remove_punct = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    texte_sans_ponct = text.lower().translate(remove_punct)
    return LemTokens(nlp(texte_sans_ponct))
"""
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
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def remerciement(sentence):
    """Si l'utilisateur envoie une salutation, répondre par une salutation"""
    for word in sentence.split():
        if word.lower() in THX_INPUTS:
            return random.choice(THX_RESPONSES)

def selectionTheme(sentence):
    """Si l'utilisateur envoie une selection de theme, met à jour le nom du document source pour qu'il pointe vers ce thème"""
    global monFichierDeDonnee
    for word in sentence.split():
        if word.lower() in MENU_INPUTS: 
            if(word.lower() == MENU_INPUTS[0]):
                monFichierDeDonnee = MENU_RESPONSES[0]
                return "Vous pouvez poser des questions sur les sujets suivant : la réorientation, les concours, ...\n"
            else:
                if(word.lower() == MENU_INPUTS[1]):
                    monFichierDeDonnee = MENU_RESPONSES[1]
                    return print("Vous pouvez poser des questions sur les sujets suivant : "+"carte étudiante \t\t soutien \t\t ENT "+"\n fonctionnement \t\t notes et cours \t\t stage"+"\n association \t\t services proposés par l'université\n")
                else:
                    if(word.lower() == MENU_INPUTS[2]):
                        monFichierDeDonnee = MENU_RESPONSES[2]
                        return "Découvrez-en plus sur le programme Erasmus : "+"à qui s'adresser, aides financières, ...\n"
             
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


def recogniseTransverse(user_response):
    if (greeting(user_response)):
        return greeting(user_response)
    elif (remerciement(user_response)):
        return remerciement(user_response)
    elif (user_response=='menu'):
        return menu(user_response)

    
def quitter():
    print("NAO: Bye bye ! A bientot !")
    return false


def menu(user_response):
    txtMenu = ("\nQuelle thématique vous intéresse ? "+"\n• Orientation : dites Orientation afin d'en savoir plus sur la réorientation, les concours, ..."+"\n• Questions Pratiques : Dites Pratique pour en savoir plus sur la carte étudiante, l'ENT, le fonctionnement des notes et des cours, ..."+"\n• Erasmus : dites Erasmus afin d'en savoir plus sur le programme Erasmus")
    print(txtMenu)
    user_response = input()
    user_response=user_response.lower()
    global sent_tokens
    theme = selectionTheme(user_response)
    if(theme!=0):
        print("\nNAO: "+ theme)
        sent_tokens = get_tokens()
        global tmp
        tmp = 1
        return "\n Chargement terminé !"

  
#Lecture de la source de données      
def get_tokens():
    with open(monFichierDeDonnee, 'r', encoding='utf8', errors ='ignore') as maBDD:
        raw = maBDD.read().lower()
    #remove the punctuation using the character deletion step of translate
    #Tokenisation
    global sent_tokens
    """global nlp
    nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser") # updated
    doc = nlp(raw) # Passage du texte par le pipeline
    word_tokens = [w for w in doc] # converts to list of words
    sent_tokens = [s.string.strip() for s in doc.sents] #converts to list of sentences 
    """
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    return sent_tokens


#Main
def main ():
    flag=True
    print("\nNAO: Mon nom est NAO. Je répondrai à vos questions sur la MIAGE de Nanterre Université. Pour quitter, tapez Bye")
    print("\nNAO: Pour changer de thème, tapez Menu")
    global tmp
    global sent_tokens
    tmp=0
    while(flag==True):
        user_response = input()
        user_response=user_response.lower()
        if (user_response=='bye'):
            flag=False
        elif(recogniseTransverse(user_response)!=None):
            print("\nNAO: Vous pouvez entrer votre demande")
        elif(user_response != None and tmp==1):
            print("\nNAO: ",end="")
            print(response(user_response))
            sent_tokens.remove(user_response)
       