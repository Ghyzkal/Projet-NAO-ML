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

global sent_tokens


# Preprocessing
def LemTokens(tokens):
    return [(token.lemma_) for token in tokens]

remove_punct = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    texte_sans_ponct = text.lower().translate(remove_punct)
    return LemTokens(nlp(texte_sans_ponct))

# Mots d'accueil
GREETING_INPUTS = ("hello", "salut", "coucou", "bonjour", "ca va ?","hey")
GREETING_RESPONSES = ["salut", "bonjour", "quoi de neuf docteur ?", "mes hommages", "hello"]

def greeting(sentence):
    """Si l'utilisateur envoie une salutation, répondre par une salutation"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Selection des thèmes (documents)
MENU_INPUTS = ("orientation", "pratique", "erasmus")
MENU_RESPONSES = ["data-orientation.txt", "data-pratiques.txt", "data-erasmus.txt"]

def selectionTheme(sentence):
    """Si l'utilisateur envoie une selection de theme, met à jour le nom du document source pour qu'il pointe vers ce thème"""
    for word in sentence.split():
        if word.lower() in MENU_INPUTS:
            global monFichierDeDonnee 
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
        else:
            return 0
             
#Lecture de la source de données
#with open('data-erasmus.txt','r', encoding='utf8', errors ='ignore') as maBDD:
#    raw = maBDD.read().lower()
    
#Lecture de la source de données      
def get_tokens():
    with open(monFichierDeDonnee, 'r', encoding='utf8', errors ='ignore') as maBDD:
        raw = maBDD.read().lower()
    #remove the punctuation using the character deletion step of translate
    #Tokenisation
    global nlp
    nlp = spacy.load('fr_core_news_md') 
    nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser") # updated
    doc = nlp(raw) # Passage du texte par le pipeline
    word_tokens = [w for w in doc] # converts to list of words
    sent_tokens_local = [s.string.strip() for s in doc.sents] #converts to list of sentences 

    return sent_tokens_local


# Générer une réponse
def response(user_response):
    NAO_response=''
    sent_tokens.append(user_response)
    stopWords = spacy.lang.fr.stop_words.STOP_WORDS
    
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
    idx=vals.argsort(kind = 'quicksort')[0][-2]
   
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

#Main
flag=True
print("\nNAO: Mon nom est NAO. Je répondrai à vos questions sur la MIAGE de Nanterre Université. Pour quitter, tapez Bye")
print("\nQuelle thématique vous intéresse ? "+
                "\n• Orientation : dites Orientation afin d'en savoir plus sur la réorientation, les concours, ..."+
                "\n• Questions Pratiques : Dites Pratique pour en savoir plus sur la carte étudiante, l'ENT, le fonctionnement des notes et des cours, ..."+
                "\n• Erasmus : dites Erasmus afin d'en savoir plus sur le programme Erasmus")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='merci' or user_response=='merci beaucoup' ):
            print("NAO: Avec plaisir...")
        else:
            if(user_response=='menu'):
                print("\nNAO: Quelle thématique vous intéresse ? "+
                "\n• Orientation : dites Orientation afin d'en savoir plus sur la réorientation, les concours, ..."+
                "\n• Questions Pratiques : Dites Pratique pour en savoir plus sur la carte étudiante, l'ENT, le fonctionnement des notes et des cours, ..."+
                "\n• Erasmus : dites Erasmus afin d'en savoir plus sur le programme Erasmus")
            else:
                if(greeting(user_response)!=None):
                    print("\nNAO: "+greeting(user_response))
                else:
                    if(selectionTheme(user_response)!=0):
                        print("\nNAO: "+ selectionTheme(user_response))
                        print("Merci d'attendre la fin du chargement ...")
                        sent_tokens = get_tokens()
                        menu=True
                        while (menu == True):
                            print("Veuillez entrer votre demande")
                            user_response = input()
                            user_response=user_response.lower()
                            print("NAO: ",end="")
                            print(response(user_response))
                            sent_tokens.remove(user_response)
                            print("NAO: Pour changer de thème, tapez Menu")
                            if(user_response=="menu"):
                                menu=False
                            if(user_response=="bye"):
                                menu=False
                                flag=False

    else:
        flag=False
        print("NAO: Bye bye ! A bientot !")  