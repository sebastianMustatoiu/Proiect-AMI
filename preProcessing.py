import nltk
import unidecode
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def inlocuire_diacritice(text):
    return unidecode.unidecode(text)

def eliminare_punctuatie(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def procesare_versuri_fara_stopwords(versuri):
    cuvinte_fara_stopwords_lista = []
    for lyrics in versuri:
        versuri_fara_diacritice = inlocuire_diacritice(lyrics)
        versuri_fara_punctuatie = eliminare_punctuatie(versuri_fara_diacritice)
        cuvinte = word_tokenize(versuri_fara_punctuatie.lower())

        stop_words = set(nltk.corpus.stopwords.words('romanian'))
        cuvinte_fara_stopwords = [cuvant for cuvant in cuvinte if cuvant not in stop_words and len(cuvant) > 2]

        cuvinte_fara_stopwords_lista.append(cuvinte_fara_stopwords)

    return cuvinte_fara_stopwords_lista

def procesare_versuri_cu_stopwords(versuri):
    cuvinte_lista = []
    for lyrics in versuri:
        versuri_fara_diacritice = inlocuire_diacritice(lyrics)
        versuri_fara_punctuatie = eliminare_punctuatie(versuri_fara_diacritice)
        cuvinte = word_tokenize(versuri_fara_punctuatie.lower())
        cuvinte_lista.append(cuvinte)
    return cuvinte_lista
