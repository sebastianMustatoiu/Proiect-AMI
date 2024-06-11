import nltk
from collections import Counter
import spacy
from gensim import corpora
from gensim.models import LdaModel
from preProcessing import procesare_versuri_fara_stopwords

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("ro_core_news_sm")

def analiza_versuri(versuri):
    cuvinte_fara_stopwords_lista = []
    frecventa_lista = []
    entitati_lista = []
    numar_cuvinte_lista = []
    lungime_medie_cuvinte_lista = []
    counter_cuvinte_lista = []

    for lyrics in versuri:
        cuvinte_fara_stopwords = procesare_versuri_fara_stopwords([lyrics])[0]

        cuvinte_fara_stopwords_lista.append(cuvinte_fara_stopwords)

        frecventa = nltk.FreqDist(cuvinte_fara_stopwords)
        frecventa_lista.append(frecventa)

        doc = nlp(" ".join(cuvinte_fara_stopwords))
        entitati = [(ent.text, ent.label_) for ent in doc.ents]
        entitati_lista.append(entitati)

        numar_cuvinte = len(cuvinte_fara_stopwords)
        lungime_medie_cuvant = int(sum(len(cuvant) for cuvant in cuvinte_fara_stopwords) / numar_cuvinte)
        numar_cuvinte_lista.append(numar_cuvinte)
        lungime_medie_cuvinte_lista.append(lungime_medie_cuvant)

        counter_cuvinte = Counter(cuvinte_fara_stopwords)
        counter_cuvinte_lista.append(counter_cuvinte)

    return {
        "cuvinte_fara_stopwords_lista": cuvinte_fara_stopwords_lista,
        "frecventa_lista": frecventa_lista,
        "entitati_lista": entitati_lista,
        "numar_cuvinte_lista": numar_cuvinte_lista,
        "lungime_medie_cuvinte_lista": lungime_medie_cuvinte_lista,
        "counter_cuvinte_lista": counter_cuvinte_lista
    }

def modelare_lda(cuvinte_fara_stopwords_lista, num_topics=2):
    dictionary = corpora.Dictionary(cuvinte_fara_stopwords_lista)
    corpus = [dictionary.doc2bow(text) for text in cuvinte_fara_stopwords_lista]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topics = lda_model.print_topics(num_words=6)
    for topic in topics:
        print(topic)
    return lda_model, dictionary, corpus
