import random
import joblib
from preProcessing import procesare_versuri_cu_stopwords, procesare_versuri_fara_stopwords
from sklearn.metrics import classification_report

def train_ngram_model(cuvinte_lista, n):
    ngrams = {}
    all_words = [word for sublist in cuvinte_lista for word in sublist]
    for i in range(len(all_words) - n):
        gram = tuple(all_words[i:i + n])
        next_word = all_words[i + n]
        if gram not in ngrams:
            ngrams[gram] = []
        ngrams[gram].append(next_word)
    return ngrams


def generate_lyrics(ngram_model, n, num_words=100):
    current_gram = random.choice(list(ngram_model.keys()))
    result = list(current_gram)
    all_words = set([word for sublist in ngram_model.values() for word in sublist])  # Set cu toate cuvintele valide

    for _ in range(num_words - n):
        if current_gram in ngram_model:
            possible_next_words = ngram_model[current_gram]
            next_word = random.choice(possible_next_words)

            # Verificare pentru a evita repetarea cuvintelor consecutive și cuvintele care nu sunt valide
            attempts = 0
            max_attempts = 10  # Limităm numărul de încercări pentru a evita buclele infinite
            while (next_word == result[-1] or next_word not in all_words) and attempts < max_attempts:
                next_word = random.choice(possible_next_words)
                attempts += 1

            result.append(next_word)
            current_gram = tuple(result[-n:])
        else:
            break
    return ' '.join(result)

model = joblib.load('genre_classifier_model.pkl')

versuri = [
    """Oo, oo, imi plac prea mult ochii tai
Oo, oo, ce sa ma mai fac cu ei
Oo, oo, cred ca bag la amanet
Sa merg sa iti iau bilet
Te duc undeva secret

Pai naa, pai naa
Nu te lasa mama ta
Sa umbli cu Babasha
Ca te ia de langa ea

Pai na, pai na, pai na
Are dulce vrajeala
Il ajuta si pielea
Poate-ti fura inima

Oo, oo, la tine-am abonament
Oo, oo, sa te iubesc permanent
Oo, oo, dragoste cu gust de Cola
Mi-ai dat peste cap busola
Sa te mananc ca pe Ola

Ale, ale-le-le, tu esti fix pe placul meu
Si oricat mi-ar fi de greu, te vreau mereu, gen
Ale, aleleu, se-anunta mare stabor
Ca in neam de Spoitori sa te-nsori nu e usor

Pai naa, pai naa
Nu te lasa mama ta
Sa umbli cu Babasha
Ca te ia de langa ea

Pai na, pai na, pai na
Are dulce vrajeala
Il ajuta si pielea
Poate-ti fura inima

Pai na, pai na, nu te lasa mama ta
Pai na, na, na, na, na, ca te ia de langa ea
Si iti fura inima""",
"""
Ma stie toata tara
Ma plac para, para
Uite cum urc scara
Din Craiova-n  Timisoara

Azi sunt jos, maine sunt sus
Asa e viata mea
Ba pe minus, ba pe plus
Ca-i usoara, ca-i grea

Niciodata nu m-am plans
Si mai mereu mi-am spus
Tin capul sus
Tin capul sus

Cand nu aveam nimic
Eu am crezut in mine
Nimic nu m-a oprit
Sa ajung la inaltime

Am spart milioane odata
Am facut lumea sa taca
Am baut un Perignon
Am baut si apa plata

Ma stie toata tara
Ma plac para, para
Uite cum urc scara
Din Craiova-n  Timisoara

Azi sunt jos, maine sunt sus
Asa e viata mea
Ba pe minus, ba pe plus
Ca-i usoara, ca-i grea

Niciodata nu m-am plans
Si mai mereu mi-am spus
Tin capul sus
Tin capul sus

Ma stie toata tara
Ma plac para, para
Uite cum urc scara
Din Craiova-n  Timisoara

Azi sunt jos, maine sunt sus
Asa e viata mea
Ba pe minus, ba pe plus
Ca-i usoara, ca-i grea

Niciodata nu m-am plans
Si mai mereu mi-am spus
Tin capul sus
Tin capul sus

O zi buna, una rea
Una mai comsi-comca
Usoara sau grea
Cum o fi, e viata mea

Visele mele
Cum or fi ele
Poate se vor implini
Toate poate intr-o zi

Ma stie toata tara
Ma plac para, para
Uite cum urc scara
Din Rosiori-n  Timisoara

Azi sunt jos, maine sunt sus
Asa e viata mea
Ba pe minus, ba pe plus
Ca-i usoara, ca-i grea

Niciodata nu m-am plans
Si mai mereu mi-am spus
Tin capul sus
Tin capul sus

Azi sunt jos, maine sunt sus
Asa e viata mea
Ba pe minus, ba pe plus
Ca-i usoara, ca-i grea

Niciodata nu m-am plans
Si mai mereu mi-am spus
Tin capul sus
Tin capul sus""",
"""
Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă
Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă

Paranormal, facem panaramă, ce pana mea?
Stau la pândă, papagalii ăștia sunt paletă, nu-s păpușă, 's păpușari, dă-vă-n pula mea
Panseluțele astea plâng în pernă, nevoie de pansament cu Paraziții să se schimbe lumea
Paparazzi umblă după mine ca să poată afla cu cine se pupă Macarena
Ooh
Fetele bine pe bass, da
Fustă scurtă, balans
Mami, ce haz, ooh
Vrea cu mine acasă
Sunt Vrăjitoru' din Oz, Oz

Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă
Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă

E sezonu' de alergii și de small dick energy, ah
"Hocus-pocus preparatus, da' de ce nu faci copii?
Ai o vârstă, fată, știi? Fii mai feminină și
Te-mbraci aiurea rău, ești prea boschetară, zău
Doamna Delia, nu vă supărați
Știu că nu m-a-ntrebat nimeni, dar exagerați
Doamna Delia e delulu, doamna Delia e-n tendințe
Nu vă supărați, aș vrea să vă dau niște semințe"

Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă
Fac să sară hainele de pe tine, abracadabra
Abracadabra, fă, abracadabra
Fac să nu te mai uiți la altcineva decât la mine
Abracadabra, fă, abracadabra, fă""",
"""
De ce ești posomorât?
Când mă vezi te uiți urât
Îți bagi unghia-n gât
Din Urus am coborât
Aseară m-am omorât

N-am timp de prăjeală, vrăjeală
Ce-am io-n cap nu-i de la școală
N-ai dosar, dar ești penală
Te pup, îți urez o seară

Da, da, da, eu sunt ăla
Stii ca sunt sef pe zona
N-am timp, scump, imi dai tona
La noi a trecut ora

Va mananc ca pe Ola
Nu prea va merge rola
Vi s-a stricat consola
Stati jos si beti o Cola

Nu am somn, am filme în cap, sunt loco
Știi că nu fac lei, fac coco
Anul ăsta tot al meu, e soto
Buzunaru’ plin, la mine bani pronto
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco

De ce ma vorbiti intr-una
Nici n-ati parasit comuna
Nu mai stau, nu mai stau
La vrajeala nu mai stau
Sef si pe digital
Stau bine si pe capital

Da, da, da, eu sunt ăla
Stii ca sunt sef pe zona
N-am timp, scump, imi dai tona
La noi a trecut ora

Va mananc ca pe Ola
Nu prea va merge rola
Vi s-a stricat consola
Stati jos si beti o Cola

Nu am somn, am filme în cap, sunt loco
Știi că nu fac lei, fac coco
Anul ăsta tot al meu, e soto
Buzunaru’ plin, la mine bani pronto
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco
Coco
Tu faci lei, io fac coco

Tu faci lei, io fac coco
Tu faci lei, io fac coco
Tu faci lei, io fac coco
Tu faci lei, io fac coco
Tu faci lei, io fac coco
Tu faci lei, io fac coco
Tu faci lei, io fac coco""",
"""
Ce ar fi lumea
Fara iubire
Asa sunt eu fara tine

Nici toata marea
Nu ne desparte
Nici macar stelele toate

Soarele meu
Cand ma iubesti
Cel mai frumos stralucesti

Ce ar fi lumea
Fara iubire
Asa sunt eu fara tine

Ca…
Inima mea e locul tau
N-am cum sa schimb asta eu
Ploi si furtuni de vor urma
Nu-mi ating inima

Te privesc fara cuvant
Cel mai bogat om pe pamant
Ploi si furtuni vor fi mereu
Dar sunt eu

Clipa mea buna
Despre iubire
E cand esti tu langa mine

Buzele tale
Cu foc sarutare
Ma-ndragostesc si mai tare

Ochii-mi vorbesc
Nu vad nevoie
Sa spun ce mult te iubesc

Doar ce ar fi lumea
Fara iubire
Asa sunt eu fara tine

Ca…
Inima mea e locul tau
N-am cum sa schimb asta eu
Ploi si furtuni de vor urma
Nu-mi ating inima

Te privesc fara cuvant
Cel mai bogat om pe pamant
Ploi si furtuni vor fi mereu
Dar sunt eu

Ploi si furtuni vor fi mereu
Dar sunt eu""",

"""Când apare da peisaje, îi dă pe toți gata E bruneta aia bună, giga ciocolata Îi place ca viața, e ca haladită Digirididida, de mor toți, dar eu mi-am luat gagică. Că nici eu nu suflu, fraier, am foanele mele Le fur eu inimile, le fac zilele grele Sex după sex, dacă ai belea pe piele Lumea ro zice, ai danesanit, sai Nele. Ce sa fac, ro place viața Și alcoolul, ca substanța, Dar tot ro mențin prestanța Se scot vorbe de mine Maga – maga, fac ca rața. Și acum, da aplauzele dumneavoastră Tot Real Skitt, dar un pic diferit De dimineața ca până seara Basul, contrabasul de toamna până vara Îmi place viața, lumea sa vorbește Lumea sană bea cu mine, lumea rea sa dușsanește Banii nu sunt o problema, la fel ca femeile, Dar eu când vad bruneta asta, ro vin toate ideile Da cum aș pune-o, cum aș face-o, ca pe-o parte, ca pe alta Pleacă asta, vine alta, sex pervers, hai, gata, fata. Ce sa fac, ro place viața Și alcoolul, ca substanța, Dar tot ro mențin prestanța Se scot vorbe de mine Maga – maga, fac ca rața""",
"""Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca Țin da mine, dar tu tot sa auzi. Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca, Am ochii albaștri, rocai, curca.Am ochii negri plânca, de aia-s blânzi Rănile nu trebuie sa le mai ascunzi Cicatricile te fac mândru, te uiți la ele ca râzi Mi-am lăsat orgoliul deoparte Tu ca eu da locul nostru, da noapte Nu zi nimic, vreau doar fapte Vreau note daalte, nu șase Nu te vad, nu te aud Vocea mea răsună da camera ca-mi place mult N-am răbdare, vreau sa sa vad Nu-mi da din prima, trebuie sa aștept Mereu vreau mai mult, Dau tot, iusirea mea nu se împarte Că ca ea da trecut te-a lăsat departe Nu zi nimic, lasa-te pe spate Doar ochii vad dincolo de șoapte. Detașată, dar nu sunt rece Nu mai vrea sa plece singurul ca-nțelege, Detașată, dar nu sunt rece Nu mai vrea sa plece singurul ca-nțelege.Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca Țin da mine, dar tu tot sa auzi. Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca, Am ochii albaștri, rocai, curca.Și am capul alb, sa nu sa se sperie, când ro vede Ochii injectați ca cearcănele tot mai negre La dracu, sunt tare, nu-mi mai trece, nu se pierde Mă pierd da ochii ei ca nu mai am regrete Încerc sa fac cumva sa sa simt din nou Ce ne desparte – e golul tău Mi-e greu, nici ce vad, nici ce simt, tot ce aud E mult prea fals ca da ecou Holes in a pictures, privirea ta e cu ochii, tot mai faci tablou Doare tare, e tot mai pula, fac ce fac ca eu sunt din nou.Detașată, dar nu sunt rece Nu mai vrea sa plece singurul ca-nțelege, Detașată, dar nu sunt rece Nu mai vrea sa plece singurul ca-nțelege.Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca Țin da mine, dar tu tot sa auzi. Am ochii albaștri, rocai, curca Semnele doar tu poți sa mi le-ascunzi Am ochii albaștri, rocai, curca, Am ochii albaștri, rocai, curca, Am ochii albaștri, rocai, curca, Am ochii albaștri, rocai, curca."""
]

cuvinte_fara_stopwords_lista = procesare_versuri_cu_stopwords(versuri)
ngram_model = train_ngram_model(cuvinte_fara_stopwords_lista, 4)

num_generated_lyrics = 20
generated_lyrics_list = []

for _ in range(num_generated_lyrics):
    generated_lyrics = generate_lyrics(ngram_model, 4)
    generated_lyrics_list.append(generated_lyrics)
    print("Versuri generate:")
    print(generated_lyrics)

gen_versuri_procesate = procesare_versuri_fara_stopwords(generated_lyrics_list)
gen_versuri_test = [" ".join(versuri) for versuri in gen_versuri_procesate]
gen_genuri_pred = model.predict(gen_versuri_test)

print("Genurile versurilor generate:")
print(gen_genuri_pred)

# Generarea raportului de clasificare pentru versurile generate
gen_genuri_test = ["manea"] * (num_generated_lyrics // 2) + ["trap"] * (num_generated_lyrics - num_generated_lyrics // 2)
print("Raport de clasificare pentru identificarea genului muzical:")
print(classification_report(gen_genuri_test, gen_genuri_pred, zero_division=0))