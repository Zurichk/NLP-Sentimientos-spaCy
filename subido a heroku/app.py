from flask import Flask, render_template, request
import spacy
import nltk
#Analisis de sentimientos
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer #Ingles
sid = SentimentIntensityAnalyzer()
sentiment_es = SentimentIntensityAnalyzer()

#https://github.com/pysentimiento/pysentimiento/
#from pysentimiento import analyzer  #Español
#sentiment_es = analyzer.SentimentAnalyzer(lang='es')
#emotion_es = analyzer.EmotionAnalyzer(lang='es')
#from classifier import SentimentClassifier #Español
#clf = SentimentClassifier()

#Detectar idioma
from langdetect import detect

app = Flask(__name__)

@app.route("/")
def indice():
    return render_template('index.html')
                        
@app.route('/process', methods = ['POST'])
def procesa_texto():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        idioma = detect(rawtext)
        sentimientos = []
        if idioma=='es':
            try:
               nlp = spacy.load("es_core_news_md")
            except: # Si el idioma no existe, descargalo
               spacy.cli.download("es_core_news_md")
               nlp = spacy.load("es_core_news_md")
            print("Idioma Español: cargamos es_core_news_md")
            #Analisis de sentimiento
            sentimientos = sentiment_es.predict(rawtext)
            #print(sentimientos)
            
        elif idioma=='en':
            try:
               nlp = spacy.load("en_core_web_md")
            except: # Si el idioma no existe, descargalo
               spacy.cli.download("en_core_web_md")
               nlp = spacy.load("en_core_web_md")
            print("Idioma Ingles: cargamos en_core_web_md")
            #Analisis de sentimiento
            sentimientos = sid.polarity_scores(rawtext)
            #sentimientos2 = sentimientos["compound"]
            #print(sentimientos)
            
        else:
            nlp = spacy.load("es_core_news_md") 
            print("No reconozco el idioma,se queda cargado Español: es_core_news_sm, por defecto ")
        
        opcion = request.form['taskoption']
        doc = nlp(rawtext)
        results = [(entidad.label_, entidad.text) for entidad in doc.ents]

        entidad_ORG = []
        entidad_PER = []
        entidad_LOC = []
        entidad_TIME = []   
        entidad_LANGUAGE = []
        entidad_MISC = []
        
        for entidad in doc.ents:
            if entidad.label_ == 'ORG':
                entidad_ORG.append((entidad.label_, entidad.text))
            elif entidad.label_ == 'PER' or entidad.label_ == 'PERSON':
                entidad_PER.append((entidad.label_, entidad.text))
            elif entidad.label_ == 'LOC' or entidad.label_ == 'GPE':
                entidad_LOC.append((entidad.label_, entidad.text))
            elif entidad.label_ == 'TIME':
                entidad_TIME.append((entidad.label_, entidad.text))           
            elif entidad.label_ == 'LANGUAGE':
                entidad_LANGUAGE.append((entidad.label_, entidad.text))
            elif entidad.label_ == 'MISC':
                entidad_MISC.append((entidad.label_, entidad.text))
                        
        if opcion == 'organization':
            results = entidad_ORG
            num_of_results = len(results)
        elif opcion == 'person':
            results = entidad_PER
            num_of_results = len(results)
        elif opcion == 'location':
            results = entidad_LOC
            num_of_results = len(results)
        elif opcion == 'time':
            results = entidad_TIME
            num_of_results = len(results)
        elif opcion == 'language':
            results = entidad_LANGUAGE
            num_of_results = len(results)
        elif opcion == 'misc':
            results = entidad_MISC
            num_of_results = len(results)        
        else:
            results = [ (entity.label_, entity.text) for entity in doc.ents]
            num_of_results = len(results)
                  
    return render_template('index.html', num_of_results = num_of_results, results = results, sentimientos = sentimientos)#, snt = snt)

if __name__ == '__main__':
    app.run(debug = True)

"""Texto de prueba:
Adrián García estuvo en Madrid y visitó la tienda Apple, el pasado jueves 02 de Marzo de 2022, y fue atendido por Charles, que hablaba en ingles, durante 2 horas, y se gastó unos 600 euros en un Iphone 10. 
Por la tarde, se fue a pasear por el parque del Retiro y estuvo con Miguel Pérez durante 3 horas. Los 2 estaban muy felices.
"""
"""
Test text:
Adrián García was in Madrid and visited the Apple store, last Thursday, March 02, 2022, and was attended by Charles, who spoke in english, for 2 hours, and spent about 600 euros on an Iphone 10. 
In the afternoon, He went for a walk in the Retiro park and was with Miguel Pérez for 3 hours.
"""
