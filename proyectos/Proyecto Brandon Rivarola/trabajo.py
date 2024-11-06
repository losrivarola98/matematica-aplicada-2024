import pandas #Manejador de datasets
import nltk #Biblioteca de analizadores de sentimientos
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import sentiwordnet as swn #El lexicón a usar
from nltk.tokenize import word_tokenize #Para dividir oraciones en palabras separadas
from nltk.corpus import wordnet #Lexicón de donde deriva SentiWordNet
import skfuzzy #Funciones para lógica difusa
import numpy #Trabaja con arrays
import time #Calculo de tiempo de ejecución

tweets = pandas.read_csv('test_data.csv') #Se almacena el dataset en tweets

def preprocesar(texto):
    texto = texto.lower() #Todo a minúsculas
    palabras = texto.split() 
    palabras = [palabra for palabra in palabras if not (palabra.startswith('http') or palabra.startswith('www'))] 
    texto = ' '.join(palabras) #Se borran todos las direcciones URL
    palabras = texto.split() 
    palabras = [palabra for palabra in palabras if not palabra.startswith('@') and not palabra.startswith('#')] 
    texto = ' '.join(palabras) #Se borran los # y @
    palabras = texto.split() 
    return texto

#Wordnet tiene constantes que sirven para determinar el tipo de palabra que está analizando (adjetivo, verbo, sustantivo o adverbio)
#Esta funcion convierte las etiquetas de pos_tag a etiquetas de wordnet para usarlas en el método senti_synsets
#senti_synsets sólo acepta los cuatro tipo de palabras mencionados arriba, por lo que el resto se descarta
def obtenerEtiqueta(etiqueta):
    if etiqueta.startswith('J'):
        return wordnet.ADJ
    elif etiqueta.startswith('V'):
        return wordnet.VERB
    elif etiqueta.startswith('N'):
        return wordnet.NOUN
    elif etiqueta.startswith('R'):
        return wordnet.ADV
    else:
        return None

def analizarSentimiento(tweet):
    palabras = word_tokenize(tweet['sentence']) #Divide en palabras separadas
    palabrasEtiquetadas = nltk.pos_tag(palabras) #A cada palabra se le etiqueta según su categoría
    positivo = 0
    negativo = 0
    for palabra, etiqueta in palabrasEtiquetadas:
        wdEtiqueta = obtenerEtiqueta(etiqueta) #Ver línea 31
        if wdEtiqueta is None: 
            continue #Si no es adjetivo, vebo, sustantivo ni adverbio, pasa a la siguiente palabra
        synsets = list(swn.senti_synsets(palabra,wdEtiqueta)) #Se obtiene una puntuacion de sentimientos de acuerdo con la palabra
        if synsets:
            if synsets[0].pos_score() > synsets[0].neg_score(): #Se calcula el puntaje segun sea mayor el positivo o el negativo
                positivo += synsets[0].pos_score()
            else:
                negativo += synsets[0].neg_score()
    return pandas.Series([positivo,negativo])

def calcularPuntaje(tweet):
    inicio = time.time()
    fuzzificado = fuzzificar(tweet['positivo'],tweet['negativo'])
    defuzzificado = defuzzificar(fuzzificado)
    fin = time.time()
    return pandas.Series([defuzzificado,fin-inicio])

def fuzzificar(pos,neg):
    #Encontramos los puntos límites de los números difusos triangulares
    pMin = tweets['positivo'].min()
    pMax = tweets['positivo'].max()
    pMed = (pMin+pMax)/2
    nMin = tweets['negativo'].min()
    nMax = tweets['negativo'].max()
    nMed = (nMin+nMax)/2
    #Rangos para las entradas
    auxRangoPos = tweets['positivo'].to_numpy()
    auxRangoNeg = tweets['negativo'].to_numpy()
    rangoPos = auxRangoPos.copy() #Se realiza una copia independiente para que el ordenamiento no afecte al dataset, sino sólo al rango
    rangoNeg = auxRangoNeg.copy()
    rangoPos.sort() #El rango debe estar ordenado
    rangoNeg.sort()
    rangoOp = numpy.arange(0,10.1,0.1)
    #Se crean las funciones de membresia
    pos_low = skfuzzy.trimf(rangoPos,[pMin,pMin,pMed])
    pos_med = skfuzzy.trimf(rangoPos,[pMin,pMed,pMax])
    pos_high = skfuzzy.trimf(rangoPos,[pMed,pMax,pMax])
    neg_low = skfuzzy.trimf(rangoNeg,[nMin,nMin,nMed])
    neg_med = skfuzzy.trimf(rangoNeg,[nMin,nMed,nMax])
    neg_high = skfuzzy.trimf(rangoNeg,[nMed,nMax,nMax])
    op_neg = skfuzzy.trimf(rangoOp,[0,0,5])
    op_med = skfuzzy.trimf(rangoOp,[0,5,10])
    op_pos = skfuzzy.trimf(rangoOp,[5,10,10])
    #Se obtienen los grados de membresía
    gPos_low = skfuzzy.interp_membership(rangoPos,pos_low,pos)
    gPos_med = skfuzzy.interp_membership(rangoPos,pos_med,pos)
    gPos_high = skfuzzy.interp_membership(rangoPos,pos_high,pos)
    gNeg_low = skfuzzy.interp_membership(rangoNeg,neg_low,neg)
    gNeg_med = skfuzzy.interp_membership(rangoNeg,neg_med,neg)
    gNeg_high = skfuzzy.interp_membership(rangoNeg,neg_high,neg)
    #Se crean las reglas
    regla1 = numpy.fmin(gPos_low,gNeg_low)
    regla2 = numpy.fmin(gPos_med,gNeg_low)
    regla3 = numpy.fmin(gPos_high,gNeg_low)
    regla4 = numpy.fmin(gPos_low,gNeg_med)
    regla5 = numpy.fmin(gPos_med,gNeg_med)
    regla6 = numpy.fmin(gPos_high,gNeg_med)
    regla7 = numpy.fmin(gPos_low,gNeg_high)
    regla8 = numpy.fmin(gPos_med,gNeg_high)
    regla9 = numpy.fmin(gPos_high,gNeg_high)
    #Se hacen las agregaciones
    intensidadNeg = numpy.fmax(regla4,numpy.fmax(regla7,regla8))
    intensidadMed = numpy.fmax(regla1,numpy.fmax(regla5,regla9))
    intensidadPos = numpy.fmax(regla2,numpy.fmax(regla3,regla6))
    activacionNeg = numpy.fmin(intensidadNeg,op_neg)
    activacionMed = numpy.fmin(intensidadMed,op_med)
    activacionPos = numpy.fmin(intensidadPos,op_pos)
    agregado = numpy.fmax(activacionNeg,numpy.fmax(activacionMed,activacionPos))
    return agregado

def defuzzificar(fuzzificado): #Utiliza el método del centroide
    rangoOp = numpy.arange(0,10.1,0.1)
    numerador = numpy.sum(rangoOp * fuzzificado)
    denominador = numpy.sum(fuzzificado)
    if denominador != 0:
        return numerador/denominador
    else:
        return 0

tweets['sentence'] = tweets['sentence'].apply(preprocesar) #Preprocesa el dataset
tweets[['positivo','negativo']] = tweets.apply(analizarSentimiento,axis=1) #Analiza el sentimiento de cada tweet y crea los nuevos puntajes
tweets[['puntaje','tiempo']] = tweets.apply(calcularPuntaje,axis=1) #Calcula el puntaje (fuzzifica y defuzzifica)
pandas.set_option('display.float_format','{:.7f}'.format) #Muestra más precisión de decimales en pantalla
print(tweets)
tweets.to_csv('nuevo_dataset.csv',index=False) #Guarda el nuevo dataset
print("Total tweets positivos:",tweets[tweets['puntaje'] >= 6.7]['puntaje'].count())
print("Total tweets neutrales:",tweets[(tweets['puntaje'] < 6.7) & (tweets['puntaje'] >= 3.3)]['puntaje'].count())
print("Total tweets negativos:",tweets[tweets['puntaje'] < 3.3]['puntaje'].count())
print("Tiempo de ejecución promedio:",tweets['tiempo'].mean())