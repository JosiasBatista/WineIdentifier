#coding: utf-8

import pandas as pd
from sklearn import model_selection
from sklearn import ensemble

test = model_selection.train_test_split
modelo = ensemble.ExtraTreesClassifier(n_estimators=100)

arquivo = pd.read_csv('../../../Downloads/wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#Separando as variaveis entre alvo e preditoras:
alvo = arquivo['style']
preditores = arquivo.drop('style', axis=1)

#Criando os conjuntos de dados de treino e teste:
preditores_treino, preditores_teste, alvo_treino, alvo_teste = test(preditores, alvo, test_size = 0.3)

#Criando o modelo
modelo.fit(preditores_treino, alvo_treino)

#Imprimindo os resultados:
resultado = modelo.score(preditores_teste, alvo_teste)
print("Acurácia: {}".format(resultado*100))

previsoes = modelo.predict(preditores_teste[400:403])
print(previsoes)
print('---')
print(alvo_teste[400:403])
