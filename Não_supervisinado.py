import pandas as pd 
import plotly.express as px #gráficos.
import plotly.graph_objects as go #gráficos.
import numpy as np 
from sklearn.preprocessing import StandardScaler #Para padronizar os dados. Para deixar os dados na mesma escala. 

df = pd.read_csv('D:/Estudos Python/bancos de dados/credit_card_clients.csv', header = 1)

#%% k-means

from sklearn.cluster import KMeans

#somando todas as dividas. 
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

#Convertendo em array.
df_card = df.iloc[:, [1,25]].values

#Padronizando os dados.
scaler_cartao = StandardScaler()
df_card = scaler_cartao.fit_transform(df_card) 

# criando os clusters para fazer tipo o gridsearch.
wcss = []

for i in range(1,11):
    kmeans_cartao = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao.fit(df_card)
    wcss.append(kmeans_cartao.inertia_)
    
#vizualizando os clusters. Obs: o gráfico só plota no collab.
grafico = px.line(x = range(1,11), y = wcss)
grafico.show()

#prevendo os agrupamentos.
kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(df_card)

#gráfico para mostrar os agrupamentos. Obs: o gráfico só plota no collab.
grafico = px.scatter(x = df_card[:,0], y = df_card[:,1], color=rotulos)
grafico.show()

# unindo o dataframe original com a classificação de agrupamento.
df_clientes = np.column_stack((df, rotulos))
df_clientes = df_clientes[df_clientes[:,26].argsort()] #agrupando pela ordem da ultima coluna.

# selecionando apenas alguns atributos para teste.

atr_df = df.iloc[:,[1,2,3,4,5,25]].values

scaler_cartao = StandardScaler()
atr_df = scaler_cartao.fit_transform(atr_df) 

wcss = []

for i in range(1,11):
    kmeans_cartao = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao.fit(atr_df)
    wcss.append(kmeans_cartao.inertia_)
    
grafico = px.line(x = range(1,11), y = wcss)
grafico.show()

kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(atr_df)

grafico = px.scatter(x = atr_df[:,0], y = atr_df[:,1], color=rotulos)
grafico.show()

#%% agrupamento hierárquico.

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('D:/Estudos Python/bancos de dados/credit_card_clients.csv', header = 1)

#somando todas as dividas. 
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

#Convertendo em array.
df_card = df.iloc[:, [1,25]].values

#Padronizando os dados.
scaler_cartao = StandardScaler()
df_card = scaler_cartao.fit_transform(df_card)

# gráfico para analizar o número de clusters.
dendrograma = dendrogram(linkage(df_card, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância');

##prevendo os agrupamentos.

hc_df = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'ward')
rotulos = hc_df.fit_predict(df_card)

#gráfico para mostrar os agrupamentos. Obs: o gráfico só plota no collab.
grafico = px.scatter(x = df_card[:,0], y = df_card[:,1], color=rotulos)
grafico.show()

#%% DBSCAN.

from sklearn.cluster import DBSCAN

# reaproveitando a banco já treinado.
df_card

# criando os grupos.
dbscan_card = DBSCAN(eps = 0.37, min_samples=5)
rotulos = dbscan_card.fit_predict(df_card)

# conferinco a quantidade de grupos com numpy para ver se o eps deu certo.
np.unique(rotulos, return_counts=True)

#gráfico para mostrar os agrupamentos. Obs: o gráfico só plota no collab.
grafico = px.scatter(x = df_card[:,0], y = df_card[:,1], color=rotulos)
grafico.show()

#%% comparando os três algoritimos. K-means x Hierárquico x DBSCAN.

from sklearn import datasets

# gerando dados aleatorios para um dataset. x = dados, y = classificação.
x_random, y_random = datasets.make_moons(n_samples=1500, noise = 0.09)

#analizando a disperção de x. obs: gráfico só plota no colab.
grafico = px.scatter(x = x_random[:,0], y = x_random[:,1])
grafico.show()

#agrupamento de hierárquico.
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'ward')
rotulos = hc.fit_predict(x_random)

grafico = px.scatter(x = x_random[:,0], y = x_random[:,1], color=rotulos)
grafico.show()

#DBSCAN.
dbscan = DBSCAN(eps = 0.1)
rotulos = dbscan.fit_predict(x_random)

grafico = px.scatter(x = x_random[:,0], y = x_random[:,1], color=rotulos)
grafico.show()

