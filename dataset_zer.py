# -*- coding: utf-8 -*-
"""fou 2 Versão 4 KFCM-W-2.ipynb - Dataset fou

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yebIyn13eUAjjf5JRgq-WTPI3BgL5T0k
"""

!pip install imgkit
!apt-get install -y wkhtmltopdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from google.colab import drive
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, RocCurveDisplay, det_curve, auc)
from decimal import Decimal, getcontext
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import imgkit
from IPython.display import Image

drive.mount('/content/drive')

#Gabi
data = pd.read_csv('/content/drive/MyDrive/Datasets/mfeat-fac', header=None,delim_whitespace=True)
label= pd.read_csv('/content/drive/MyDrive/Datasets/label.csv', header=None,delim_whitespace=True)

datan = pd.read_csv('/content/drive/MyDrive/Datasets/mfeat-fac', header=None,delim_whitespace=True)
labeln= pd.read_csv('/content/drive/MyDrive/Datasets/label.csv', header=None,delim_whitespace=True)

#Dataset dos números

X1=data.to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X1)

import numpy as np

class KFCM_K_W2:
    def __init__(self, data, c=10, m=1.6, max_iter=100, tol=1e-6):
        self.data = data
        self.c = c
        self.m = m
        self.n, self.p = data.shape
        self.U = np.zeros((self.c, self.n))
        self.g = np.zeros((self.c, self.p))
        self.s = np.ones((self.c, self.p))
        self.max_iter = max_iter
        self.tol = tol

#Kernel
    def gaussian_kernel(self, x, g, s):
          kernel = kernel = np.exp(-0.5 * (np.sum((x - g)**2 * s, axis=1)))
          return kernel

#Início daa Matrizes U e G
    def initialize(self):
        self.U = np.random.rand(self.c, self.n)
        self.U /= self.U.sum(axis=1)[:, np.newaxis]
        idx = np.random.choice(self.n, self.c, replace=False)
        self.g = self.data[idx]
        return self.U, self.g

#Eq.8
    def update_membership(self):
        epsilon = 1e-10
        for k in range(self.n):
            for i in range(self.c):
                kernel_values_i = self.gaussian_kernel(np.tile(self.data[k], (self.c, 1)), self.g[i], self.s[i])
                for h in range(self.c):
                  kernel_values_h = self.gaussian_kernel(np.tile(self.data[k], (self.c, 1)), self.g[h], self.s[i])

                sum_denominator = np.sum((2 - 2 * kernel_values_i) / ((2 - 2 * kernel_values_h) + epsilon))
                sum_denominator = (sum_denominator ** (1 / (self.m - 1)) + epsilon)
                self.U[i][k] = (1/sum_denominator)
        return self.U

#Eq.15b
    def compute_prototypes(self):
      numerator = np.ones(self.p, dtype=np.float128)
      denominator = np.ones(self.p, dtype=np.float128)
      epsilon = 1e-10
      for i in range(self.c):
        for k in range(self.n):
            value_UK  = ((self.U[i][k])**self.m)  * self.gaussian_kernel(self.data, self.g[i], self.s[i])
            numerator = np.sum(value_UK[:, None] * self.data, axis=0)
            denominator = np.sum(value_UK)
        self.g[i] = numerator/(denominator + epsilon)
        return self.g


#eq.14b
    getcontext().prec = 120
    def compute_widths(self):
        for i in range(self.c):
            value_UK = np.multiply((self.U[i])**self.m, self.gaussian_kernel(self.data, self.g[i], self.s[i]))
            for j in range(self.p):
                numerator = Decimal('1')
                expoente = Decimal(self.p)
                pexp = Decimal('1')/expoente
                for h in range(self.p):
                    #Subtracao numerador
                    numdiff = self.data[:,h] - self.g[i,h]
                    # (x-g)^2
                    numerator_aux = np.multiply(value_UK, np.multiply(numdiff,numdiff))
                    numerator_aux = np.sum(numerator_aux)
                    numerator_aux = Decimal(numerator_aux)
                    numerator = numerator*numerator_aux
                numerator = numerator**pexp
                numerator = float(numerator)
                dendiff = self.data[:,j] - self.g[i,j]
                denominator = np.multiply(value_UK, np.multiply(dendiff,dendiff))
                denominator = np.sum(denominator)
                self.s[i, j] = numerator/denominator
        max_value = np.max(self.s)
        self.s = self.s / max_value
        return self.s

#Eq.13
    def compute_JNew(self):
        J = 0
        for i in range(self.c):
            kernel_values = self.gaussian_kernel(self.data, self.g[i], self.s[i])
            J += np.sum((self.U[i]**self.m) * (2 - 2 * kernel_values))
        return J

#Função do Algoritmo
    def fit(self):
        U, g = self.initialize() #inicializa os protótipos
        J_old = np.inf
        contador = 1
        for iter_count in range(self.max_iter):
            s = self.compute_widths() #s janela s
            g = self.compute_prototypes() #g computa os protótipos
            U = self.update_membership() #update dos parâmetros uik da matriz U (matriz de pertinência)
            J_new = self.compute_JNew() #update função de custo
            if abs(J_new - J_old) < self.tol:
                break
            J_old = J_new
            contador += 1
        return s, g, U, J_old

for num_epocas in range(50):
  print("***********************")
  print("Épocas: ", num_epocas)
  if __name__ == '__main__':
    algo = KFCM_K_W2(X)
    algo.fit()
    s = algo.compute_widths()
    g = algo.compute_prototypes()
    U = algo.update_membership()
    J_custo = algo.compute_JNew()

    print("Função de custo: ", J_custo)

    # Partição Crisp
    x = np.array(range(0, 2000))
    pred_label=[]
    for i in range(2000):
      indice_max = np.argmax(U[:,i])
      pred_label.append(indice_max)

    print("U Crisp:",pred_label)

    ##ARI considerando a partição CRISP
    print("Predict labels:",pred_label)
    print("True labels:",np.transpose(label.to_numpy()))
    respred_label= np.reshape(pred_label, (2000,1)).ravel()
    ari=ARI(label.to_numpy().ravel(),respred_label)
    print("ARI:",ari)
    #Calcular
    precision = precision_score(label, pred_label,average='macro')
    print("----------------")
    print("Precision", precision)
    recall = recall_score(label, pred_label, average='macro')
    print("----------------")
    print("Recall", recall)
    f1score = f1_score(label, pred_label, average='macro')
    print("----------------")
    print("F1score", f1score)
    accuracy = accuracy_score(label, pred_label)
    print("----------------")
    print("Accuracy", accuracy)

    #Calcular partition coefficient
    cluster=10
    partition_coefficient = 0
    for i in range(cluster):
      for j in range(len(X)):
        partition_coefficient += U[i,j]**2
    partition_coefficient /= len(X)
    print("Fuzzy Partition Coefficient:", partition_coefficient)

    #Modified partition Coefficient
    MPC = 1 - (cluster/cluster-1)*(1 - partition_coefficient)
    print("Modified partition Coefficient:", MPC)

    #Matriz confusão
    conf_matrix = confusion_matrix(label.to_numpy().ravel(), pred_label)
    # Plotar a matriz de confusão
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cbar=True,
        cmap='YlGnBu'
    )
    ax.set_xlabel("Classificação do modelo", labelpad=20)
    ax.set_ylabel("Classes reais", labelpad=20)
    plt.show()

    # incrementa o número de épocas
    num_epocas += 1

print("Parâmetros de Largura s: ")
    matriz_s = s
    pd.set_option('display.max_columns', 220)
    df_s = pd.DataFrame(matriz_s)
    df_s.style.highlight_max(axis=0, color='green')
    df_s

#Matriz S
html_table = df_s.to_html()
with open('output.html', 'w') as f:
  f.write(html_table)
  options = {
          'format': 'png',
          'width': '1000',
          'encoding': 'utf-8'
      }
imgkit.from_file('output.html', 'output.png', options=options)
Image('output.png')

print("Protótipos de cada grupo g:")
    matriz_g = g
    pd.set_option('display.max_columns', 220)
    df_g = pd.DataFrame(matriz_g)
    df_g.style.highlight_max(axis=0, color='green')
    df_g

#Matriz G
html_table = df_g.to_html()
with open('output.html', 'w') as f:
  f.write(html_table)
  options = {
          'format': 'png',
          'width': '1000',
          'encoding': 'utf-8'
      }
imgkit.from_file('output.html', 'output.png', options=options)
Image('output.png')

print("Matriz U: ")
    matriz_u = U
    pd.set_option('display.max_columns', 220)
    df_u = pd.DataFrame(matriz_u)
    df_u.style.highlight_max(axis=0, color='green')
    df_u
