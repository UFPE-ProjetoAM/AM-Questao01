# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#%%
pip install -U scikit-fuzzy
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.pyplot import figure
from decimal import Decimal, getcontext
#%%
data = pd.read_csv('/kaggle/input/projeto-aprendizado-de-mquina/mfeat-fou', header=None,delim_whitespace=True)
label= pd.read_csv('/kaggle/input/label-kfcm/label.csv', header=None,delim_whitespace=True)
#%%
X1=data.to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X1)
#%%
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

#Testar colocando o módulo
    def gaussian_kernel(self, x, g, s):
          kernel = kernel = np.exp(-0.5 * (np.sum((x - g)**2 * s, axis=1)))
          #np.exp(-0.5 * np.sum([(x[:, j] - g[:, j])**2 * s[:, j]]))
          #print("Gaussian Kernel:", kernel)
          return kernel

        #print("x", x)
        #print("g", g)
        #print("s", s)
        #print("np.sum((x - g)**2 * s, axis=1)", np.sum((x - g)**2 * s, axis=1))
        

    def initialize(self):
        self.U = np.random.rand(self.c, self.n)
        #self.U /= self.U.sum(axis=0)
        self.U /= self.U.sum(axis=1)[:, np.newaxis]
        #self.g = self.data[[0,15,25],:]
        idx = np.random.choice(self.n, self.c, replace=False)
        self.g = self.data[idx]

        #printando a matriz U
        #print("Matriz U:",self.U)
        #print(self.U.shape)
        #print("Soma dos elementos da linha da matriz U", np.sum(self.U[0,:]))
        #print("Soma dos elementos da linha da matriz U", np.sum(self.U[1,:]))
        #print("Soma dos elementos da linha da matriz U", np.sum(self.U[2,:]))
        #print("Matriz G:",self.g)
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
                sum_denominator = sum_denominator ** (1 / (self.m - 1))
                self.U[i][k] = (1/sum_denominator)
                #if kernel_values_h[k]>=1:
                    #self.U[i][k] = 1

                #print("Update U", self.U)
        return self.U

#Eq.15b
    def compute_prototypes(self):
      numerator = np.zeros(self.p, dtype=np.float128)
      denominator = np.zeros(self.p, dtype=np.float128)

      for i in range(self.c):
        for k in range(self.n):
            value_UK  = ((self.U[i][k])**self.m)  * self.gaussian_kernel(self.data, self.g[i], self.s[i])
            numerator = np.sum(value_UK[:, None] * self.data, axis=0)
            denominator = np.sum(value_UK)
        self.g[i] = numerator/denominator
        #print(self.g)
        #return self.g


#eq.14b
    getcontext().prec = 120
    def compute_widths(self):
        for i in range(self.c):
            # U*kernel
            #value_UK = np.multiply(self.U[:i], kernel_values[:i])
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
                    #print("numerador", numerator)
                    numerator_aux = np.sum(numerator_aux)
                    numerator_aux = Decimal(numerator_aux)
                    numerator = numerator*numerator_aux
                numerator = numerator**pexp
                numerator = float(numerator)
                #print("numerator", numerator)
                dendiff = self.data[:,j] - self.g[i,j]
                denominator = np.multiply(value_UK, np.multiply(dendiff,dendiff))
                denominator = np.sum(denominator)
                #print("denominator_sum", denominator)
                self.s[i, j] = numerator/denominator
        max_value = np.max(self.s)
        self.s = self.s / max_value
        #mean = np.mean(self.s)
        #std_dev = np.std(self.s)
        #self.s = (self.s - mean) / std_dev
        #print("Update S", self.s)
        return self.s
#Eq.13
    def compute_JNew(self):
        J = 0
        for i in range(self.c):
            kernel_values = self.gaussian_kernel(self.data, self.g[i], self.s[i])
            J += np.sum((self.U[i]**self.m) * (2 - 2 * kernel_values))
        return J

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
#%%
num_epocas = 1
for num_epocas in range(50):
  print("***********************")
  print("Épocas: ", num_epocas)
  if __name__ == '__main__':
    algo = KFCM_K_W2(X)
    s, g, U, J_custo = algo.fit()

    print("Função de custo: ", J_custo)
    print("----------------")
    print("Parâmetros de Largura s: ", s)
    print("----------------")
    print("Protótipos de cada grupo g:", g)
    print("----------------")
    print("Matriz U: ", U)

    # Partição Crisp
    x = np.array(range(0, 2000))
    pred_label=[]
    for i in range(2000):
      indice_max = np.argmax(U[:,i])
      pred_label.append(indice_max)

    #plt.scatter(x, y, marker='x')
    #plt.title("U Crisp elements")
    #plt.xlabel("Amostras")
    #plt.ylabel("Classes")
    #plt.show()
    print("U Crisp:",pred_label)

    ##ARI considerando a partição CRISP
    print("Predict labels:",pred_label)
    print("True labels:",np.transpose(label.to_numpy()))
    respred_label= np.reshape(pred_label, (2000,1)).ravel()
    ari=ARI(label.to_numpy().ravel(),respred_label)
    print("ARI:",ari)

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
