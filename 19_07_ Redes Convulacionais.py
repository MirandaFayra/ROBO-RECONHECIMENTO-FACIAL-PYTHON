#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install keras')


# Importando modulos

# In[8]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


# In[12]:


lfw_people = fetch_lfw_people (min_faces_per_person=70, resize=0.4) #trazendo os dados para o phyton


# In[13]:


imagemTeste = lfw_people.images[15] #pegando imagem 15


# In[16]:


plt.imshow(imagemTeste, cmap='gray') #visualizando minha imagem teste


# In[18]:


lfw_people.target[15]


# In[20]:


lfw_people.target_names [3]


# In[22]:


lfw_people.target_names #nomes de todas as pessoas do banco de dados


# In[21]:


#Separando os dados
x = lfw_people.data
y =lfw_people.target


# In[25]:


#Transformando dados para o Tensorflow
from keras.utils import np_utils


# In[27]:


y2 = np_utils.to_categorical(y)


# In[29]:


# ajustes 1288 qtd linhas, 1850 qtdcolunas
x.shape


# In[30]:


# (1288 = qtd de fotos, 50 por 37 dimensões da foto,1 constante do tensor flow)
x2 = x.reshape (1288, 50, 37,1)


# Criar dados de treino e teste

# In[31]:


from sklearn.model_selection import train_tast_split


# In[ ]:




