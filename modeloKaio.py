# Importando as bibliotecas necessárias para esse projeto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pycaret
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import setup, create_model, predict_model, save_model

df = pd.read_csv("train.csv/train.csv")

# Selecionando as features relevantes
features = ['danceability', 'energy', 'tempo', 'valence', 'explicit', 'track_genre', 
            'duration_ms', 'speechiness', 'loudness']

# Filtrando apenas as colunas selecionadas
X = df[features]

# Target - 'popularity_target' será a variável alvo
y = df['popularity_target']

# Convertendo 'explicit' que é uma variável booleana para 0/1
X['explicit'] = X['explicit'].astype(int)

# Convertendo o gênero musical 'track_genre' para valores numéricos
label_encoder = LabelEncoder()
X['track_genre'] = label_encoder.fit_transform(X['track_genre'])

from pycaret.classification import setup, compare_models

# Configuração do PyCaret
clf_setup = setup(data=df,
                  target='popularity_target', 
                  numeric_features=['danceability', 'energy', 'tempo', 'valence', 'explicit', 'duration_ms', 
                                    'speechiness', 'loudness'],
                  categorical_features=['track_genre'])

# Comparar todos os modelos e selecionar o melhor
best_model = compare_models(sort='Accuracy', verbose=True)