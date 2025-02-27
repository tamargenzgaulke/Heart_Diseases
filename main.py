#%%
!pip install seaborn

#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


#%%
# Carregar o dataset
df = pd.read_csv("/Users/tamargenzgaulke/Desktop/Data Science/Heart_Disease/heart_disease_uci copy.csv") 

#%%# Exibir as primeiras linhas
display(df.head())

# Informações do dataset
display(df.info())

# Verificar valores ausentes
print("Valores nulos:")
print(df.isnull().sum())

#%% Limpeza dos dados

# Remover colunas que não serão utilizadas
df.drop(columns=['id','dataset'], inplace=True)


#%%
# Remover colunas com muitos valores nulos
df = df.drop(columns=['ca', 'thal', 'slope'])
#%%
# Preencher valores nulos em colunas numéricas com a mediana
df.loc[:, 'trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df.loc[:, 'chol'] = df['chol'].fillna(df['chol'].median())
df.loc[:, 'thalch'] = df['thalch'].fillna(df['thalch'].median())
df.loc[:, 'oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].median())

# Preencher valores nulos em colunas categóricas com a moda
df.loc[:, 'fbs'] = df['fbs'].fillna(df['fbs'].mode()[0])
df.loc[:, 'restecg'] = df['restecg'].fillna(df['restecg'].mode()[0])
df.loc[:, 'exang'] = df['exang'].fillna(df['exang'].mode()[0])


#%%
df.info()


#%% Transformando variáveis explicativas categóricas em dummies

df = pd.get_dummies(df, 
                       columns=['sex', 'exang', 'fbs'], 
                       drop_first=True,
                       dtype='int')


#%%
df = pd.get_dummies(df, 
                       columns=['cp', 'restecg'], 
                       drop_first=False,
                       dtype='int')

#%%
# Visualizar a distribuição da variável alvo
sns.countplot(x='num', data=df)
plt.title('Distribuição de Casos de Doença Cardíaca')
plt.show()


#%%
# Separar variáveis independentes e dependente
X = df.drop(columns=['num'])  # Features
y = df['num']  # Target

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% Gerando a árvore de decisão

# Vamos iniciar com uma árvore pequena: profundidade máxima 2 (max_depth)
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=100)
tree_clf.fit(X_train, y_train)

#%% Plotando a árvore

plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_clf,
          feature_names=X.columns.tolist(),
          class_names=['Sem doença','Com doença'],
          proportion=False,
          filled=True,
          node_ids=True)
plt.show()

#%% Analisando os resultados dos splits

tree_split = pd.DataFrame(tree_clf.cost_complexity_pruning_path(X_train, y_train))
tree_split.sort_index(ascending=False, inplace=True)

print(tree_split)

#%% Importância das variáveis preditoras

tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':tree_clf.feature_importances_})

print(tree_features)

#%%

# Treinar o modelo corretamente
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=100)
tree_clf.fit(X_train, y_train)

# Fazer previsões
y_pred = tree_clf.predict(X_test)


#%%
# 4. Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')
print('Relatório de Classificação:\n', classification_report(y_test, y_pred))

#%%

# Importar bibliotecas para SHAP
!pip install shap
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#%%
rf = RandomForestClassifier()
rf.fit(X, y)

#%% Calcular os 'shap values'
amostra = X_test.sample(frac=0.1)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(amostra)

#%% Gráfico Resumo
shap.summary_plot(shap_values, amostra, feature_names=X.columns)