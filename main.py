import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv("/Users/tamargenzgaulke/Documents/GitHub/Heart_Diseases/heart_disease_uci.csv") 

df.info
df.head

## Continuar com o sandbox

 

