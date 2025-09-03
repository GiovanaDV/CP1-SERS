# Exercício 26 - Carregamento e inspeção inicial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Carregar dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
df = pd.read_csv(url)

print("Primeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações gerais do dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# Exercício 27 - Distribuição do consumo
plt.figure(figsize=(10,5))
plt.hist(df['Appliances'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histograma do Consumo de Energia (Appliances)')
plt.xlabel('Consumo (Wh)')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df['Appliances'], color='orange')
plt.title('Série Temporal do Consumo de Energia (Appliances)')
plt.xlabel('Amostras (tempo)')
plt.ylabel('Consumo (Wh)')
plt.show()

# Exercício 28 - Correlações com variáveis ambientais
cols_ambientais = [
    'T1','T2','T3','T4','T5','T6','T7','T8','T9',
    'RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9',
    'T_out','RH_out'
]

correlacoes = df[cols_ambientais + ['Appliances']].corr()['Appliances'].sort_values(ascending=False)
print("Correlação de Appliances com variáveis ambientais:")
print(correlacoes)

# Exercício 29 - Normalização dos dados
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
colunas_numericas.remove('Appliances')  # manter target sem normalizar

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

print("Dataset após normalização (Min-Max Scaling):")
print(df_scaled.head())

# Exercício 30 - PCA e visualização
X = df_scaled[colunas_numericas]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1','PC2'])

plt.figure(figsize=(10,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df['Appliances'], cmap='viridis', s=10)
plt.colorbar(label='Appliances (Wh)')
plt.title('PCA - 2 Componentes Principais')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Exercício 31 - Regressão Linear Múltipla
y = df['Appliances']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² do modelo: {r2:.4f}")
print(f"Erro médio absoluto (MAE): {mae:.2f} Wh")

# Exercício 32 - Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"RMSE - Regressão Linear: {rmse_lr:.2f} Wh")
print(f"RMSE - Random Forest: {rmse_rf:.2f} Wh")

# Exercício 33 - K-Means Clustering
X_cluster = X
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_cluster)
df_scaled['Cluster'] = clusters

consumo_por_cluster = df_scaled.groupby('Cluster')['Appliances'].mean()
print("Consumo médio de Appliances por cluster:")
print(consumo_por_cluster)

# Visualização dos clusters em 2D (PCA)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_cluster)

plt.figure(figsize=(10,6))
plt.scatter(X_pca_2d[:,0], X_pca_2d[:,1], c=clusters, cmap='viridis', s=10)
plt.title('K-Means Clustering (4 clusters) - PCA 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Exercício 34 - Classificação binária
mediana_consumo = df['Appliances'].median()
df_scaled['Consumo_Alto'] = (df['Appliances'] > mediana_consumo).astype(int)

X = df_scaled[colunas_numericas]
y = df_scaled['Consumo_Alto']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("Logistic Regression:")
print(classification_report(y_test, y_pred_log, target_names=['Baixo', 'Alto']))

print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf, target_names=['Baixo', 'Alto']))

# Exercício 35 - Avaliação de classificação
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Baixo', 'Alto'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.show()

print("Métricas - Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf, target_names=['Baixo', 'Alto']))

