'''
Individual Household Electric Power Consumption --> contém medições de consumo de energia elétrica em uma casa na França
- registrados a cada min entre 2006 - 2010
'''

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

caminho = "C:\\Users\\labsfiap\\Downloads\\household_power_consumption.txt"

df = pd.read_csv(
    caminho,
    sep=';',
    na_values='?',
    low_memory=False
)

print(df.columns) # mostra os nomes das colunas

pd.set_option('display.max_columns', None) # fala pra mostrar TODAS as colunas

print("1. Carregue o dataset e exiba as 10 primeiras linhas ----------------------------------------------------------- \n")
print(df.head(10)) # imprime as primeiras 10 linhas
print("\n")

print("3. Verifique se existem valores ausentes no dataset. Quantifique-os.-------------------------------------------- \n")
print(df.isnull().sum()) # verifica e conta os valores nulos por coluna
print("\n")

print("4. Converta a coluna Date para o tipo datetime e crie uma nova coluna com o dia da semana correspondente. ------ \n")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') # converte a coluna Date para o tipo datetime
df['Dia_da_semana'] = df['Date'].dt.day_name() # cria nova coluna com o nome do dia da semana
print(df[['Date', 'Dia_da_semana']].head())  # para checar se deu certo
print("\n")

print("5. Filtre os registros apenas do ano de 2007 e calcule a média de consumo diário de Global_active_power. ------- \n")
df_2007 = df[df['Date'].dt.year == 2007] # registros do ano de 2007
media_diaria = df_2007.groupby('Date')['Global_active_power'].mean() # media diaria Global_active_power
print(media_diaria.head())
media_geral_2007 = media_diaria.mean() # calcula a média de consumo diario
print(f"Média de consumo diário em 2007: {media_geral_2007:.2f} kW \n")

# 6. Gere um gráfico de linha mostrando a variação de Global_active_power em um único dia à sua escolha
dia_escolhido = "2007-03-10" # dia escolhido
df_dia = df[df['Date'] == dia_escolhido]
df_dia['DateTime'] = pd.to_datetime(df_dia['Date'].astype(str) + " " + df_dia['Time']) # converter a coluna Time em datetime junto com Date (eixo X)
plt.figure(figsize=(12, 6)) # plotar o gráfico
plt.plot(df_dia['DateTime'], df_dia['Global_active_power'], color='blue', linewidth=0.7)
plt.title(f"Variação do Consumo de Energia em {dia_escolhido}", fontsize=14)
plt.xlabel("Hora do Dia", fontsize=12)
plt.ylabel("Consumo Global Ativo (kW)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
#plt.show()

# 7. Crie um histograma da variável Voltage.
plt.figure(figsize=(10,6))
plt.hist(df['Voltage'].dropna(), bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.title("Distribuição da Variável Voltage", fontsize=14)
plt.xlabel("Voltage (Volts)", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.show()

print("8. Calcule o consumo médio por mês em todo o período disponível no dataset.  ----------------------------------- \n")
df['AnoMes'] = df['Date'].dt.to_period('M') # agrupando por ano e mes
media_mensal = df.groupby('AnoMes')['Global_active_power'].mean()
print("Consumo médio por mês em todo o período:")
print(media_mensal)
print("\n")

print("9. Identifique o dia com maior consumo de energia ativa global (Global_active_power). -------------------------- \n")
consumo_diario = df.groupby('Date')['Global_active_power'].sum() # consumo diário (valores de cada dia)
dia_maior_consumo = consumo_diario.idxmax() # dia com maior consumo
valor_maior_consumo = consumo_diario.max()
print(f"O dia com maior consumo foi {dia_maior_consumo.date()} com {valor_maior_consumo:.2f} kWh \n")

print("10. Compare o consumo médio de energia ativa global em dias de semana versus finais de semana. ----------------- \n")
df['FimDeSemana'] = df['Dia_da_semana'].isin(['Saturday', 'Sunday']) # ver se é final de semana
consumo_diario = df.groupby(['Date', 'FimDeSemana'])['Global_active_power'].sum().reset_index() # média de consumo diário
media_comparacao = consumo_diario.groupby('FimDeSemana')['Global_active_power'].mean() # média de dias de semana vs fim de semana
print("Dias de semana:", round(media_comparacao.loc[False], 2), "kWh")
print("Finais de semana:", round(media_comparacao.loc[True], 2), "kWh \n")

# correlaçao --> como duas variaveis estao relacionadas / valor proximo de 1: forte correlação positiva / valor proximo de -1: forte correlacao negativa / valor proximo de 0: pouca ou nenhuma correlacao
print("11. Calcule a correlação entre as variáveis Global_active_power, Global_reactive_power, Voltage e Global_intensity \n")
colunas = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
correlacao = df[colunas].corr() # calcula a correlacao
print("Correlação entre as variáveis: ")
print(correlacao)
print("\n")

print("12. Crie uma nova variável chamada Total_Sub_metering que some Sub_metering_1, Sub_metering_2 e Sub_metering_3 --\n")
df['Total_Sub_metering'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
print(df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Total_Sub_metering']].head())
print("\n")

print("13. Verifique se há algum mês em que Total_Sub_metering ultrapassa a média de Global_active_power -------------- \n")
media_mensal_sub = df.groupby('AnoMes')['Total_Sub_metering'].mean()
comparacao = media_mensal_sub > media_mensal
meses_ultrapassados = comparacao[comparacao == True].index
print("Meses em que Total_Sub_metering foi maior que Global_active_power (média mensal): ")
for mes in meses_ultrapassados:
    print(mes)

print("14. Faça um gráfico de série temporal do Voltage para o ano de 2008. ------------------------------------------- \n")
df_2008 = df[df['Date'].dt.year == 2008].copy() # filtrar dados para o ano de 2008
df_2008['DateTime'] = pd.to_datetime(df_2008['Date'].astype(str) + " " + df_2008['Time']) # coluna DateTime com Date e Time
plt.figure(figsize=(14, 6))
plt.plot(df_2008['DateTime'], df_2008['Voltage'], color='green', linewidth=0.5)
plt.title("Variação da Tensão Elétrica (Voltage) em 2008", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Voltage (Volts)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print("\n")

print("15.  Compare o consumo entre os meses de verão e inverno (no hemisfério norte). -------------------------------- \n")
meses_verao = [6, 7, 8]  # junho, julho, agosto
meses_inverno = [12, 1, 2]  # dezembro, janeiro, fevereiro
df['Mes'] = df['Date'].dt.month # pegando mes e ano
media_mensal = df.groupby('Mes')['Global_active_power'].mean()
media_verao = media_mensal.loc[meses_verao].mean()
media_inverno = media_mensal.loc[meses_inverno].mean()
print(f"Consumo médio mensal nos meses de VERÃO (jun, jul, ago): {media_verao:.2f} kW")
print(f"Consumo médio mensal nos meses de INVERNO (dez, jan, fev): {media_inverno:.2f} kW \n")

print("16. Aplique uma amostragem aleatória de 1% dos dados e verifique se a distribuição de Global_active_power é semelhante à da base completa. \n")
df_amostra = df.sample(frac=0.01, random_state=42) # amostragem aleatória de 1% dos dados
plt.figure(figsize=(14,6))
plt.subplot(1, 2, 1) # histograma da base completa
plt.hist(df['Global_active_power'].dropna(), bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribuição de Global_active_power (Base Completa)')
plt.xlabel('Global_active_power (kW)')
plt.ylabel('Frequência')
plt.subplot(1, 2, 2) # histograma da amostra 1%
plt.hist(df_amostra['Global_active_power'].dropna(), bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribuição de Global_active_power (Amostra 1%)')
plt.xlabel('Global_active_power (kW)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()
print("Estatísticas da base completa:")
print(df['Global_active_power'].describe())
print("\nEstatísticas da amostra 1%:")
print(df_amostra['Global_active_power'].describe())

print("17. Utilize uma técnica de normalização (Min-Max Scaling) para padronizar as variáveis numéricas principais. ----\n")
colunas_numericas = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Total_Sub_metering'] # variáveis numéricas principais
scaler = MinMaxScaler()
df_normalizado = df.copy() # copia dataset original
df_normalizado[colunas_numericas] = scaler.fit_transform(df_normalizado[colunas_numericas])
print("Antes da normalização:")
print(df[colunas_numericas].describe().loc[['min', 'max']])
print("\nDepois da normalização:")
print(df_normalizado[colunas_numericas].describe().loc[['min', 'max']])

print("18. Aplique K-means para segmentar os dias em 3 grupos distintos de consumo elétrico. ---------------------------\n")
consumo_diario = df.groupby('Date')['Global_active_power'].sum().reset_index()
X = consumo_diario[['Global_active_power']]
kmeans = KMeans(n_clusters=3, random_state=42) # aplicando K means
consumo_diario['cluster'] = kmeans.fit_predict(X)
print(consumo_diario['cluster'].value_counts())
print("\n")
print("Centroids (média do consumo por cluster):")
print(kmeans.cluster_centers_)
print("\n")

print("19. Realize uma decomposição de série temporal (tendência, sazonalidade e resíduo) para Global_active_power em um período de 6 meses. \n")
inicio = '2007-01-01' # filtrando um periodo de 6 meses
fim = '2007-06-30'
df_6meses = df[(df['Date'] >= inicio) & (df['Date'] <= fim)]
serie_diaria = df_6meses.groupby('Date')['Global_active_power'].mean()
serie_diaria = serie_diaria.interpolate(method='linear')
serie_diaria.index = pd.to_datetime(serie_diaria.index)
decomposicao = seasonal_decompose(serie_diaria, model='additive', period=7) # decomposicao
decomposicao.plot()
plt.suptitle('Decomposição da Série Temporal - Global_active_power (6 meses)', fontsize=16)
plt.tight_layout()
plt.show()

print("20. Treine um modelo de regressão linear simples para prever Global_active_power a partir de Global_intensity. --\n")
df_reg = df[['Global_active_power', 'Global_intensity']].dropna()
X = df_reg[['Global_intensity']].values  # variavel independente
y = df_reg['Global_active_power'].values # variavel dependente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression() # criando e treinando o modelo
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente angular (slope): {modelo.coef_[0]:.4f}")
print(f"Intercepto (intercept): {modelo.intercept_:.4f}")
print(f"R² (coeficiente de determinação): {r2:.4f}")
plt.figure(figsize=(10,6)) # grafico
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Previsão (reta de regressão)')
plt.title("Regressão Linear Simples: Global_active_power vs Global_intensity")
plt.xlabel("Global_intensity")
plt.ylabel("Global_active_power (kW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

'''
Parte 2: Exercícios adicionais no dataset inicial
'''

print("21. Treine um modelo de regressão linear simples para prever Global_active_power a partir de Global_intensity. --\n")







