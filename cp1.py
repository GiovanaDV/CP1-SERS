'''
Individual Household Electric Power Consumption --> contém medições de consumo de energia elétrica em uma casa na França
- registrados a cada min entre 2006 - 2010
'''

import pandas as pd

caminho = "C:\\Users\\giova\\OneDrive\\Desktop\\FIAP\\SERS\\CP1-SERS\\household_power_consumption.txt"

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
import matplotlib.pyplot as plt
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
print("Finais de semana:", round(media_comparacao.loc[True], 2), "kWh")












