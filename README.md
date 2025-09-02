# CP1-SERS
CP1 2 Semestre de SERS
Giovana Dias Valentini RM562390

# CHECKPOINT 01 – Data Science e Machine Learning no Python e Orange Data Mining

Perguntas dissertativas 

PARTE 1 – Exercícios iniciais com Individual Household Electric Power Consumption 

2. Explique a diferença entre as variáveis Global_active_power e Global_reactive_power. 
  - As vari[aveis representam tipos diferentes de potência elétrica. A Global_active_power representa a potência ativa, a energia realmente consumida para realizar trabalho útil (a que vai ser consumida). Já a Global_reactive_power mostra a potência reativa, a energia que circula entre os componentes sem ser consumida (ocupa espaço, circula)

6. Gere um gráfico de linha mostrando a variação de Global_active_power em um único dia à 
sua escolha. 
<img width="1497" height="746" alt="Captura de tela 2025-08-30 092723" src="https://github.com/user-attachments/assets/21d52440-fa3e-46b7-97ed-4503badf744d" />

7. Crie um histograma da variável Voltage. O que pode ser observado sobre sua distribuição? 
<img width="1238" height="740" alt="Captura de tela 2025-08-30 092740" src="https://github.com/user-attachments/assets/af01de79-7ce7-471b-83da-e4593a2ccae0" />
  - É possível observar que o gráfico tem um formato de curva em sino, isso significa então que a maioria das medidas de tensão elétrica fica em torno de um valor central.
  - A variável Voltage ficou concentrada em torno de 240V, mostrando que o fornecimento de energia foi relativamente estável.

18. Aplique K-means para segmentar os dias em 3 grupos distintos de consumo elétrico. 
Interprete os resultados.
  - Cluster 1 (933 kWh/dia, 505 dias) → baixo consumo, Cluster 0 (1699 kWh/dia, 741 dias) → consumo médio (mais comum), Cluster 2 (2608 kWh/dia, 196 dias) → alto consumo.
  - A casa apresenta principalmente dias de consumo médio, com alguns de baixo e poucos de alto consumo.

20. Treine um modelo de regressão linear simples para prever Global_active_power a partir de 
Global_intensity. Avalie o erro do modelo
  - O modelo tem erro muito baixo e é altamente confiável para prever o consumo a partir da intensidade.

22. Pergunta: existem padrões repetidos diariamente?
    - há padrões de consumo que se repetem diariamente, mas a regularidade diminui conforme o intervalo de tempo aumenta.

23.  Analise a variância explicada por cada componente. 
  - Componente 1 explica 60,86% da variação dos dados, e a Componente 2 explica 22,43%.
  - Elas explicam 83,29% da variância total, então, ambas as componentes mantêm a maior parte da informação original das 4 variáveis, permitindo reduzir a dimensionalidade com pouca perda de dados.

24.  Pergunta: os grupos se separam de forma clara?
  -  Os clusters não estão claramente separados no espaço PCA.
  -  Há bastante sobreposição entre os grupos.
    
25.  Analise RMSE e a curva ajustada.
  - A regressão polinomial ajustou-se melhor aos dados, visualmente e numericamente (menor RMSE).
  - A regressão linear foi simples demais para capturar a relação entre Voltage e Global_active_power.
