# MeteoForecast Visualizer

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.6%2B-green)

Framework modular para análise e visualização de dados meteorológicos processados por diferentes arquiteturas de redes neurais (FCNN, LSTM, GNN). Simplifica o fluxo de trabalho entre múltiplas equipes que desenvolvem modelos distintos.

## 🌟 Características

- **Processamento Unificado**: Interface comum para modelos FCNN, LSTM e GNN
- **Métricas Avançadas**: Cálculo automático de MSE, RMSE, R² para avaliação de desempenho
- **Visualizações Ricas**: Geração de mapas meteorológicos, GIFs animados e grids comparativos
- **Configuração Simplificada**: Definição hard-coded ou via linha de comando
- **Modularidade**: Estrutura em componentes independentes para fácil extensão

## 🔧 Pré-requisitos

- Python 3.6 ou superior
- Bibliotecas principais:
  - pandas, numpy
  - matplotlib, cartopy
  - torch
  - PIL
  - sklearn

## 📊 Exemplos de Saída

- Mapas meteorológicos comparando previsão vs. realidade
- Cálculo de erro absoluto entre predições e valores reais
- Métricas de precisão para cada dia de previsão
- GIFs animados mostrando a evolução temporal das previsões

## 🔍 Detalhes dos Módulos

### main.py
- Define a configuração do processamento
- Oferece interface simplificada para execução

### processor.py
- Carrega e prepara os dados dos modelos
- Coordena cálculo de métricas e geração de visualizações

### metrics.py
- Calcula MSE, RMSE, R² para avaliação de desempenho
- Fornece utilidades como cálculo de MAPE e adição de ruído

### visualizer.py
- Gera grades de comparação visual entre modelos
- Cria GIFs animados para visualizar previsões ao longo do tempo

