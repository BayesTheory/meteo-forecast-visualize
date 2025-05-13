# MeteoForecast Visualizer & Analyzer

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)

Framework modular para análise, visualização e comparação de dados meteorológicos processados por diferentes arquiteturas de redes neurais (FCNN, LSTM, GNN). Projetado para simplificar o fluxo de trabalho em lote e a avaliação de múltiplos modelos.

## 🌟 Características Principais

-   **Processamento em Lote Configurável**: Execute análises para múltiplos modelos e configurações através de um único arquivo `job_config.json`.
-   **Interface Unificada**: Suporte para modelos FCNN, LSTM e GNN através de um pipeline de processamento comum.
-   **Métricas Detalhadas**: Cálculo automático de MSE, RMSE e R² para cada dia da janela de previsão.
-   **Visualizações Ricas por Modelo**:
    -   Mapas meteorológicos comparando previsão vs. realidade.
    -   GIFs animados mostrando a evolução temporal das previsões e das diferenças.
    -   Grids comparativos de imagens.
    -   Gráficos da evolução diária de RMSE, MSE e R² para cada modelo individualmente.
-   **Relatórios de Resumo Comparativo**:
    -   Geração automática de gráficos comparando o RMSE, MSE e R² acumulados de todos os modelos processados.
    -   Criação de uma tabela sumarizada em formato de imagem, comparando as métricas diárias e médias de todos os modelos.
-   **Modularidade**: Estrutura em componentes independentes (`main`, `processor`, `metrics`, `visualizer`, `reporting`) para fácil manutenção e extensão.
-   **Configuração Flexível**: Definição de tarefas via arquivo JSON centralizado.

## 🔍 Detalhes dos Módulos

-   **`main.py`**:
    -   Ponto de entrada principal do framework.
    -   Lê o `job_config.json` para obter a lista de tarefas.
    -   Orquestra o processamento de cada tarefa.
    -   Coordena a geração dos relatórios de resumo finais (gráficos comparativos e tabela de métricas).

-   **`processor.py`**:
    -   Carrega e pré-processa os dados de um arquivo `.pkl` específico do modelo.
    -   Adiciona colunas de métricas diárias (RMSE, MSE, R²) ao DataFrame da amostra.
    -   Calcula métricas agregadas sobre as amostras.
    -   Chama o `visualizer.py` para gerar as visualizações específicas do modelo.
    -   Retorna as métricas calculadas para o `main.py`.

-   **`metrics.py`**:
    -   `posprocessDataframe()`: Calcula métricas diárias (MSE, RMSE, R²) por amostra no DataFrame.
    -   `calculate_model_metrics()`: Agrega as métricas diárias sobre todas as amostras.
    -   Contém funções utilitárias adicionais (MAPE, magnitude, ruído).

-   **`visualizer.py`**:
    -   `generate_visualizations()`: Função principal para criar todas as saídas visuais para uma única tarefa/modelo.
    -   `plot_daily_metric_for_model()`: Gera gráficos da evolução diária de RMSE, MSE e R² para o modelo.
    -   Funções para criar grades de comparação, mapas e GIFs animados (ex: `plot_images_in_grid`, `get_gif_forecasting`).

-   **`reporting.py`**:
    -   `plot_cumulative_metric_graph()`: Cria gráficos comparativos de métricas (RMSE, R², MSE) acumuladas ao longo dos dias para todos os modelos processados.
    -   `create_metrics_summary_table()`: Gera uma imagem de tabela resumida comparando as métricas diárias e médias de todos os modelos.

## 🚀 Próximos Passos e Contribuições

-   [ ] Adicionar suporte para mais tipos de modelos.
-   [ ] Implementar mais métricas de avaliação.
-   [ ] Melhorar a customização dos gráficos e tabelas via configuração.
-   [ ] Adicionar testes unitários.



