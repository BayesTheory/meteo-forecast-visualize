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

## 🔧 Pré-requisitos

-   Python 3.8 ou superior (devido a algumas funcionalidades e dependências mais recentes)
-   Bibliotecas Python (instalar via `pip install -r requirements.txt` ou individualmente):
    -   `pandas`
    -   `numpy`
    -   `matplotlib`
    -   `cartopy` (para mapas meteorológicos)
    -   `torch` (usado em `metrics.py` para alguns cálculos)
    -   `Pillow` (PIL) (para manipulação de imagens e GIFs)
    -   `scikit-learn` (para `r2_score`)

    _Nota: A instalação do Cartopy pode ter dependências de sistema adicionais (como GEOS, PROJ). Consulte a documentação do Cartopy para detalhes._

## 🚀 Como Usar

1.  **Clone o Repositório:**
    ```
    git clone [URL_DO_SEU_REPOSITORIO]
    cd MeteoForecastVisualizer
    ```

2.  **Instale as Dependências:**
    ```
    pip install pandas numpy matplotlib cartopy torch Pillow scikit-learn
    # Ou, se você criar um requirements.txt:
    # pip install -r requirements.txt
    ```

3.  **Prepare seus Dados:**
    *   Coloque seus arquivos de modelo `.pkl` no diretório configurado (ex: `/workspace/EXPORT/resultadospkl/`). Cada arquivo `.pkl` deve ser um DataFrame pandas contendo as colunas necessárias (ex: `y_rol`, `y_rol_pred`, `lat`, `lon`, `data`).

4.  **Configure as Tarefas de Processamento:**
    *   Edite o arquivo `job_config.json` localizado no mesmo diretório que `main.py`.
    *   Defina uma lista de tarefas em `model_tasks`. Para cada tarefa, especifique:
        *   `task_id`: Um identificador único.
        *   `model_type`: "FCNN", "LSTM", ou "GNN".
        *   `model_file`: Caminho completo para o arquivo `.pkl` do modelo.
        *   `output_directory`: Pasta onde os resultados visuais desta tarefa específica serão salvos.
        *   `visualization_pos`: Índice do dia (0-6) a ser destacado nas visualizações principais (GIFs, grids).
        *   `enabled`: `true` para executar a tarefa, `false` para pulá-la.
    *   Exemplo de `job_config.json`:
        ```
        {
          "model_tasks": [
            {
              "task_id": "FCNN_ModeloA",
              "model_type": "FCNN",
              "model_file": "/caminho/para/seu/modelo_a.pkl",
              "output_directory": "/caminho/para/saidas/FCNN_ModeloA_Output",
              "visualization_pos": 0,
              "enabled": true
            },
            {
              "task_id": "LSTM_ModeloB_Desabilitado",
              "model_type": "LSTM",
              "model_file": "/caminho/para/seu/modelo_b.pkl",
              "output_directory": "/caminho/para/saidas/LSTM_ModeloB_Output",
              "visualization_pos": 0,
              "enabled": false
            }
            // ... mais tarefas
          ]
        }
        ```

5.  **Execute o Script Principal:**
    ```
    python main.py
    ```
    O script processará todas as tarefas habilitadas no `job_config.json`.

## 📊 Saídas Geradas

Para cada tarefa habilitada no `job_config.json`, serão gerados arquivos no `output_directory` especificado:
-   Mapas meteorológicos comparando previsão vs. realidade.
-   Cálculo de erro absoluto entre predições e valores reais (se implementado no `visualizer.py`).
-   GIFs animados mostrando a evolução temporal das previsões.
-   Gráficos da evolução diária de RMSE, MSE e R² para o modelo.

Adicionalmente, na pasta `relatorios_finais_batch` (ou o nome configurado em `main.py`), serão gerados:
-   **Gráficos Comparativos de Métricas Acumuladas**: `cumulative_rmse.png`, `cumulative_r2.png`, `cumulative_mse.png`.
-   **Tabela de Resumo das Métricas**: `resumo_metricas_modelos.png` (comparando todos os modelos processados).

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

Contribuições são bem-vindas! Por favor, abra uma *issue* para discutir mudanças ou envie um *pull request*.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo `LICENSE` para detalhes.
