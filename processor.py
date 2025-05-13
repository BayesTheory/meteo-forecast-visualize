# processor.py
import os
import pandas as pd
import numpy as np
# Certifique-se de que os caminhos de importação para metrics e visualizer estão corretos
# Se eles estiverem no mesmo diretório, as importações diretas funcionam.
# Ex: from metrics import calculate_model_metrics (se metrics.py estiver no mesmo nível)
# Se estiverem em subpastas ou de forma diferente, ajuste o sys.path ou use imports relativos.
from metrics import posprocessDataframe, calculate_model_metrics
from visualizer import generate_visualizations

def process_model(model_type, file_path, output_dir, pos=0):
    """
    Processa um modelo, calcula métricas e gera visualizações.
    O output_dir fornecido já é o diretório final e específico da tarefa.

    Parameters:
    -----------
    model_type : str
        Tipo de modelo ('FCNN', 'LSTM', 'GNN')
    file_path : str
        Caminho para o arquivo .pkl do modelo
    output_dir : str
        Diretório final para salvar todas as saídas desta tarefa específica.
    pos : int
        Posição (dia) para visualização.

    Returns:
    --------
    dict or None
        Um dicionário contendo as métricas calculadas (ex: {'rmse': [...], 'r2': [...]})
        ou None se o processamento da tarefa falhar criticamente (ex: dados não puderam ser carregados).
    """
    print(f"Iniciando processamento do modelo {model_type}...")
    print(f"  Lendo modelo de: {file_path}")
    print(f"  Salvando resultados em: {output_dir}")

    # Carregar e processar dados
    # A função load_model_data agora retorna None em caso de erro crítico
    df = load_model_data(file_path, model_type)

    if df is None:
        print(f"Falha ao carregar/processar dados para o modelo {model_type} do arquivo {file_path}.")
        print(f"Abortando processamento da tarefa para {model_type}.")
        return None # Sinaliza para o main.py que esta tarefa falhou

    # Calcular e exibir métricas
    metrics = calculate_model_metrics(df) # Supondo que esta função exista em metrics.py

    print("\n===== MÉTRICAS =====")
    if metrics and isinstance(metrics, dict) and 'rmse' in metrics and 'r2' in metrics and \
       len(metrics['rmse']) > 0 and len(metrics['r2']) > 0:
        for day in range(min(7, len(metrics['rmse']))): # Exibe métricas para até 7 dias
            print(f"Dia {day+1}:")
            print(f"  RMSE: {metrics['rmse'][day]:.4f}")
            print(f"  R²: {metrics['r2'][day]:.4f}")
    else:
        print("AVISO: Métricas não foram calculadas corretamente ou estão ausentes.")
        # Mesmo sem métricas válidas, podemos tentar gerar visualizações se o df existir.
        # Se as métricas forem cruciais para a visualização, você pode querer retornar None aqui também.

    # O diretório 'output_dir' já foi criado pelo main.py.
    # Apenas garantimos que ele exista (não prejudica verificar novamente).
    os.makedirs(output_dir, exist_ok=True)

    # Gerar visualizações
    print("\nGerando visualizações...")
    try:
        # Passe o 'output_dir' diretamente para generate_visualizations,
        # pois ele já é o diretório final e específico da tarefa.
        generate_visualizations(df, model_type, output_dir, position=pos)
    except Exception as e_vis:
        print(f"ERRO ao gerar visualizações para {model_type}: {e_vis}")
        # Decida se um erro na visualização deve invalidar o retorno das métricas.
        # Por ora, as métricas ainda serão retornadas se calculadas.

    print(f"Processamento do modelo {model_type} concluído!")
    # Retorna o dicionário de métricas para ser coletado pelo main.py
    # Se as métricas não foram calculadas corretamente, 'metrics' pode ser None ou um dict incompleto.
    # O main.py e a função de relatório devem lidar com isso.
    return metrics

def load_model_data(file_path, model_type_info="modelo"): # model_type é mais para log aqui
    """
    Carrega e prepara os dados de um modelo a partir de um arquivo .pkl.

    Parameters:
    -----------
    file_path : str
        Caminho para o arquivo .pkl.
    model_type_info : str
        Informação do tipo de modelo, usada para mensagens de log.

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame processado ou None em caso de erro crítico.
    """
    print(f"Carregando dados para {model_type_info} do arquivo: {file_path}")

    try:
        df = pd.read_pickle(file_path)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de modelo {file_path} não encontrado.")
        return None
    except Exception as e_pickle:
        print(f"ERRO CRÍTICO ao ler o arquivo pickle {file_path}: {e_pickle}")
        return None

    try:
        # Remover espaços extras nos nomes das colunas
        df.columns = df.columns.str.strip()

        # Criar a coluna 'data' caso não exista, a partir de 'dia_mes_ano'
        if 'data' not in df.columns and 'dia_mes_ano' in df.columns:
            df["data"] = pd.to_datetime(df["dia_mes_ano"]) + pd.Timedelta(hours=12)
        elif 'data' not in df.columns:
            print("AVISO: Coluna 'data' não encontrada e 'dia_mes_ano' também não presente para criar 'data'.")
            # Considere se a coluna 'data' é essencial. Se for, retorne None.
            # if True: # se 'data' for obrigatória
            #     print("ERRO CRÍTICO: Coluna 'data' é obrigatória e não pôde ser criada.")
            #     return None

        # Reshape para o formato esperado (354*360, 7)
        # Verifique se as colunas existem antes de tentar o reshape
        required_reshape_cols = ['y_rol', 'y_rol_pred']
        for col_name in required_reshape_cols:
            if col_name not in df.columns:
                print(f"ERRO CRÍTICO: Coluna obrigatória para reshape '{col_name}' não encontrada no DataFrame.")
                return None

        # Assegure-se que o conteúdo das colunas é compatível com np.array e reshape
        # Esta é uma operação delicada e depende muito do formato exato dos seus dados no pickle
        try:
            df['y_rol'] = df['y_rol'].apply(lambda x: np.array(x, dtype=float).reshape(354*360, 7))
            df['y_rol_pred'] = df['y_rol_pred'].apply(lambda x: np.array(x, dtype=float).reshape(354*360, 7))
        except Exception as e_reshape:
            print(f"ERRO CRÍTICO durante o reshape das colunas 'y_rol' ou 'y_rol_pred': {e_reshape}")
            # Tentar imprimir informações de debug sobre as formas
            if 'y_rol' in df.columns and len(df['y_rol']) > 0:
                 print(f"  Exemplo de tipo de dado em y_rol[0]: {type(df['y_rol'].iloc[0])}")
                 if isinstance(df['y_rol'].iloc[0], (list, np.ndarray)):
                     print(f"  Exemplo de forma de y_rol[0]: {np.array(df['y_rol'].iloc[0]).shape}")
            return None

        # Pós-processamento do DataFrame (ex: cálculo de erros, etc.)
        # Supondo que posprocessDataframe exista em metrics.py e retorne o df modificado
        df = posprocessDataframe(df)
        if df is None: # Se posprocessDataframe puder falhar e retornar None
            print("ERRO CRÍTICO: Falha durante o posprocessDataframe.")
            return None

        print(f"Dados carregados e processados com sucesso. Total de {len(df)} entradas.")
        return df

    except Exception as e_general:
        print(f"ERRO INESPERADO ao processar dados do arquivo {file_path}: {e_general}")
        import traceback
        traceback.print_exc()
        return None
