# processor.py
import os
import pandas as pd
import numpy as np
from metrics import posprocessDataframe, calculate_model_metrics
from visualizer import generate_visualizations

def process_model(model_type, file_path, output_dir="./output", pos=0):
    """
    Processa um modelo, calcula métricas e gera visualizações
    
    Parameters:
    -----------
    model_type : str
        Tipo de modelo ('FCNN', 'LSTM', 'GNN')
    file_path : str
        Caminho para o arquivo .pkl
    output_dir : str
        Diretório para salvar as saídas
    pos : int
        Posição (dia) para visualização
    """
    print(f"Iniciando processamento do modelo {model_type}...")
    
    # Carregar e processar dados
    df = load_model_data(file_path, model_type)
    
    # Calcular e exibir métricas
    metrics = calculate_model_metrics(df)
    print("\n===== MÉTRICAS =====")
    for day in range(min(7, len(metrics['rmse']))):
        print(f"Dia {day+1}:")
        print(f"  RMSE: {metrics['rmse'][day]:.4f}")
        print(f"  R²: {metrics['r2'][day]:.4f}")
    
    # Criar subdiretório para o modelo
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Gerar visualizações
    print("\nGerando visualizações...")
    generate_visualizations(df, model_type, model_dir, position=pos)
    
    print(f"Processamento do modelo {model_type} concluído!")
    return metrics

def load_model_data(file_path, model_type):
    """
    Carrega e prepara os dados de um modelo
    
    Parameters:
    -----------
    file_path : str
        Caminho para o arquivo .pkl
    model_type : str
        Tipo de modelo ('FCNN', 'LSTM', 'GNN')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame processado
    """
    print(f"Carregando dados do arquivo: {file_path}")
    
    # Carregar o DataFrame
    df = pd.read_pickle(file_path)
    
    # Remover espaços extras nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Criar a coluna 'data' caso não exista
    if 'data' not in df.columns and 'dia_mes_ano' in df.columns:
        df["data"] = pd.to_datetime(df["dia_mes_ano"]) + pd.Timedelta(hours=12)
    
    try:
        # Reshape para o formato esperado
        df['y_rol'] = df['y_rol'].apply(lambda x: np.array(x).reshape(354*360, 7))
        df['y_rol_pred'] = df['y_rol_pred'].apply(lambda x: np.array(x).reshape(354*360, 7))
    except Exception as e:
        print(f"Erro ao fazer reshape dos dados: {e}")
        print("Verificando formatos disponíveis:")
        print(f"Formato y_rol: {df['y_rol'].iloc[0].shape}")
        print(f"Formato y_rol_pred: {df['y_rol_pred'].iloc[0].shape}")
        raise
    
    # Calcular métricas
    df = posprocessDataframe(df)
    
    print(f"Dados carregados com sucesso. Total de {len(df)} entradas.")
    return df
