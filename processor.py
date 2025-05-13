# processor.py
import os
import pandas as pd
import numpy as np
from metrics import posprocessDataframe, calculate_model_metrics
from visualizer import generate_visualizations # Assumindo que visualizer.py está atualizado

# DEFINIR A CONSTANTE GLOBALMENTE NO TOPO DO ARQUIVO
NUM_DAYS_METRICS = 7 

def process_model(model_type, file_path, output_dir, pos=0):
    """
    Processa um modelo, calcula métricas e gera visualizações.
    'pos' do JSON é usado para selecionar a amostra do df (se houver múltiplas)
    e também como o 'day_for_main_viz' para generate_visualizations.
    """
    print(f"Iniciando processamento do modelo {model_type}...")
    print(f"  Lendo modelo de: {file_path}")
    print(f"  Diretório de saída da tarefa: {output_dir}")
    print(f"  Posição/Dia de destaque para visualização principal: {pos + 1} (índice {pos})")

    df_loaded = load_model_data(file_path, model_type)

    if df_loaded is None or df_loaded.empty:
        print(f"Falha ao carregar/processar dados para o modelo {model_type} do arquivo {file_path}.")
        print(f"Abortando processamento da tarefa para {model_type}.")
        return None 

    aggregated_metrics = calculate_model_metrics(df_loaded) 

    print("\n===== MÉTRICAS AGREGADAS (Média sobre amostras, por dia) =====")
    if aggregated_metrics and isinstance(aggregated_metrics, dict) and \
       'rmse' in aggregated_metrics and 'r2' in aggregated_metrics and \
       len(aggregated_metrics['rmse']) > 0 and len(aggregated_metrics['r2']) > 0:
        for day_idx in range(min(NUM_DAYS_METRICS, len(aggregated_metrics['rmse']))): # Usar NUM_DAYS_METRICS aqui também
            print(f"Dia {day_idx+1}:")
            print(f"  RMSE Médio: {aggregated_metrics['rmse'][day_idx]:.4f}")
            print(f"  R² Médio: {aggregated_metrics['r2'][day_idx]:.4f}")
            if 'mse' in aggregated_metrics and len(aggregated_metrics['mse']) > day_idx:
                 print(f"  MSE Médio: {aggregated_metrics['mse'][day_idx]:.4f}")
    else:
        print("AVISO: Métricas agregadas não foram calculadas corretamente ou estão ausentes.")

    os.makedirs(output_dir, exist_ok=True)

    single_sample_data_for_viz = None
    if not df_loaded.empty:
        # Validar 'pos' contra o número de dias REALMENTE disponíveis após o reshape
        # No entanto, a 'pos' do JSON é para selecionar a linha/amostra, não o dia.
        # A seleção de dia para visualizações principais (GIFs, grid) é day_for_main_viz
        # que também é 'pos'.
        
        # Se pos do JSON é para selecionar a LINHA do df_loaded:
        effective_pos_for_sample_selection = pos 
        if not (0 <= effective_pos_for_sample_selection < len(df_loaded)):
            print(f"  AVISO: Posição de amostra '{effective_pos_for_sample_selection}' é inválida para o DataFrame carregado (tamanho {len(df_loaded)}). Usando a primeira amostra (índice 0).")
            effective_pos_for_sample_selection = 0
        
        single_sample_data_for_viz = df_loaded.iloc[effective_pos_for_sample_selection]
        
        # 'pos' (vindo do JSON) também é o 'day_for_main_viz' (0-6)
        # Garantir que este 'pos' seja válido como um índice de dia para as visualizações.
        # Esta verificação já é feita no visualizer.py, mas pode ser útil aqui também.
        # if not (0 <= pos < NUM_DAYS_METRICS):
        #     print(f"  AVISO: Dia de destaque '{pos}' é inválido. Ajustando para 0.")
        #     pos = 0 
            
    if single_sample_data_for_viz is None or single_sample_data_for_viz.empty:
        print(f"  ERRO: Não foi possível obter dados da amostra para visualização. Visualizações não serão geradas.")
    else:
        print(f"\n  Gerando visualizações para a amostra de índice {effective_pos_for_sample_selection} (dia de destaque para visualizações principais: {pos+1})...")
        try:
            generate_visualizations(single_sample_data_for_viz, model_type, output_dir, day_for_main_viz=pos)
        except Exception as e_vis:
            print(f"  ERRO ao gerar visualizações para {model_type}: {e_vis}")
            import traceback
            traceback.print_exc()

    print(f"\nProcessamento do modelo {model_type} concluído! Resultados em: {output_dir}")
    return aggregated_metrics

def load_model_data(file_path, model_type_info="modelo"):
    """
    Carrega e prepara os dados de um modelo a partir de um arquivo .pkl.
    Adiciona colunas de métricas diárias (rmse, mse, r2_score) ao DataFrame.
    """
    print(f"    Carregando dados para {model_type_info} do arquivo: {file_path}")

    try:
        df = pd.read_pickle(file_path)
        if not isinstance(df, pd.DataFrame):
            print(f"    ERRO CRÍTICO: O arquivo {file_path} não contém um DataFrame pandas.")
            return None
        if df.empty:
            print(f"    AVISO: O DataFrame carregado de {file_path} está vazio.")
            return df 
    except FileNotFoundError:
        print(f"    ERRO CRÍTICO: Arquivo de modelo {file_path} não encontrado.")
        return None
    except Exception as e_pickle:
        print(f"    ERRO CRÍTICO ao ler o arquivo pickle {file_path}: {e_pickle}")
        return None

    try:
        df.columns = df.columns.str.strip()

        if 'data' not in df.columns and 'dia_mes_ano' in df.columns:
            df["data"] = pd.to_datetime(df["dia_mes_ano"]) + pd.Timedelta(hours=12)
        elif 'data' not in df.columns:
            print("    AVISO: Coluna 'data' não encontrada e 'dia_mes_ano' também não presente.")

        required_reshape_cols = ['y_rol', 'y_rol_pred']
        for col_name in required_reshape_cols:
            if col_name not in df.columns:
                print(f"    ERRO CRÍTICO: Coluna obrigatória '{col_name}' não encontrada.")
                return None
            if df[col_name].isnull().all():
                 print(f"    ERRO CRÍTICO: Coluna '{col_name}' contém apenas valores None.")
                 return None
        try:
            # AQUI é onde NUM_DAYS_METRICS é usado
            df['y_rol'] = df['y_rol'].apply(lambda x: np.array(x, dtype=float).reshape(354*360, NUM_DAYS_METRICS) if x is not None else None)
            df['y_rol_pred'] = df['y_rol_pred'].apply(lambda x: np.array(x, dtype=float).reshape(354*360, NUM_DAYS_METRICS) if x is not None else None)
        except Exception as e_reshape:
            print(f"    ERRO CRÍTICO durante o reshape: {e_reshape}")
            if not df.empty:
                first_row = df.iloc[0]
                if 'y_rol' in first_row and first_row['y_rol'] is not None:
                    print(f"      Exemplo de tipo de y_rol[0]: {type(first_row['y_rol'])}, shape: {np.array(first_row['y_rol']).shape if isinstance(first_row['y_rol'], (list, np.ndarray)) else 'N/A'}")
            return None
        
        df.dropna(subset=['y_rol', 'y_rol_pred'], inplace=True)
        if df.empty:
            print(f"    AVISO: DataFrame ficou vazio após remover linhas com falha no reshape/dados ausentes em y_rol/y_rol_pred.")
            return None

        df_with_daily_metrics = posprocessDataframe(df.copy()) 
        
        if df_with_daily_metrics is None or df_with_daily_metrics.empty:
            print("    ERRO CRÍTICO: Falha durante o posprocessDataframe ou resultou em DataFrame vazio.")
            return None

        print(f"    Dados carregados e métricas diárias calculadas. Total de {len(df_with_daily_metrics)} amostras válidas.")
        return df_with_daily_metrics

    except Exception as e_general:
        print(f"    ERRO INESPERADO ao processar dados do arquivo {file_path}: {e_general}")
        import traceback
        traceback.print_exc()
        return None
