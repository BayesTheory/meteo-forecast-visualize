# metrics.py
import numpy as np
import torch # Usado em posprocessDataframe
from sklearn.metrics import r2_score # Usado em posprocessDataframe

def posprocessDataframe(df):
    """
    Calcula métricas para o DataFrame (MSE, RMSE, R²) por amostra, para cada dia.
    Adiciona colunas 'mse', 'rmse', 'r2_score' ao DataFrame, onde cada célula
    dessas colunas conterá um array de 7 valores (um para cada dia).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com colunas 'y_rol' e 'y_rol_pred' (arrays (N,7))
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame com métricas adicionadas.
    """
    # Garantir que y_rol e y_rol_pred tenham o mesmo número de "dias" (segunda dimensão)
    # para evitar erros no cálculo das métricas.
    # Assume que a primeira linha é representativa da estrutura.
    if not df.empty and 'y_rol' in df.columns and 'y_rol_pred' in df.columns and \
       isinstance(df['y_rol'].iloc[0], np.ndarray) and \
       isinstance(df['y_rol_pred'].iloc[0], np.ndarray) and \
       df['y_rol'].iloc[0].ndim == 2 and df['y_rol_pred'].iloc[0].ndim == 2:
        
        min_shape_days = min(df['y_rol'].iloc[0].shape[1], df['y_rol_pred'].iloc[0].shape[1])
        
        df['y_rol'] = df['y_rol'].apply(lambda row_array: row_array[:, :min_shape_days])
        df['y_rol_pred'] = df['y_rol_pred'].apply(lambda row_array: row_array[:, :min_shape_days])
    else:
        print("AVISO em posprocessDataframe: Colunas 'y_rol' ou 'y_rol_pred' ausentes, vazias ou com formato inesperado. Métricas podem não ser calculadas.")
        # Você pode querer retornar o df original ou None se isso for um erro crítico.
        # Por ora, prosseguirá, mas os cálculos abaixo podem falhar.
        min_shape_days = 7 # Um fallback, mas pode não ser ideal.

    # Calcular MSE por dia, para cada amostra
    # torch.mean(..., dim=0) calcula a média ao longo dos pontos da grade (N), resultando em 7 valores de MSE.
    df['mse'] = df.apply(
        lambda row: torch.mean((torch.tensor(row['y_rol'], dtype=torch.float32) - torch.tensor(row['y_rol_pred'], dtype=torch.float32)) ** 2, dim=0).numpy(),
        axis=1
    )
    
    # Calcular RMSE por dia, para cada amostra
    df['rmse'] = df['mse'].apply(lambda mse_array: np.sqrt(mse_array))
    
    # Calcular R² por dia, para cada amostra
    # r2_score(y_true_col, y_pred_col) é aplicado para cada uma das 7 colunas (dias).
    df['r2_score'] = df.apply(
        lambda row: np.array([
            r2_score(row['y_rol'][:, i], row['y_rol_pred'][:, i]) 
            for i in range(min_shape_days) # Usa min_shape_days para segurança
        ]),
        axis=1
    )
    
    return df

def calculate_model_metrics(df):
    """
    Extrai e resume métricas do DataFrame (que já foi processado por posprocessDataframe).
    Calcula a média das métricas diárias sobre todas as amostras do DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com colunas de métricas calculadas (mse, rmse, r2_score),
        onde cada célula dessas colunas é um array de 7 valores diários.
        
    Returns:
    --------
    dict
        Dicionário com métricas resumidas (média dos 7 dias sobre todas as amostras).
        Ex: {'rmse': [rmse_d1, rmse_d2, ..., rmse_d7], 'r2': [...], ...}
    """
    metrics_summary = {}
    
    # Métricas a serem agregadas
    metrics_to_aggregate = ['mse', 'rmse', 'r2_score']
    
    for metric_col_name in metrics_to_aggregate:
        if metric_col_name in df.columns and not df[metric_col_name].empty:
            try:
                # df[metric_col_name].to_numpy() retorna um array de objetos (cada objeto é um array de 7 métricas)
                # np.stack(...) converte isso em um array 2D (n_amostras, 7_dias)
                stacked_metric_arrays = np.stack(df[metric_col_name].to_numpy())
                
                # np.mean(..., axis=0) calcula a média ao longo das amostras, resultando em 7 valores (um por dia)
                mean_daily_metrics = np.mean(stacked_metric_arrays, axis=0)
                
                # Armazena no dicionário de resumo, removendo '_score' de 'r2_score' para a chave
                summary_key = metric_col_name.replace('_score', '')
                metrics_summary[summary_key] = mean_daily_metrics
            except Exception as e:
                print(f"AVISO em calculate_model_metrics: Erro ao processar a coluna de métrica '{metric_col_name}': {e}")
                # Preenche com NaNs ou array de zeros se falhar, para manter a estrutura do dict
                summary_key = metric_col_name.replace('_score', '')
                # Tenta obter o número de dias de uma métrica bem-sucedida, ou usa 7 como padrão
                num_days_fallback = len(next(iter(metrics_summary.values()))) if metrics_summary else 7
                metrics_summary[summary_key] = np.full(num_days_fallback, np.nan)
        else:
            print(f"AVISO em calculate_model_metrics: Coluna de métrica '{metric_col_name}' não encontrada ou vazia no DataFrame.")
            summary_key = metric_col_name.replace('_score', '')
            num_days_fallback = len(next(iter(metrics_summary.values()))) if metrics_summary else 7
            metrics_summary[summary_key] = np.full(num_days_fallback, np.nan)
            
    return metrics_summary

# --- Funções Utilitárias (não diretamente no fluxo principal, mas podem ser úteis) ---
def calculate_mape(y_true, y_pred):
    """
    Calcula o MAPE (Mean Absolute Percentage Error)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true != 0) # Evita divisão por zero
    if not np.any(mask): # Se todos y_true são zero
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_magnitude(vel):
    """
    Calcula a magnitude de um vetor 2D (ou uma série deles).
    vel pode ser (2,) ou (N, 2)
    """
    vel = np.array(vel)
    if vel.ndim == 1 and vel.shape[0] == 2: # Vetor único
        return np.sqrt(vel[0]**2 + vel[1]**2)
    elif vel.ndim == 2 and vel.shape[1] == 2: # Array de vetores
        return np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
    else:
        raise ValueError("Entrada 'vel' deve ser um array 2D (u,v) ou um array de N vetores 2D (N,2).")

def add_noise(tensor, noise_level=0.001):
    """
    Adiciona ruído gaussiano a um tensor numpy.
    """
    noise = np.random.normal(0, noise_level, tensor.shape)
    return tensor + noise
