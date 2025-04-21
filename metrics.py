# metrics.py
import numpy as np
import torch
from sklearn.metrics import r2_score

def posprocessDataframe(df):
    """
    Calcula métricas para o DataFrame (MSE, RMSE, R²)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com colunas 'y_rol' e 'y_rol_pred'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame com métricas adicionadas
    """
    # Garantir shapes compatíveis
    min_shape = min(df['y_rol'].iloc[0].shape[1], df['y_rol_pred'].iloc[0].shape[1])
    df['y_rol'] = df['y_rol'].apply(lambda row: row[:, :min_shape])
    df['y_rol_pred'] = df['y_rol_pred'].apply(lambda row: row[:, :min_shape])
    
    # Calcular MSE
    df['mse'] = df.apply(
        lambda row: torch.mean((torch.tensor(row['y_rol']) - torch.tensor(row['y_rol_pred'])) ** 2, dim=0).numpy(),
        axis=1
    )
    
    # Calcular RMSE
    df['rmse'] = df['mse'].apply(lambda mse: np.sqrt(mse))
    
    # Calcular R²
    df['r2_score'] = df.apply(
        lambda row: np.array([
            r2_score(row['y_rol'][:, i], row['y_rol_pred'][:, i]) 
            for i in range(min_shape)
        ]),
        axis=1
    )
    
    return df

def calculate_model_metrics(df):
    """
    Extrai e resume métricas do DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com métricas calculadas
        
    Returns:
    --------
    dict
        Dicionário com métricas resumidas
    """
    metrics = {}
    
    # Calcular médias das métricas
    for metric in ['mse', 'rmse', 'r2_score']:
        if metric in df.columns:
            stacked = np.stack(df[metric].to_numpy())
            metrics[metric.replace('_score', '')] = np.mean(stacked, axis=0)
    
    return metrics

def calculate_mape(y_true, y_pred):
    """
    Calcula o MAPE (Mean Absolute Percentage Error)
    
    Parameters:
    -----------
    y_true : numpy.array
        Valores verdadeiros
    y_pred : numpy.array
        Valores previstos
    
    Returns:
    --------
    float
        MAPE calculado
    """
    mask = (y_true != 0)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_magnitude(vel):
    """
    Calcula a magnitude de um vetor 2D
    
    Parameters:
    -----------
    vel : numpy.array
        Vetor 2D com componentes [u, v]
    
    Returns:
    --------
    numpy.array
        Magnitude calculada
    """
    vel = np.array(vel)
    return np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)

def add_noise(tensor, noise_level=0.001):
    """
    Adiciona ruído gaussiano a um tensor
    
    Parameters:
    -----------
    tensor : numpy.array
        Tensor para adicionar ruído
    noise_level : float
        Nível de ruído a adicionar
    
    Returns:
    --------
    numpy.array
        Tensor com ruído adicionado
    """
    noise = np.random.normal(0, noise_level, tensor.shape)
    return tensor + noise
