# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timedelta
from matplotlib.animation import FuncAnimation
from PIL import Image
import glob
import os

def generate_visualizations(df, model_type, output_dir, position=0):
    """
    Gera todas as visualizações para um modelo
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame processado
    model_type : str
        Tipo do modelo ('FCNN', 'LSTM', 'GNN')
    output_dir : str
        Diretório para salvar as saídas
    position : int
        Posição (dia) para visualização
    """
    # Gerar grid de imagens
    print("Gerando grid de imagens...")
    vmin = np.percentile(df['y_rol'].iloc[position], 5)
    vmax = np.percentile(df['y_rol'].iloc[position], 95)
    grid_path = os.path.join(output_dir, f"{model_type}_grid_pos{position}.png")
    plot_images_in_grid(df, 6, 3, position, prefix=model_type, output_path=grid_path, vmin=vmin, vmax=vmax)
    
    # Gerar GIFs para diferentes visualizações
    print("Gerando GIFs...")
    for prefix in ['', '_diff', '_pred']:
        gif_path = os.path.join(output_dir, f"{model_type}_pos{position}{prefix}")
        get_gif_forecasting(df, gif_path, prefix=prefix, pos=position, hour=24)

def get_gif_forecasting(df, output_path, prefix="", pos=0, hour=24):
    """
    Gera um GIF para visualização da previsão
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com dados processados
    output_path : str
        Caminho base para salvar o GIF
    prefix : str
        Prefixo para o nome do arquivo ('', '_diff', '_pred')
    pos : int
        Posição no DataFrame
    hour : int
        Intervalo de horas entre frames
    """
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Calcular vmin/vmax para normalização
    vmin = np.percentile(df['y_rol'].iloc[pos], 5)
    vmax = np.percentile(df['y_rol'].iloc[pos], 95)
    
    # Flag para adicionar barra de cores apenas uma vez
    add_colorbar = [True]  # Usando lista para manter estado entre chamadas
    
    def update(frame):
        ax.clear()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND)
        
        # Obter coordenadas
        lon = df['lat'].iloc[pos].data
        lat = df['lon'].iloc[pos].data
        shape = lon.shape
        
        # Preparar dados conforme o tipo de visualização
        if prefix == '_diff':
            y_pred = df['y_rol_pred'].iloc[pos][:, frame]
            y_real = df['y_rol'].iloc[pos][:, frame]
            y_diff = y_real - y_pred
            data = np.abs(y_diff).reshape(shape)
            scatter = ax.pcolormesh(lat, lon, data, vmin=-2, vmax=2, cmap='jet', shading='auto')
        else:
            y_ = df[f'y_rol{prefix}'].iloc[pos][:, frame]
            data = y_.reshape(shape)
            scatter = ax.pcolormesh(lat, lon, data, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        
        # Adicionar barra de cores apenas uma vez
        if add_colorbar[0]:
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label('Intensity')
            add_colorbar[0] = False
        
        # Configurar título com data
        date = df['data'].iloc[pos]
        nova_data = date + timedelta(hours=hour*frame)
        nova_data_str = nova_data.strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(f'Intensity {prefix}, date: {nova_data_str}')
    
    # Criar animação
    anim = FuncAnimation(fig, update, frames=6, interval=200)
    
    # Salvar GIF
    gif_file = f"{output_path}_{hour}h.gif"
    anim.save(gif_file, writer='pillow', fps=1)
    plt.close()
    
    print(f"GIF salvo: {gif_file}")
    return gif_file

def plot_images_in_grid(df, rows, cols, pos, prefix, output_path, vmin=None, vmax=None, hour=24):
    """
    Plota uma grade de imagens para diferentes dias de previsão
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame processado
    rows : int
        Número de linhas (dias) a mostrar
    cols : int
        Número de colunas (tipos de visualização)
    pos : int
        Posição no DataFrame
    prefix : str
        Prefixo para o título
    output_path : str
        Caminho para salvar a imagem
    vmin, vmax : float
        Valores mínimo e máximo para a escala de cores
    hour : int
        Intervalo de horas entre imagens
    """
    # Criar figura
    fig, axes = plt.subplots(
        rows, cols, figsize=(30, 25),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Ajustar espaçamento
    fig.subplots_adjust(wspace=-0.8, hspace=0.5)
    
    # Obter dados de coordenadas
    shape = df['lat'].iloc[pos].shape
    lon = df['lat'].iloc[pos].data
    lat = df['lon'].iloc[pos].data
    
    # Definir vmin/vmax se não fornecidos
    if vmin is None:
        vmin = np.percentile(df['y_rol'].iloc[pos], 5)
    if vmax is None:
        vmax = np.percentile(df['y_rol'].iloc[pos], 95)
    
    # Plotar cada célula da grade
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.clear()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            
            # Plotar dados conforme a coluna
            if j == 0:  # Verdade
                y_ = df['y_rol'].iloc[pos][:, i].reshape(shape)
                pcm = ax.pcolormesh(lon, lat, y_, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
            elif j == 1:  # Previsão
                y_ = df['y_rol_pred'].iloc[pos][:, i].reshape(shape)
                pcm = ax.pcolormesh(lon, lat, y_, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
            elif j == 2:  # Diferença
                y_pred = df['y_rol_pred'].iloc[pos][:, i]
                y_real = df['y_rol'].iloc[pos][:, i]
                y_diff = (y_real - y_pred).reshape(shape)
                pcm = ax.pcolormesh(lon, lat, np.abs(y_diff), vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
            
            # Configurar título
            date = df['data'].iloc[pos]
            nova_data = date + timedelta(hours=hour * i)
            nova_data_str = nova_data.strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(f'Intensity {prefix}, Date: {nova_data_str}', fontsize=8)
            
            # Remover ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Adicionar barra de cores
    cbar_ax = fig.add_axes([0.72, 0.15, 0.02, 0.73])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Intensity')
    
    # Salvar figura
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Grid de imagens salvo: {output_path}")
    return output_path

def combine_gifs(pattern, output_path, duration_ms=1000):
    """
    Combina múltiplos GIFs em um único arquivo
    
    Parameters:
    -----------
    pattern : str
        Padrão glob para encontrar GIFs a combinar
    output_path : str
        Caminho para salvar o GIF combinado
    duration_ms : int
        Duração em milissegundos de cada frame
    """
    all_gifs = sorted(glob.glob(pattern))
    
    if not all_gifs:
        print(f"Nenhum GIF encontrado com padrão: {pattern}")
        return None
    
    all_frames = []
    
    # Coletar todos os frames
    for gif_file in all_gifs:
        frames = Image.open(gif_file)
        while True:
            try:
                frame = frames.copy().convert("RGB")
                all_frames.append(frame)
                frames.seek(frames.tell() + 1)
            except EOFError:
                break
    
    # Redimensionar para tamanho comum
    min_size = min((f.size for f in all_frames), key=lambda s: s[0] * s[1])
    all_frames = [f.resize(min_size, Image.Resampling.LANCZOS) for f in all_frames]
    
    # Salvar GIF combinado
    all_frames[0].save(
        output_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=duration_ms,
        loop=0
    )
    
    print(f"GIF combinado salvo: {output_path}")
    return output_path
