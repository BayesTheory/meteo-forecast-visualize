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

NUM_DAYS_METRICS = 7 # Número de dias para os quais as métricas são calculadas

# FUNÇÃO PARA PLOTAR MÉTRICAS DIÁRIAS (MODIFICADA)
def plot_daily_metric_for_model(daily_metric_values, 
                                metric_name_display, 
                                metric_key_filename, 
                                model_type_label, 
                                output_dir): # Removido position_index dos parâmetros
    """
    Plota e salva um gráfico da métrica diária para um modelo.

    Parameters:
    -----------
    daily_metric_values : np.array or list
        Array/lista com os valores diários da métrica (ex: 7 valores).
    metric_name_display : str
        Nome da métrica para o título e rótulo do eixo Y (ex: "RMSE").
    metric_key_filename : str
        Chave da métrica para usar no nome do arquivo (ex: "rmse").
    model_type_label : str
        Rótulo do tipo de modelo para o título e nome do arquivo (ex: "FCNN").
    output_dir : str
        Diretório para salvar o gráfico.
    """
    if daily_metric_values is None:
        print(f"  Aviso: Valores da métrica '{metric_name_display}' são None. Gráfico não gerado.")
        return
    
    daily_metric_values_np = np.array(daily_metric_values)
    if len(daily_metric_values_np) < NUM_DAYS_METRICS:
        print(f"  Aviso: Dados insuficientes para a métrica '{metric_name_display}' (esperado {NUM_DAYS_METRICS} dias, obtido {len(daily_metric_values_np)}). Gráfico não gerado.")
        return
    
    values_to_plot = daily_metric_values_np[:NUM_DAYS_METRICS]
    days_array = np.arange(1, NUM_DAYS_METRICS + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    plt.plot(days_array, values_to_plot, marker='o', linestyle='-', 
             linewidth=2, markersize=8, color='dodgerblue')
    
    # Título modificado - sem referência à amostra/posição
    plt.title(f'{metric_name_display} Diário - Modelo {model_type_label}', fontsize=15, weight='bold')
    plt.xlabel('Dia da Previsão', fontsize=12)
    plt.ylabel(f'{metric_name_display}', fontsize=12)
    plt.xticks(days_array)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Nome do arquivo modificado - sem _pos{position_index}
    filename = f"{model_type_label}_daily_{metric_key_filename}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        plt.savefig(filepath, dpi=150)
        print(f"    Gráfico de {metric_name_display} diário salvo em: {filepath}")
    except Exception as e:
        print(f"    Erro ao salvar o gráfico de {metric_name_display} diário: {e}")
    finally:
        plt.close()


# FUNÇÃO PRINCIPAL MODIFICADA
def generate_visualizations(df_metrics_and_data, model_type, output_dir, day_for_main_viz=0):
    """
    Gera todas as visualizações para um modelo, incluindo gráficos de métricas diárias.
    O DataFrame de entrada agora é esperado como uma única linha (ou a linha relevante já selecionada)
    contendo os arrays de métricas diárias e os dados de visualização.

    Parameters:
    -----------
    df_metrics_and_data : pandas.Series or pandas.DataFrame (com uma única linha)
        Série ou DataFrame de uma linha contendo os dados para visualização
        e as colunas de métricas diárias (ex: 'rmse', 'mse', 'r2_score'),
        onde cada célula dessas colunas é um array de NUM_DAYS_METRICS valores.
        Também deve conter 'y_rol', 'y_rol_pred', 'lat', 'lon', 'data'.
    model_type : str
        Tipo do modelo ('FCNN', 'LSTM', 'GNN')
    output_dir : str
        Diretório específico da tarefa para salvar as visualizações.
    day_for_main_viz : int
        Dia específico (0 a 6) a ser destacado nas visualizações principais (GIFs, grid).
        Os gráficos de métricas diárias sempre mostrarão todos os 7 dias.
    """
    # O 'position' do JSON agora é 'day_for_main_viz' e se refere ao dia a ser
    # destacado nos GIFs e grids, não a uma linha de múltiplas amostras.
    # Assumimos que df_metrics_and_data JÁ É a amostra única a ser processada
    # ou que o processor.py já selecionou a linha correta se df fosse multi-linhas.
    # Para simplificar aqui, vamos assumir que df_metrics_and_data é uma pandas.Series
    # representando a única amostra (ou a linha já selecionada pelo processor.py).

    print(f"\n  Iniciando geração de visualizações para {model_type}, destacando dia {day_for_main_viz + 1}, em: {output_dir}")

    if df_metrics_and_data is None or df_metrics_and_data.empty:
        print("    Dados de entrada (df_metrics_and_data) vazios ou não fornecidos. Nenhuma visualização será gerada.")
        return
    
    # 'df_metrics_and_data' agora é a nossa 'sample_data'
    sample_data = df_metrics_and_data

    # --- GERAÇÃO DOS GRÁFICOS DE MÉTRICAS DIÁRIAS ---
    # Estes gráficos mostram a evolução das métricas ao longo dos 7 dias para esta execução do modelo.
    print("    Gerando gráficos de métricas diárias (RMSE, MSE, R²)...")
    metrics_to_plot_config = {
        "rmse": "RMSE",
        "mse": "MSE",
        "r2_score": "R²"
    }

    for metric_key_in_df, metric_display_label in metrics_to_plot_config.items():
        if metric_key_in_df in sample_data and sample_data[metric_key_in_df] is not None:
            daily_values = sample_data[metric_key_in_df]
            
            plot_daily_metric_for_model(
                daily_metric_values=daily_values,
                metric_name_display=metric_display_label,
                metric_key_filename=metric_key_in_df.replace('_score', ''),
                model_type_label=model_type,
                output_dir=output_dir
                # position_index não é mais necessário aqui
            )
        else:
            print(f"    Aviso: Coluna de métrica '{metric_key_in_df}' não encontrada ou vazia nos dados.")
    
    # --- VISUALIZAÇÕES EXISTENTES (GIFs, Grid) ---
    # Estas visualizações podem usar 'day_for_main_viz' para destacar um dia específico.
    
    # Gerar grid de imagens - esta função plotará 6 dias, usando 'day_for_main_viz' implicitamente se
    # a lógica interna de plot_images_in_grid for sobre frames e não uma posição de amostra.
    # A 'position' passada para plot_images_in_grid aqui é 0 porque estamos tratando 'sample_data' como a única amostra.
    print("    Gerando grid de imagens...")
    required_cols_grid = ['y_rol', 'y_rol_pred', 'lat', 'lon', 'data']
    if all(col in sample_data and sample_data[col] is not None for col in required_cols_grid):
        vmin = np.percentile(sample_data['y_rol'], 5) if sample_data['y_rol'].size > 0 else 0
        vmax = np.percentile(sample_data['y_rol'], 95) if sample_data['y_rol'].size > 0 else 1
        # O nome do arquivo do grid agora só tem o model_type e o 'day_for_main_viz' (antigo 'position')
        grid_path = os.path.join(output_dir, f"{model_type}_grid_dia{day_for_main_viz + 1}.png")
        # A função plot_images_in_grid precisa ser ciente de que 'pos' (seu 3º argumento)
        # deve ser usado em conjunto com day_for_main_viz se ela opera sobre um DataFrame multi-linhas,
        # ou simplesmente usa os dados de sample_data e plota frames/dias.
        # Para este contexto, passamos df=sample_data.to_frame().T para simular um df de 1 linha, e pos=0.
        # A lógica de 'rows' em plot_images_in_grid define quantos dias plotar.
        plot_images_in_grid(sample_data.to_frame().T, rows=NUM_DAYS_METRICS, cols=3, pos=0, 
                            prefix=model_type, output_path=grid_path, vmin=vmin, vmax=vmax, 
                            day_to_highlight=day_for_main_viz) # Passando o dia a destacar se a função suportar
    else:
        print("    Aviso: Colunas necessárias para plot_images_in_grid ausentes ou dados None. Grid não gerado.")
    
    # Gerar GIFs para diferentes visualizações
    print("    Gerando GIFs...")
    required_cols_gif = ['y_rol', 'y_rol_pred', 'lat', 'lon', 'data']
    if all(col in sample_data and sample_data[col] is not None for col in required_cols_gif):
        for prefix_gif in ['', '_diff', '_pred']:
            col_for_gif = f'y_rol{prefix_gif}' if prefix_gif in ['', '_pred'] else 'y_rol'
            if prefix_gif == '_diff' or (col_for_gif in sample_data and sample_data[col_for_gif] is not None):
                # Nome do GIF agora reflete o dia principal da visualização (se aplicável ao GIF)
                gif_base_name = f"{model_type}_dia{day_for_main_viz + 1}{prefix_gif}"
                # Passamos df=sample_data.to_frame().T e pos=0.
                # O day_for_main_viz pode ser usado dentro de get_gif_forecasting se ele precisar focar um frame.
                get_gif_forecasting(sample_data.to_frame().T, os.path.join(output_dir, gif_base_name), 
                                    prefix=prefix_gif, pos=0, hour=24, day_to_highlight=day_for_main_viz)
            else:
                print(f"    Aviso: Coluna {col_for_gif} ausente para GIF com prefixo '{prefix_gif}'. GIF não gerado.")
    else:
         print("    Aviso: Colunas necessárias para get_gif_forecasting ausentes ou dados None. GIFs não gerados.")

    print(f"  Visualizações para {model_type} (destacando dia {day_for_main_viz + 1}) concluídas.")


def get_gif_forecasting(df_single_row, output_path_base, prefix="", pos=0, hour=24, day_to_highlight=0):
    """
    Gera um GIF. df_single_row é um DataFrame com uma única linha.
    'pos' é sempre 0. 'day_to_highlight' pode ser usado para focar um frame.
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    sample_row = df_single_row.iloc[pos] # pos será 0
    
    if sample_row['y_rol'] is not None and sample_row['y_rol'].size > 0:
        vmin = np.percentile(sample_row['y_rol'], 5)
        vmax = np.percentile(sample_row['y_rol'], 95)
    else: 
        vmin, vmax = 0, 1 
        print(f"Aviso em get_gif_forecasting: y_rol é None ou vazio. Usando vmin/vmax padrão.")

    add_colorbar = [True] 
    
    # Frames para o GIF (todos os dias)
    num_frames_gif = NUM_DAYS_METRICS

    def update(frame_idx): # frame_idx vai de 0 a num_frames_gif - 1
        ax.clear()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
        
        lon = sample_row['lon']
        lat = sample_row['lat']
        
        if lon is None or lat is None:
            ax.set_title(f'Erro: Dados de lon/lat ausentes (Frame {frame_idx+1})')
            return
        shape = lon.shape

        scatter = None
        data_to_plot = None
        
        if prefix == '_diff':
            if sample_row['y_rol_pred'] is not None and sample_row['y_rol'] is not None and \
               sample_row['y_rol_pred'].shape[1] > frame_idx and sample_row['y_rol'].shape[1] > frame_idx:
                y_pred = sample_row['y_rol_pred'][:, frame_idx]
                y_real = sample_row['y_rol'][:, frame_idx]
                data_to_plot = np.abs(y_real - y_pred).reshape(shape)
                current_cmap = 'coolwarm'
                current_vmin, current_vmax = -2, 2 # Para diferença
            else:
                ax.set_title(f'Erro: Dados ausentes para diff (Frame {frame_idx+1})')
                return
        else:
            col_name = f'y_rol{prefix}' 
            if col_name in sample_row and sample_row[col_name] is not None and sample_row[col_name].shape[1] > frame_idx:
                data_to_plot = sample_row[col_name][:, frame_idx].reshape(shape)
                current_cmap = 'jet'
                current_vmin, current_vmax = vmin, vmax
            else:
                ax.set_title(f'Erro: Dados ausentes para {col_name} (Frame {frame_idx+1})')
                return
        
        if data_to_plot is not None:
             scatter = ax.pcolormesh(lon, lat, data_to_plot, cmap=current_cmap, vmin=current_vmin, vmax=current_vmax, shading='auto', transform=ccrs.PlateCarree())
        
        if add_colorbar[0] and scatter:
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
            cbar.set_label('Intensidade')
            add_colorbar[0] = False # Adiciona colorbar apenas uma vez
        
        date_val = sample_row['data'] # Data base
        if date_val is not None:
            current_frame_date = date_val + timedelta(hours=hour * frame_idx)
            current_frame_date_str = current_frame_date.strftime("%Y-%m-%d %H:%M:%S")
            title_prefix_str = 'Diferença Abs.' if prefix == '_diff' else ('Previsão' if prefix == '_pred' else 'Real')
            
            # Destacar o 'day_to_highlight' (0 a 6) no título do frame correspondente
            title_highlight = " (Dia Principal)" if frame_idx == day_to_highlight else ""
            ax.set_title(f'{title_prefix_str}, Dia {frame_idx+1}{title_highlight}\n{current_frame_date_str}', fontsize=10)
        else:
            ax.set_title(f'Erro: Data base ausente (Frame {frame_idx+1})')
            
    anim = FuncAnimation(fig, update, frames=num_frames_gif, interval=300)
    
    gif_file = f"{output_path_base}_{hour}h.gif"
    try:
        anim.save(gif_file, writer='pillow', fps=2)
        print(f"    GIF salvo: {gif_file}")
    except Exception as e:
        print(f"    Erro ao salvar GIF {gif_file}: {e}")
    finally:
        plt.close(fig)
    return gif_file


def plot_images_in_grid(df_single_row, rows, cols, pos, prefix, output_path, vmin=None, vmax=None, hour=24, day_to_highlight=0):
    """
    Plota uma grade de imagens. 'rows' é o número de dias a mostrar.
    df_single_row é um DataFrame de uma linha. 'pos' é sempre 0.
    'day_to_highlight' indica o dia a ser destacado.
    """
    # 'rows' aqui é o número de dias que você quer mostrar no grid.
    # Se rows > NUM_DAYS_METRICS, limitaremos aos dias disponíveis.
    num_days_to_plot_in_grid = min(rows, NUM_DAYS_METRICS)

    fig, axes = plt.subplots(
        num_days_to_plot_in_grid, cols, figsize=(cols * 5, num_days_to_plot_in_grid * 4.5), # Ajustado figsize
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    fig.subplots_adjust(wspace=-0.6 if cols > 1 else 0, hspace=0.4 if num_days_to_plot_in_grid > 1 else 0) # Ajustado hspace

    sample_row = df_single_row.iloc[pos] # pos é 0
    lon = sample_row['lon']
    lat = sample_row['lat']
    
    if lon is None or lat is None:
        print(f"    Aviso: Dados de lon/lat ausentes para plot_images_in_grid. Grid não gerado.")
        plt.close(fig)
        return None
    shape = lon.shape

    if vmin is None or vmax is None:
        if sample_row['y_rol'] is not None and sample_row['y_rol'].size > 0:
            vmin_calc = np.percentile(sample_row['y_rol'], 5)
            vmax_calc = np.percentile(sample_row['y_rol'], 95)
        else: vmin_calc, vmax_calc = 0, 1
        vmin = vmin if vmin is not None else vmin_calc
        vmax = vmax if vmax is not None else vmax_calc

    pcm_for_colorbar = None
    for i_day in range(num_days_to_plot_in_grid): # i_day é o índice do dia (0 a NUM_DAYS_METRICS-1)
        for j_type in range(cols): # j_type é o tipo de plot (0:Real, 1:Pred, 2:Diff)
            # Ajustar acesso a 'axes' para o caso de 1 linha ou 1 coluna
            if num_days_to_plot_in_grid == 1 and cols == 1: ax = axes
            elif num_days_to_plot_in_grid == 1: ax = axes[j_type]
            elif cols == 1: ax = axes[i_day]
            else: ax = axes[i_day, j_type]

            ax.clear()
            ax.coastlines(linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
            
            data_to_plot = None
            title_part_str = ""

            if j_type == 0:
                title_part_str = "Real"
                if sample_row['y_rol'] is not None and sample_row['y_rol'].shape[1] > i_day:
                    data_to_plot = sample_row['y_rol'][:, i_day].reshape(shape)
            elif j_type == 1:
                title_part_str = "Previsão"
                if sample_row['y_rol_pred'] is not None and sample_row['y_rol_pred'].shape[1] > i_day:
                    data_to_plot = sample_row['y_rol_pred'][:, i_day].reshape(shape)
            elif j_type == 2:
                title_part_str = "Diferença Abs."
                if sample_row['y_rol'] is not None and sample_row['y_rol_pred'] is not None and \
                   sample_row['y_rol'].shape[1] > i_day and sample_row['y_rol_pred'].shape[1] > i_day:
                    data_to_plot = np.abs(sample_row['y_rol'][:, i_day] - sample_row['y_rol_pred'][:, i_day]).reshape(shape)
            
            if data_to_plot is not None:
                current_vmin_plot, current_vmax_plot = (-2, 2) if j_type == 2 else (vmin, vmax)
                current_cmap_plot = 'coolwarm' if j_type == 2 else 'jet'
                
                pcm = ax.pcolormesh(lon, lat, data_to_plot, vmin=current_vmin_plot, vmax=current_vmax_plot, cmap=current_cmap_plot, shading='auto', transform=ccrs.PlateCarree())
                if j_type != 2: pcm_for_colorbar = pcm # Usar Real ou Pred para a colorbar principal
            else:
                ax.text(0.5, 0.5, 'Dados Indisp.', ha='center', va='center', transform=ax.transAxes, fontsize=8)

            date_val = sample_row['data'] # Data base
            if date_val is not None:
                current_plot_date = date_val + timedelta(hours=hour * i_day)
                current_plot_date_str = current_plot_date.strftime("%Y-%m-%d %H:%M")
                title_highlight_str = " (Dia Destaque)" if i_day == day_to_highlight else ""
                ax.set_title(f'{title_part_str}, Dia {i_day+1}{title_highlight_str}\n{current_plot_date_str}', fontsize=9)
            else:
                ax.set_title(f'{title_part_str}, Dia {i_day+1}{title_highlight_str}\nData N/A', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    if pcm_for_colorbar:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) # Posição e tamanho da colorbar
        cbar = fig.colorbar(pcm_for_colorbar, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Intensidade', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    
    # Título principal do Grid
    fig.suptitle(f'Comparativo Diário - {prefix} (Dia Destaque: {day_to_highlight+1})', fontsize=16, weight='bold', y=0.99)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"    Grid de imagens salvo: {output_path}")
    return output_path

# combine_gifs (sem alterações, apenas incluído para completude do arquivo)
def combine_gifs(pattern, output_path, duration_ms=1000):
    all_gifs = sorted(glob.glob(pattern))
    if not all_gifs:
        print(f"Nenhum GIF encontrado com padrão: {pattern}")
        return None
    all_frames = []
    for gif_file in all_gifs:
        try:
            with Image.open(gif_file) as frames:
                frame_idx = 0
                while True:
                    try:
                        frames.seek(frame_idx)
                        frame = frames.copy().convert("RGB")
                        all_frames.append(frame)
                        frame_idx += 1
                    except EOFError:
                        break 
        except Exception as e:
            print(f"Erro ao processar o GIF {gif_file}: {e}")
            continue 
    if not all_frames:
        print("Nenhum frame coletado dos GIFs.")
        return None
    min_width = min(f.width for f in all_frames)
    min_height = min(f.height for f in all_frames)
    min_size = (min_width, min_height)
    all_frames = [f.resize(min_size, Image.Resampling.LANCZOS) for f in all_frames]
    all_frames[0].save(
        output_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"GIF combinado salvo: {output_path}")
    return output_path

