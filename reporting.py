# reporting.py
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_DAYS_METRICS = 7 # Número de dias para os quais as métricas são calculadas

def plot_cumulative_metric_graph(metric_key, 
                                 metric_label, 
                                 all_tasks_metrics_data, 
                                 days_array, 
                                 output_directory, 
                                 filename_prefix="cumulative_"):
    """
    Plota e salva um gráfico da métrica acumulada por dia para todos os modelos.
    """
    plt.style.use('seaborn-v0_8-pastel') 
    plt.figure(figsize=(12, 7)) 

    num_tasks = len(all_tasks_metrics_data)
    colors = plt.cm.get_cmap('tab10', num_tasks if num_tasks > 0 else 1) 

    plot_successful = False
    for i, task_data in enumerate(all_tasks_metrics_data):
        task_id = task_data.get('task_id', f'Modelo_{i+1}')
        metric_values_dict = task_data.get('metrics_data', {})
        metric_values_for_key = metric_values_dict.get(metric_key)

        if metric_values_for_key is not None and len(metric_values_for_key) >= NUM_DAYS_METRICS:
            daily_values = np.array(metric_values_for_key[:NUM_DAYS_METRICS])
            cumulative_values = np.cumsum(daily_values)
            
            plt.plot(days_array, cumulative_values, marker='o', linestyle='-', 
                     linewidth=2.5, markersize=7, label=task_id, color=colors(i % colors.N))
            plot_successful = True
        else:
            print(f"Aviso: Dados ausentes/insuficientes para '{metric_key}' na tarefa '{task_id}'. Não será plotado no gráfico de {metric_label} acumulado.")

    if not plot_successful:
        print(f"Nenhum dado válido encontrado para plotar o gráfico de {metric_label} acumulado.")
        plt.close()
        return None

    plt.title(f'{metric_label} Acumulado por Dia (Comparativo de Modelos)', fontsize=16, weight='bold', pad=15)
    plt.xlabel('Dia da Previsão', fontsize=14, labelpad=10)
    plt.ylabel(f'{metric_label} Acumulado', fontsize=14, labelpad=10)
    plt.xticks(days_array, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    fig = plt.gcf()
    if num_tasks > 5:
        plt.legend(fontsize=10, title="Modelos", title_fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.subplots_adjust(right=0.80 if num_tasks <= 10 else 0.75)
    else:
        plt.legend(fontsize=11, title="Modelos", title_fontsize='12')

    filename = f'{filename_prefix}{metric_key}.png'
    filepath = os.path.join(output_directory, filename)
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Gráfico de {metric_label} acumulado salvo em: {filepath}")
    except Exception as e:
        print(f"Erro ao salvar o gráfico de {metric_label} acumulado: {e}")
        filepath = None
    finally:
        plt.close()
    return filepath


def create_metrics_summary_table(all_tasks_metrics_data, output_image_path):
    """
    Cria uma tabela resumida TRANSPOSTA, com estilo similar à imagem [1]
    (cabeçalhos horizontais, linhas divisórias proeminentes).
    """
    if not all_tasks_metrics_data:
        print("Nenhum dado de métrica fornecido para gerar a tabela de resumo.")
        return

    column_headers = ["Métrica / Dia"]
    task_ids_for_header = []
    for task_data in all_tasks_metrics_data:
        task_ids_for_header.append(task_data.get("task_id", "N/A_Task"))
    column_headers.extend(task_ids_for_header)

    row_labels_and_data_source = []
    metric_display_names = {"rmse": "RMSE", "r2": "R²", "mse": "MSE"}
    metric_internal_keys = ["rmse", "r2", "mse"]

    row_labels_and_data_source.append(("Model Type", "model_type", None))
    for key in metric_internal_keys:
        display_name = metric_display_names.get(key, key.upper())
        for day_idx in range(NUM_DAYS_METRICS):
            row_labels_and_data_source.append((f"{display_name} Dia {day_idx+1}", key, day_idx))
        row_labels_and_data_source.append((f"{display_name} Média", key, "mean"))

    table_data = []
    for row_label, metric_key_or_type, aggregation_type in row_labels_and_data_source:
        current_row_values = [row_label]
        for task_data in all_tasks_metrics_data:
            if metric_key_or_type == "model_type":
                current_row_values.append(task_data.get("model_type", "N/A"))
            else:
                metrics = task_data.get("metrics_data")
                value_to_append = "N/A"
                if metrics and metric_key_or_type in metrics and \
                   isinstance(metrics[metric_key_or_type], (list, np.ndarray)) and \
                   len(metrics[metric_key_or_type]) >= NUM_DAYS_METRICS:
                    daily_values = metrics[metric_key_or_type][:NUM_DAYS_METRICS]
                    if isinstance(aggregation_type, int):
                        if aggregation_type < len(daily_values):
                            value_to_append = f"{daily_values[aggregation_type]:.4f}"
                    elif aggregation_type == "mean":
                        mean_val = np.mean(daily_values)
                        value_to_append = f"{mean_val:.4f}"
                current_row_values.append(value_to_append)
        table_data.append(current_row_values)

    if not table_data:
        print("Não foi possível preparar dados para a tabela de resumo transposta.")
        return

    num_data_rows = len(table_data)
    num_cols = len(column_headers)

    # --- Ajustes de Tamanho da Figura e Fonte ---
    avg_char_width_for_header = 0.10 
    longest_task_id_len = max(len(tid) for tid in task_ids_for_header) if task_ids_for_header else 10
    
    model_col_width_abs = max(1.6, longest_task_id_len * avg_char_width_for_header) 
    first_col_width_abs = 2.8 # Largura para "Métrica / Dia" (aumentada ligeiramente)

    fig_width = first_col_width_abs + (num_cols - 1) * model_col_width_abs
    fig_width = max(10, fig_width) # Reduzida largura mínima se poucos modelos

    base_fig_height_per_row = 0.40 # Altura por linha (pode aumentar para mais espaço)
    total_fig_height = (num_data_rows + 1) * base_fig_height_per_row + 2.0 # +2.0 para título e margens amplas
    fig_height = max(6, total_fig_height) # Reduzida altura mínima se poucas linhas


    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # --- Criação da Tabela ---
    col_widths_mpl = [first_col_width_abs / fig_width]
    if num_cols > 1:
        for _ in range(num_cols - 1):
            col_widths_mpl.append(model_col_width_abs / fig_width)
    
    table = ax.table(cellText=table_data,
                     colLabels=column_headers,
                     colWidths=col_widths_mpl,
                     loc='center',
                     cellLoc='center' 
                    )

    table.auto_set_font_size(False)
    table.set_fontsize(10) # Fonte base ligeiramente maior

    # --- Estilização Detalhada para replicar Imagem [1] ---
    header_bg_color = 'white' # Fundo branco para cabeçalho como na imagem [1]
    header_font_color = 'black'
    row_label_bg_color = 'white' # Fundo branco para rótulos de linha
    data_font_color = 'black'
    strong_line_color = 'black' # Linhas divisórias pretas
    strong_line_width = 1.5     # Espessura das linhas divisórias
    default_line_width = 0.8    # Espessura das outras bordas (mais finas)
    default_border_color = '#CCCCCC' # Cor para bordas não proeminentes (cinza claro)


    for (r_mpl, c_mpl), cell in table.get_celld().items():
        # Configurar todas as bordas inicialmente de forma mais sutil
        cell.set_edgecolor(default_border_color)
        cell.set_linewidth(default_line_width)

        # Borda inferior proeminente para todas as linhas (cabeçalho e dados)
        # Para replicar o estilo de "caixas" da imagem [1]
        # Acessar a borda inferior diretamente (pode ser versão-dependente do Matplotlib)
        # Tentativa mais robusta: desenhar linhas com ax.hlines depois
        # Por agora, vamos definir a borda inferior da célula como proeminente
        # No entanto, a tabela do matplotlib desenha uma caixa ao redor de cada célula.
        # Para ter apenas linhas horizontais fortes, é melhor usar ax.hlines

        if r_mpl == 0: # Linha de Cabeçalho (Task IDs)
            cell.set_text_props(weight='normal', color=header_font_color, ha='center', va='center', rotation=0) # Fonte normal, não negrito
            cell.set_facecolor(header_bg_color)
            cell.set_fontsize(10) 
            cell.PAD = 0.03 # Padding
        else: # Linhas de Dados
            cell_data_row_index = r_mpl - 1
            # is_mean_row = table_data[cell_data_row_index][0].endswith("Média") # Não usado para estilização de borda agora

            if c_mpl == 0: # Primeira Coluna (Rótulos de Métricas/Dias)
                cell.set_text_props(weight='normal', color=data_font_color, ha='left', va='center') # Fonte normal
                cell.set_facecolor(row_label_bg_color)
                cell.PAD = 0.04 # Padding à esquerda
            else: # Células de dados numéricos
                cell.set_text_props(color=data_font_color, ha='center', va='center')
                cell.set_fontsize(10) 
                # A imagem não mostra negrito para médias, então removeremos:
                # if is_mean_row:
                #     cell.set_text_props(weight='bold') 
                #     cell.set_facecolor('#F3F6FA') 
    
    # Desenhar linhas horizontais fortes APÓS a tabela ser criada
    # Isso sobrepõe as bordas das células, criando o efeito desejado
    
    # Coordenadas y das bordas superiores das linhas (incluindo a linha abaixo do cabeçalho)
    # O objeto table não tem um método fácil para obter as coordenadas y de cada linha
    # de forma programática e robusta para ax.hlines.
    # A estilização da imagem [1] parece ser bordas de células grossas.
    # Vamos tentar definir a borda inferior de cada célula para ser grossa.
    
    for r_mpl in range(num_data_rows + 1): # +1 para incluir a linha de cabeçalho
        for c_mpl_idx in range(num_cols):
            cell = table.get_celld().get((r_mpl, c_mpl_idx))
            if cell:
                # Tentar modificar a borda inferior. Isso é experimental.
                # Matplotlib < 3.5 pode não ter _edges como dict.
                try:
                    # Esta é a forma mais direta se funcionar
                    cell.get_edges()['bottom'].set_linewidth(strong_line_width)
                    cell.get_edges()['bottom'].set_color(strong_line_color)
                    if r_mpl == 0: # Cabeçalho também precisa de borda superior
                         cell.get_edges()['top'].set_linewidth(strong_line_width)
                         cell.get_edges()['top'].set_color(strong_line_color)

                except (AttributeError, TypeError, KeyError):
                    # Fallback: Definir todas as bordas da célula para serem um pouco mais visíveis
                    # e rely on the overall structure. This won't make *only* the bottom edge thick.
                    # Se a linha de cabeçalho ou uma linha de dados, dar uma borda um pouco mais forte.
                    if r_mpl <= num_data_rows : # Aplica a todas as linhas visíveis
                        cell.set_edgecolor(strong_line_color)
                        cell.set_linewidth(1.0) # Um pouco menos que strong_line_width para não ser excessivo

    plt.title("Resumo Comparativo das Métricas dos Modelos", fontsize=16, weight='bold', pad=25)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)

    try:
        plt.savefig(output_image_path, bbox_inches='tight', dpi=200)
        print(f"Tabela de resumo (estilo imagem [1]) salva em: {output_image_path}")
    except Exception as e:
        print(f"Erro ao salvar a imagem da tabela (estilo imagem [1]): {e}")
    finally:
        plt.close(fig)
