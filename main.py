# main.py
import os
import json
import numpy as np # Adicionado para np.arange
from processor import process_model
# Importar ambas as funções de reporting.py
from reporting import create_metrics_summary_table, plot_cumulative_metric_graph 
# Se NUM_DAYS_METRICS está definido em reporting.py e você quer usá-lo:
# from reporting import NUM_DAYS_METRICS 

# NOME E CAMINHO FIXOS PARA O ARQUIVO JSON DE CONFIGURAÇÃO
DEFAULT_CONFIG_FILENAME = "job_config.json"
NUM_DAYS_METRICS = 7 # Definir aqui ou importar de reporting.py

def load_config_from_json(json_file_path):
    """Carrega a configuração de um arquivo JSON."""
    try:
        with open(json_file_path, 'r') as f:
            config_data = json.load(f)
        print(f"Configuração carregada com sucesso de: {json_file_path}")
        return config_data
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de configuração JSON '{json_file_path}' não encontrado.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERRO CRÍTICO: Falha ao decodificar o arquivo JSON '{json_file_path}'. Erro: {e}")
        return None
    except Exception as e:
        print(f"ERRO CRÍTICO: Ocorreu um erro inesperado ao carregar o JSON: {e}")
        return None

def main():
    """
    Script principal para processamento em lote de modelos de previsão,
    gerando gráficos de métricas acumuladas e uma tabela de resumo.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, DEFAULT_CONFIG_FILENAME)

    batch_config_data = load_config_from_json(config_file_path)

    if batch_config_data is None or "model_tasks" not in batch_config_data:
        print("ERRO CRÍTICO: Configuração de lote não pôde ser carregada ou 'model_tasks' está ausente no JSON.")
        return

    model_tasks_list = batch_config_data["model_tasks"]

    print(f"\n--- Iniciando Processamento em Lote ---")
    print(f"Total de tarefas definidas no JSON: {len(model_tasks_list)}")

    successful_tasks_count = 0
    failed_tasks_count = 0
    skipped_tasks_count = 0
    all_task_metrics_results = [] # Lista para armazenar os resultados das métricas

    # Diretório para salvar relatórios finais (gráficos e tabela)
    reports_output_dir = os.path.join(script_dir, "relatorios_finais_batch") # Nome do diretório de relatórios
    os.makedirs(reports_output_dir, exist_ok=True) 

    for i, task_config in enumerate(model_tasks_list):
        task_id = task_config.get("task_id", f"Tarefa_NaoIdentificada_{i+1}")
        print(f"\n--- Avaliando Tarefa [{i+1}/{len(model_tasks_list)}]: {task_id} ---")

        if not task_config.get("enabled", True):
            print(f"  Tarefa '{task_id}' está desabilitada. Pulando.")
            skipped_tasks_count += 1
            continue
        
        print(f"  Tarefa '{task_id}' está habilitada. Iniciando processamento...")

        required_keys = ["model_type", "model_file", "output_directory", "visualization_pos"]
        missing_keys = [key for key in required_keys if key not in task_config]

        if missing_keys:
            print(f"  ERRO na Tarefa '{task_id}': Chaves obrigatórias ausentes: {', '.join(missing_keys)}. Pulando tarefa.")
            failed_tasks_count += 1
            continue

        current_model_type = task_config["model_type"]
        current_model_file = task_config["model_file"]
        # output_directory do JSON é para os resultados da tarefa individual
        task_specific_output_dir = task_config["output_directory"] 
        current_viz_pos = task_config["visualization_pos"]

        print(f"  Tipo de Modelo: {current_model_type}")
        print(f"  Arquivo do Modelo: {current_model_file}")
        print(f"  Diretório de Saída da Tarefa: {task_specific_output_dir}")
        print(f"  Posição para Visualização: {current_viz_pos}")

        if not os.path.exists(current_model_file):
            print(f"  ERRO: Arquivo de modelo '{current_model_file}' não encontrado. Pulando tarefa '{task_id}'.")
            failed_tasks_count += 1
            continue

        try:
            os.makedirs(task_specific_output_dir, exist_ok=True) # Cria o diretório de saída da tarefa
            
            task_metrics = process_model(
                current_model_type,
                current_model_file,
                task_specific_output_dir, # Passa o diretório da tarefa para process_model
                current_viz_pos
            )

            if task_metrics and isinstance(task_metrics, dict):
                all_task_metrics_results.append({
                    "task_id": task_id,
                    "model_type": current_model_type,
                    "metrics_data": task_metrics # Deve ser {'rmse': [d1..d7], 'r2': [d1..d7], ...}
                })
                print(f"  Tarefa '{task_id}' processada com sucesso.")
                successful_tasks_count += 1
            else:
                print(f"  AVISO: Tarefa '{task_id}' concluída, mas não retornou métricas válidas ou no formato esperado.")
                failed_tasks_count +=1 

        except Exception as e:
            print(f"  ERRO INESPERADO durante o processamento da tarefa '{task_id}': {e}")
            import traceback
            traceback.print_exc()
            failed_tasks_count += 1
    # --- FIM DO LOOP DE PROCESSAMENTO DAS TAREFAS ---

    print(f"\n--- Resumo do Processamento em Lote ---")
    print(f"Total de tarefas configuradas: {len(model_tasks_list)}")
    print(f"Tarefas processadas com sucesso (com métricas): {successful_tasks_count}")
    print(f"Tarefas com falha ou sem métricas: {failed_tasks_count}")
    print(f"Tarefas puladas (desabilitadas): {skipped_tasks_count}")

    # --- GERAÇÃO DOS GRÁFICOS E DA TABELA ---
    if all_task_metrics_results: # Só tenta gerar se houver resultados de métricas
        days_for_plotting = np.arange(1, NUM_DAYS_METRICS + 1) # Eixo X para os gráficos (Dia 1, Dia 2, ...)
        
        print(f"\n--- Gerando Relatórios Finais em: {reports_output_dir} ---")
        
        # 1. Gerar os gráficos de métricas acumuladas
        # Define quais métricas plotar e seus rótulos para os gráficos
        metrics_to_plot_config = {
            "rmse": "RMSE",
            "r2": "R²",
            "mse": "MSE"
        }
        
        generated_graph_paths = []
        for internal_metric_key, display_metric_label in metrics_to_plot_config.items():
            print(f"  Gerando gráfico para {display_metric_label} acumulado...")
            graph_path = plot_cumulative_metric_graph(
                metric_key=internal_metric_key,
                metric_label=display_metric_label,
                all_tasks_metrics_data=all_task_metrics_results,
                days_array=days_for_plotting,
                output_directory=reports_output_dir # Salva gráficos neste diretório
            )
            if graph_path:
                 generated_graph_paths.append(graph_path)
        
        if generated_graph_paths:
            print("\nGráficos de métricas acumuladas gerados com sucesso.")
        else:
            print("\nNenhum gráfico de métrica acumulada foi gerado (verificar dados ou logs).")

        # 2. Gerar a tabela de resumo das métricas
        summary_table_filename = "resumo_metricas_modelos.png" 
        summary_table_path = os.path.join(reports_output_dir, summary_table_filename)
        
        print(f"\nGerando tabela de resumo das métricas em: {summary_table_path}...")
        create_metrics_summary_table(all_task_metrics_results, summary_table_path) # Função do reporting.py
    else:
        print("\nNenhuma métrica foi coletada das tarefas processadas para gerar os relatórios.")

    print(f"\n--- Processamento em Lote e Geração de Relatórios Concluídos ---")

if __name__ == "__main__":
    main()
