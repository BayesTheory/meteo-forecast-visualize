# main.py
import os
import argparse
from processor import process_model

def main():
    """
    Script principal para processamento de modelos de previsão
    usando configuração hard-coded ou argumentos da linha de comando
    """
    # Configuração hard-coded (padrão)
    # MODIFIQUE ESTAS VARIÁVEIS DIRETAMENTE NO CÓDIGO
    config = {
        "model": "FCNN",          # Opções: "FCNN", "LSTM", "GNN"
        "file": "/caminho/para/FCNN_model.pkl",
        "output": "./resultados", 
        "pos": 0                  # Posição para visualização
    }
    
    # Criar parser de argumentos para manter compatibilidade com linha de comando
    parser = argparse.ArgumentParser(description='Processador de Modelos de Previsão')
    parser.add_argument("--model", type=str, choices=['FCNN', 'LSTM', 'GNN'],
                        help='Tipo de modelo (FCNN, LSTM ou GNN)')
    parser.add_argument("--file", type=str, help='Caminho para o arquivo .pkl do modelo')
    parser.add_argument("--output", type=str, help='Diretório para saída')
    parser.add_argument("--pos", type=int, help='Posição (dia) para visualização')
    parser.add_argument("--use_cmd", action="store_true", 
                        help='Use argumentos da linha de comando em vez da configuração no código')
    
    args = parser.parse_args()
    
    # Se --use_cmd for especificado, use os argumentos da linha de comando
    # Caso contrário, use a configuração hard-coded
    if args.use_cmd:
        # Verificar argumentos obrigatórios na linha de comando
        if not args.model or not args.file:
            parser.error("Ao usar --use_cmd, os argumentos --model e --file são obrigatórios")
        
        # Pegar valores da linha de comando
        model = args.model
        file_path = args.file
        output_dir = args.output or config["output"]
        pos = args.pos if args.pos is not None else config["pos"]
    else:
        # Usar configuração hard-coded
        model = config["model"]
        file_path = config["file"]
        output_dir = config["output"]
        pos = config["pos"]
    
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Processar o modelo
    print(f"Processando modelo {model} do arquivo: {file_path}")
    process_model(model, file_path, output_dir, pos)
    print(f"Processamento concluído. Resultados salvos em {output_dir}")

if __name__ == "__main__":
    main()
