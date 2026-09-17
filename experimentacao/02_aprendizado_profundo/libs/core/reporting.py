import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Força o backend para geração de arquivos, sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from core.config import CONFIG
from adjustText import adjust_text


def plot_confusion_matrix(model_name, y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig(os.path.join(output_path, f'matriz_confusao_{model_name}.png'))
    plt.close('all')    
    
def plot_summary_metrics(model_name, output_path, file_name):
    # Carregar o relatório salvo como CSV
    df_report = pd.read_csv(os.path.join(output_path, file_name), index_col=0)
    # 1. Extração correta dos valores escalares
    # Usamos .iloc[0,0] ou acessamos a coluna específica para garantir que venha um NÚMERO
    try:
        total_acc = df_report.loc['accuracy', 'precision'] # A acurácia é repetida nas colunas
        macro_f1 = df_report.loc['macro avg', 'f1-score']
    except Exception as e:
        print(f"Erro ao extrair métricas: {e}")
        return

    metrics = ['Accuracy', 'Macro F1-Score']
    values = [float(total_acc), float(macro_f1)] # Forçamos para float puro
    
    plt.figure(figsize=(8, 5))
    colors = ['#1abc9c', '#3498db']
    
    # Agora o 'values' é uma lista simples de dois números [0.11, 0.07]
    bars = plt.bar(metrics, values, color=colors, width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
                 f'{yval:.2%}', ha='center', fontweight='bold')

    plt.ylim(0, 1.1)
    plt.title(f'Performance Geral - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'summary_{model_name}.png'))
    plt.close('all')
    
def plot_report_metrics(model_name, output_path, file_report):
    # Carregar o relatório salvo como CSV
    df_report = pd.read_csv(os.path.join(output_path, file_report), index_col=0)
    
    # 1. Preparação: Removemos as linhas de médias globais para focar nas classes
    # e removemos a coluna 'support' que tem uma escala diferente
    df_classes = df_report.iloc[:-3, :].drop(columns=['support'])
    df_classes = df_classes.sort_values(by='f1-score', ascending=True)

    # --- GRÁFICO 1: RANKING DE F1-SCORE ---
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(df_classes))) # Gradiente de vermelho para verde
    
    bars = plt.barh(df_classes.index, df_classes['f1-score'], color=colors)
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Meta 80%')
    
    plt.title(f'Ranking de F1-Score por Classe - {model_name}', fontsize=14)
    plt.xlabel('F1-Score')
    plt.xlim(0, 1.0)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'ranking_f1_{model_name}.png'))
    plt.close('all')

    # --- GRÁFICO 2: COMPARATIVO PRECISION VS RECALL ---
    # Ideal para ver se o modelo está "chutando" muito uma classe ou ignorando ela
    df_classes.plot(kind='bar', figsize=(14, 7), color=['#3498db', '#e67e22', '#2ecc71'])
    plt.title(f'Precision, Recall e F1 por Classe - {model_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'comparativo_metricas_{model_name}.png'))
    plt.close('all')

    # --- GRÁFICO 3: HEATMAP DO RELATÓRIO ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_classes, annot=True, cmap='YlGnBu', fmt='.2f', cbar=False)
    plt.title(f'Mapa de Calor de Performance - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'heatmap_report_{model_name}.png'))
    plt.close('all')

    print(f"📊 Gráficos de análise salvos em {output_path}")
    
def plot_comparison_results(results):
    df_results = pd.DataFrame(results)
    # 1. Limpeza de dados (Converter 'Total_Params' de string para numérico)
    df = df_results.copy()
    df['Total_Params_Numeric'] = df['Total_Params'].str.replace(',', '').astype(float)
    
    # Criar pasta para os comparativos
    output_path = CONFIG['OUTPUT_MODEL']
    folder_name_output = os.path.basename(CONFIG['OUTPUT_MODEL'])
    
    # --- GRÁFICO 1: ACURÁCIA VS FPS (O Gráfico de Decisão) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotar cada modelo como um ponto
    texts = []
    for i, row in df.iterrows():
        texts.append(ax.text(row['FPS (RTX 4060)'], row['Acurácia'], row['Modelo'], fontsize=9))
        ax.scatter(row['FPS (RTX 4060)'], row['Acurácia'], s=100)
        ax.annotate(row['Modelo'], (row['FPS (RTX 4060)'], row['Acurácia']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
        
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_title('Trade-off: Acurácia vs. Velocidade de Inferência', fontsize=14)
    ax.set_xlabel('FPS (Frames Per Second) - Mais é melhor →', fontsize=12)
    ax.set_ylabel('Acurácia (Validação) - Mais é melhor ↑', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, folder_name_output+'_comparison_accuracy_vs_fps.png'))
    plt.close('all')

    # --- GRÁFICO 2: TEMPO DE TREINO VS ACURÁCIA ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Ordenar por acurácia para ficar visual
    df_sorted = df.sort_values(by='Acurácia', ascending=True)
    
    ax1.bar(df_sorted['Modelo'], df_sorted['Acurácia'], color='skyblue', alpha=0.7, label='Acurácia')
    ax1.set_ylabel('Acurácia', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Eixo secundário para o tempo
    ax2 = ax1.twinx()
    ax2.plot(df_sorted['Modelo'], df_sorted['Tempo_FT_Seg'], color='red', marker='o', linewidth=2, label='Tempo Treino (s)')
    ax2.set_ylabel('Tempo de Fine-Tuning (Segundos)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Investimento de Treino vs. Resultado Obtido', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'comparison_training_efficiency.png'))
    plt.close('all')

    print(f"📈 Gráficos comparativos gerados com sucesso em: {output_path}")    
    
    
def plot_learning_curves(df_history, model_name, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Gráfico de Acurácia ---
    ax1.plot(df_history['accuracy'], label='Treino')
    ax1.plot(df_history['val_accuracy'], label='Validação')
    ax1.set_title(f'Acurácia - {model_name}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()

    # --- Gráfico de Perda (Loss) ---
    ax2.plot(df_history['loss'], label='Treino')
    ax2.plot(df_history['val_loss'], label='Validação')
    ax2.set_title(f'Perda (Loss) - {model_name}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"learning_curves_{model_name}.png"))
    plt.close('all')   
    
def gerar_graficos_avancados(results):
    df = pd.DataFrame(results)
    output_path = CONFIG['OUTPUT_MODEL']
    # Tratamento de dados: remover vírgulas e converter para float
    df['Total_Params'] = df['Total_Params'].str.replace(',', '').astype(float)
    
    # Novos Indicadores
    df['Latência (ms)'] = 1000 / df['FPS (RTX 4060)']
    df['Eficiência (Acc/Milhão Params)'] = df['Acurácia'] / (df['Total_Params'] / 1_000_000)

    # --- GRÁFICO 1: GRÁFICO DE BOLHAS (TRADE-OFF COMPLETO) ---
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='FPS (RTX 4060)', y='Acurácia', 
                    size='Total_Params', hue='VRAM_Estimada', 
                    sizes=(100, 1000), alpha=0.6, palette='viridis')
    
    for i in range(df.shape[0]):
        plt.text(df['FPS (RTX 4060)'][i]+0.5, df['Acurácia'][i], df['Modelo'][i], fontsize=10)

    plt.title('Performance Multidimensional: Acurácia vs. FPS vs. Parâmetros', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_path, 'comparativo_bolhas_poc.png'))
    plt.close('all')

    # --- GRÁFICO 2: LATÊNCIA EM MILISSEGUNDOS ---
    plt.figure(figsize=(10, 6))
    df_sorted_lat = df.sort_values('Latência (ms)')
    sns.barplot(x='Modelo', y='Latência (ms)', data=df_sorted_lat, palette='magma')
    plt.axhline(y=33.3, color='r', linestyle='--', label='Tempo Real (30 FPS / 33ms)')
    plt.title('Latência de Inferência por Imagem (ms)')
    plt.ylabel('Milissegundos (Quanto menor, melhor)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'latencia_modelos.png'))
    plt.close('all')

    # --- GRÁFICO 3: EFICIÊNCIA DE ARQUITETURA ---
    plt.figure(figsize=(10, 6))
    df_sorted_eff = df.sort_values('Eficiência (Acc/Milhão Params)', ascending=False)
    sns.barplot(x='Modelo', y='Eficiência (Acc/Milhão Params)', data=df_sorted_eff, palette='Greens_r')
    plt.title('Eficiência Arquitetural (Acurácia por Milhão de Parâmetros)')
    plt.savefig(os.path.join(output_path, 'eficiencia_parametros.png'))
    plt.close('all')    