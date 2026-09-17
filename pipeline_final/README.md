# Pipeline Final YOLO - Classificador de Fertilizantes

Este diretório contém o pipeline completo para o treinamento e execução em produção do modelo YOLO de classificação de fertilizantes, incluindo o servidor da API e um painel de curadoria ativa.

---

## Estrutura de Diretórios

- `pyproject.toml`: Especificação de dependências do Python e index do CUDA GPU para o UV.
- `config.py`: Classe central `Config` de constantes para parametrização do projeto.
- `modelo/`: Diretório contendo o modelo YOLO executável (`best.pt`).
- `train/`: Diretório com o script `train.py` para treinamento de novos modelos de produção.
- `api/`: Diretório com o código do servidor FastAPI (`main.py`) e o cliente de testes (`test_client.py`).

---

## Como Instalar as Dependências

Este projeto utiliza o gerenciador `uv` para configurar o ambiente virtual do Python e instalar todas as dependências de forma rápida e otimizada (incluindo PyTorch com suporte a GPU CUDA).

1. Abra o console na pasta `pipeline_final/`.
2. Execute o comando:
   ```powershell
   uv sync
   ```
   Isso criará a pasta `.venv` local e instalará todas as bibliotecas necessárias.

---

## Como Executar o Treinamento

O script de treinamento lê os parâmetros da classe `Config` (em `config.py`), consolida o dataset de treino/validação no diretório de produção e inicia o ajuste fino do YOLO. Ao finalizar, copia o melhor peso (`best.pt`) automaticamente para a pasta `modelo/`.

Para rodar de dentro da pasta `pipeline_final`:
```powershell
uv run train/train.py
```

---

## Como Iniciar a API

O servidor da API carrega o modelo em `modelo/best.pt`, expõe o endpoint de predição e serve as páginas HTML de testes e curadoria.

Para rodar a partir da pasta `pipeline_final`:
```powershell
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível nos endereços:
- **Interface Web Principal**: [http://localhost:8000/](http://localhost:8000/)
- **Painel de Curadoria**: [http://localhost:8000/curadoria](http://localhost:8000/curadoria)
- **Documentação Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Como Testar a API (Linha de Comando)

Você pode simular o envio de imagens de validação para a API usando o script de testes. Se nenhuma imagem for passada por parâmetro, ele selecionará automaticamente a primeira imagem válida no diretório de validação configurado no `config.py`.

De dentro da pasta `pipeline_final` (com a API ativa), execute:
```powershell
uv run api/test_client.py
```

Para testar uma imagem específica:
```powershell
uv run api/test_client.py --image "caminho/para/imagem.jpg"
```

---

## Fluxo de Curadoria

1. Todas as imagens enviadas via endpoint `/predict` entram automaticamente no diretório temporário `pipeline_final/api/curadoria/imagens/` e na fila do `dados.json`.
2. Um profissional de volume acessa a tela [http://localhost:8000/curadoria](http://localhost:8000/curadoria) para revisar as predições pendentes.
3. Se o modelo acertou, clique em **Confirmar Predição**. Se o modelo errou, selecione o volume correto e clique em **Reclassificar**.
4. O sistema move fisicamente o arquivo para a pasta definitiva de treinamento (`data/processadas_limpas/train/<classe>/`), pronto para ser computado no próximo treinamento de produção (`train.py`).
