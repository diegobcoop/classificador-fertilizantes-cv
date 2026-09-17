import os
import io
import math
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import cv2 as cv 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np  # Biblioteca NumPy para manipulação de arrays numéricos (imagens são arrays)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
# from collections import Counter
# # modelos de testes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
import joblib

class Utils:
    DEFAULT_PCA_COMPONENTS = 42
    DEFAULT_BINS = 128
    DEFAULT_POINTS_MASK = np.array([[10,700], [10,400], [500,50], [800,50], [1270,400], [1270,700]], dtype=np.int32) 


    @staticmethod
    def carregarCaminhoImagens(raiz):
        caminhos = []    
        # percore todas as pastas dentro do diretorio raiz.    
        if os.path.isdir(raiz):
            for pasta in os.listdir(raiz):
                # valida se é um caminho valido.
                caminho_completo = os.path.join(raiz, pasta)
                if os.path.isdir(caminho_completo):
                    # percore todos os arquivo da pasta, verificando se o arquivo é uma imagem, se for adiciona o caminho completo em uma lista.
                    for arquivo in os.listdir(caminho_completo):
                        if arquivo.endswith('.jpg'):
                            caminhos.append(os.path.join(raiz, pasta, arquivo))
                else:                
                    if caminho_completo.endswith('.jpg'):
                        caminhos.append(caminho_completo)
        
        return caminhos

    @staticmethod
    def getModelos(nome=None):
        models = { 
                   "KNN": KNeighborsClassifier(n_neighbors=3),
                   "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
                   "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
                   "SVM": SVC(kernel='rbf', class_weight='balanced'),
                   "GradientBoosting": GradientBoostingClassifier(),
                   "NaiveBayes": GaussianNB(),
                   "ExtraTrees": ExtraTreesClassifier(n_estimators=100)
                 }

        # models = {
        #     # ---- Clássicos rápidos / baseline fortes ----
        #     "KNN": KNeighborsClassifier(
        #         n_neighbors=7,        # mais estável que 3
        #         weights="distance"    # ajuda em regiões densas
        #     ),

        #     "RandomForest": RandomForestClassifier(
        #         n_estimators=600,
        #         max_depth=None,
        #         min_samples_leaf=2,   # reduz overfitting
        #         max_features="sqrt",
        #         n_jobs=-1,
        #         random_state=42
        #     ),

        #     "ExtraTrees": ExtraTreesClassifier(
        #         n_estimators=1000,    # ET gosta de muitas árvores
        #         max_features="sqrt",
        #         min_samples_leaf=2,
        #         n_jobs=-1,
        #         random_state=42
        #     ),

        #     "GradientBoosting": GradientBoostingClassifier(
        #         n_estimators=300,
        #         learning_rate=0.05,
        #         max_depth=3,
        #         subsample=0.8,
        #         random_state=42
        #     ),

        #     "HistGradientBoosting": HistGradientBoostingClassifier(
        #         # geralmente mais rápido/robusto que GB clássico
        #         max_depth=None,
        #         learning_rate=0.1,
        #         max_iter=300,
        #         random_state=42
        #     ),

        #     "LogisticRegression": LogisticRegression(
        #         C=3.0,
        #         max_iter=2000,
        #         solver="lbfgs",       # multiclasse OK
        #         # se usar 'saga' com l1/l2 elástica, pode setar n_jobs
        #     ),

        #     "RidgeClassifier": RidgeClassifier(
        #         alpha=1.0             # linear com regularização L2 (às vezes funciona bem em PCA)
        #     ),

        #     "SVM": SVC(
        #         kernel="rbf",
        #         C=3.0,
        #         gamma="scale",
        #         probability=False,     # coloque True só se realmente precisar de predict_proba
        #         random_state=42
        #     ),

        #     "NaiveBayes": GaussianNB(),

            # "XGBoost": XGBClassifier(
            #     n_estimators=400,
            #     learning_rate=0.1,
            #     max_depth=6,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     n_jobs=-1,
            #     random_state=42,
            #     tree_method="hist",    # rápido/estável
            #     eval_metric="logloss"
            # ),

            # "LightGBM": LGBMClassifier(
            #     n_estimators=600,
            #     learning_rate=0.05,
            #     num_leaves=31,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=42,
            #     n_jobs=-1
            # )
        # }
        
        if not nome:
            return [(name, model) for name, model in models.items()]
        else:
            return [(nome, models[nome])]

    @staticmethod
    def desenharPoligono(imagem, cor=None, espessura=2):
        imagem_com_poligono = imagem.copy()
        cor_final = cor

        if cor_final is None:
            if len(imagem_com_poligono.shape) == 2: # Imagem em tons de cinza
                cor_final = 255  # Branco em escala de cinza
            else: # Imagem colorida
                cor_final = (255, 255, 255) # Branco em BGR

        # Desenha o polígono com a cor apropriada
        cv.polylines(imagem_com_poligono, [Utils.DEFAULT_POINTS_MASK], 
                    isClosed=True, 
                    color=cor_final, 
                    thickness=espessura)
        
        return imagem_com_poligono

    @staticmethod
    def mascaraPoligono(imagem):
        # 2. Criar uma máscara vazia com as mesmas dimensões da imagem original
        mascara_poligono = np.zeros(imagem.shape[:2], dtype=np.uint8)
        # mascara_poligono = np.any(imagem > 0, axis=2).astype(np.uint8) * 255

        # 3. Desenhar o polígono preenchido na máscara
        #    cv.fillPoly espera uma lista de arrays de pontos, por isso [pts_box_interesse]
        cv.fillPoly(mascara_poligono, [Utils.DEFAULT_POINTS_MASK], 255) # 255 para branco

        return mascara_poligono

    @staticmethod
    def aplicarPoligono(imagem):
        return cv.bitwise_and(imagem, imagem, mask=Utils.mascaraPoligono(imagem))
    

    @staticmethod
    def carregaDataset(dataset):
        X = []
        y = []
        
        data = np.load(dataset)
        X = data['X']
        y = data['y']
        
        return X, y
    

    @staticmethod
    def aplicaSmote(X, y):    
        sm = SMOTE(random_state=42)

        # Aplica o SMOTE
        X_res, y_res = sm.fit_resample(X, y)

        # contagem = Counter(y_res)
        # for classe, qtd in contagem.items():
        #     print(f"Classe '{classe}': {qtd} registros")
            
        # contagem = Counter(y)
        # for classe, qtd in contagem.items():
        #     print(f"Classe '{classe}': {qtd} registros")
            
            
        return X_res, y_res
    


    @staticmethod
    def geraPca(X, y, datasetName=""):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=Utils.DEFAULT_PCA_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)

        # Salva scaler e PCA para usar depois na classificação
        if datasetName:
            joblib.dump(scaler, rf"..\datasets\model\scaler_{datasetName}.pkl")
            joblib.dump(pca, rf"..\datasets\model\pca_{datasetName}.pkl")

        return X_pca, y
        
       
    @staticmethod
    def treinarModelo(X, y, tamanhoTeste, titulo, plot=True, modelo=None, nome_arquivo=""):
        # Codifica os rótulos
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Divide treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=tamanhoTeste, random_state=42, stratify=y_encoded)

        resultados = []

        for nome, modelo in Utils.getModelos(modelo):
            modelo.fit(X_train, y_train)
            joblib.dump(modelo, rf'..\datasets\model\{nome}_{nome_arquivo}.pkl')
            joblib.dump(le, rf'..\datasets\model\le_{nome}_{nome_arquivo}.pkl')

            y_pred = modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            acc_train = modelo.score(X_train, y_train)

            resultados.append({
                'Modelo': nome,
                'Dataset': titulo,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'y_test': y_test,
                'y_pred': y_pred,
                'Train_Accuracy': acc_train, 
            })

            if plot:
                # Matriz de confusão (ainda exibe individual)
                plt.figure(figsize=(6, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
                plt.title(f'Matriz de Confusão - {nome} ({titulo})')
                plt.ylabel('Real')
                plt.xlabel('Previsto')
                plt.tight_layout()
                plt.show()

        return pd.DataFrame(resultados)

    @staticmethod
    def compararResultados(X1, y1, titulo1, X2, y2, titulo2, test_size=0.2):
        df1 = Utils.treinarModelo(X1, y1, test_size, titulo1, False)
        df2 = Utils.treinarModelo(X2, y2, test_size, titulo2, False)

        df_result = pd.concat([df1, df2], ignore_index=True)

        # Plot comparativo de acurácia
        # plt.figure(figsize=(10, 5))
        # sns.barplot(data=df_result, x='Modelo', y='Accuracy', hue='Dataset')
        # plt.title('Comparação de Acurácia entre Datasets')
        # plt.ylim(0, 1.0)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        
        df_melted = pd.melt(
            df_result,
            id_vars=['Modelo', 'Dataset'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1'],
            var_name='Métrica',
            value_name='Valor'
        )
        
        # Cria os subplots com separação clara por métrica
        g = sns.catplot(
            data=df_melted,
            kind="bar",
            x="Modelo", y="Valor",
            hue="Dataset", row="Métrica",   
            palette="Set2",
            height=4, aspect=2,
            sharey=False
        )

        # Ajustes de layout
        g.figure.subplots_adjust(top=0.92)
        g.figure.suptitle("Comparação de Accuracy, Precision, Recall e F1", fontsize=16)

        # Rotaciona os nomes dos modelos
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelrotation=45)
            
        g._legend.set_bbox_to_anchor((1.02, 1)) # type: ignore

        plt.tight_layout()
        plt.show()

        return df_result
    
    @staticmethod
    def gerarHistograma(imagem, caminho, mascara):
        # 2. Calcula o histograma com N bins
        hist = cv.calcHist([imagem], [0], mask=mascara, histSize=[Utils.DEFAULT_BINS], ranges=[0, 256])
        hist = cv.normalize(hist, hist).flatten()  # normaliza e transform
        
        return hist, os.path.basename(os.path.dirname(caminho))
    
    @staticmethod
    def gerarHistogramaHsv(imagem, mascara):
        imagem_hsv = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)
        
        # Separa os 3 canais BGR
        canais = cv.split(imagem_hsv)  # retorna [H, S, V]
        vetor_final = []

        for canal in canais:
            hist = cv.calcHist([canal], [0],  mask=mascara, histSize=[Utils.DEFAULT_BINS], ranges=[0, 256])        
            hist = cv.normalize(hist, hist).flatten()
            vetor_final.extend(hist)

        return np.array(vetor_final)
    
    @staticmethod
    def gerarHistogramaRgb(imagem, mascara):
        # Separa os 3 canais BGR
        canais = cv.split(imagem)  # retorna [B, G, R]
        vetor_final = []

        for canal in canais:
            hist = cv.calcHist([canal], [0],  mask=mascara, histSize=[Utils.DEFAULT_BINS], ranges=[0, 256])        
            hist = cv.normalize(hist, hist).flatten()
            vetor_final.extend(hist)

        return np.array(vetor_final)
    
    
    @staticmethod
    def plotarColuna(imagens, titulos=None, _ncols=3):
        """
        Plota uma lista de imagens ou histogramas em uma grade dinâmica.
        imagens: lista de arrays (imagens 2D/3D ou histogramas 1D/2D).
        titulos: lista opcional de strings com títulos para cada item.
        """
        n_imagens = len(imagens)

        if n_imagens == 0:
            print("Nenhuma imagem fornecida.")
            return

        ncols = min(_ncols, n_imagens)
        nrows = math.ceil(n_imagens / ncols)

        figsize = (ncols * 15, nrows * 10)
        plt.figure(figsize=figsize)

        for idx, item in enumerate(imagens):
            plt.subplot(nrows, ncols, idx + 1)

            # Converte histogramas 2D para 1D
            if isinstance(item, np.ndarray) and (len(item.shape) == 1 or (len(item.shape) == 2 and item.shape[1] == 1)):
                # hist = item.ravel()
                # hist_norm = hist / hist.max()  # normaliza
                # plt.bar(range(len(hist_norm)), hist_norm, width=1.0, color='gray')
                # plt.xlim([0, 256])
                # plt.ylim([0, 1])
                # plt.grid(alpha=0.3)
                # plt.xlabel('Intensidade', fontsize=14)
                # plt.ylabel('Frequência normalizada', fontsize=14)
                
                hist = item.ravel()  # achata para 1D
                plt.plot(hist)
                plt.title(titulos[idx] if titulos and idx < len(titulos) else f'Histograma {idx+1}', fontsize=25)
                plt.xlabel('Intensidade')
                plt.ylabel('Frequência')
            else:
                if len(item.shape) == 2 or item.ndim == 2:
                    plt.imshow(item, cmap='gray')
                else:
                    plt.imshow(cv.cvtColor(item, cv.COLOR_BGR2RGB))
                # plt.title(titulos[idx] if titulos and idx < len(titulos) else f'Imagem {idx+1}', fontsize=25)
                plt.axis('off')
                
            # Define título se existir
            if titulos and idx < len(titulos):
                plt.title(titulos[idx], fontsize=35)
            else:
                plt.title(f'Imagem {idx+1}')

        plt.tight_layout(pad=2.0)
        plt.show()

    @staticmethod
    def histogramaRgbParaImagem(hist_b, hist_g, hist_r):
        figsize=(6,4)
        # plota as 3 curvas numa figura
        fig = plt.figure(figsize=figsize)
        plt.plot(hist_r, color='r', label='R')
        plt.plot(hist_g, color='g', label='G')
        plt.plot(hist_b, color='b', label='B')
        plt.xlabel('Intensidade'); 
        plt.ylabel('Frequência')
        plt.grid(True); plt.legend()

        # salva em memória (PNG) e lê com OpenCV (já vem BGR)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=fig.dpi)
        plt.close(fig)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr_plot = cv.imdecode(data, cv.IMREAD_COLOR)  # BGR
        return img_bgr_plot      
    
    @staticmethod
    def histogramaHsvParaImagem(hist_h, hist_s, hist_v):
        figsize=(6,4)
        fig = plt.figure(figsize=figsize)
        plt.plot(hist_h, label='H', color='m')  # magenta p/ H
        plt.plot(hist_s, label='S', color='c')  # ciano p/ S
        plt.plot(hist_v, label='V', color='y')  # amarelo p/ V
        plt.xlabel('Intensidade'); plt.ylabel('Frequência')
        plt.grid(True); plt.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=fig.dpi)
        plt.close(fig)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr_plot = cv.imdecode(data, cv.IMREAD_COLOR)  # BGR
        return img_bgr_plot      
    

    @staticmethod
    def tratarImagemRgb(arquivo, X, y):
        # 1. Carrega imagem
        img = cv.imread(arquivo)
        img = cv.resize(img, (1280, 720))

        vetor = Utils.gerarHistogramaRgb(img, None)    
        X.append(vetor)
        y.append(os.path.basename(os.path.dirname(arquivo)))

    @staticmethod
    def tratarImagemRgbPoligono(arquivo, X, y, img=None):
        if img is None:
            imagem = cv.imread(arquivo)
            if imagem is not None:
                imagem = cv.resize(imagem, (1280, 720))
        else:
            imagem = img

        vetor = Utils.gerarHistogramaRgb(imagem, Utils.mascaraPoligono(imagem))
        X.append(vetor)
        y.append(os.path.basename(os.path.dirname(arquivo)))

    @staticmethod
    def tratarImagemHsv(arquivo, X, y):
        # 1. Carrega imagem
        img = cv.imread(arquivo)
        img = cv.resize(img, (1280, 720))
        vetor = Utils.gerarHistogramaHsv(img, None)
        X.append(vetor)
        y.append(os.path.basename(os.path.dirname(arquivo)))

    @staticmethod
    def tratarImagemHsvPoligono(arquivo, X, y, img=None):
        if img is None:
            imagem = cv.imread(arquivo)
            if imagem is not None:
                imagem = cv.resize(imagem, (1280, 720))
        else:
            imagem = img
        vetor = Utils.gerarHistogramaHsv(imagem, Utils.mascaraPoligono(imagem))
        X.append(vetor)
        y.append(os.path.basename(os.path.dirname(arquivo)))

    @staticmethod
    def tratarImagemGray(arquivo, X, y):
        img = cv.imread(arquivo, 0)
        img = cv.resize(img, (1280, 720))
        X_temp, y_temp = Utils.gerarHistograma(img, arquivo, None)
        X.append(X_temp)
        y.append(y_temp)

    @staticmethod
    def tratarImagemGrayPoligono(arquivo, X, y, img=None):
        if img is None:
            imagem = cv.imread(arquivo, 0)
            if imagem is not None:
                imagem = cv.resize(imagem, (1280, 720))
        else:
            imagem = img
        X_temp, y_temp = Utils.gerarHistograma(imagem, arquivo, Utils.mascaraPoligono(imagem))
        X.append(X_temp)
        y.append(y_temp)

    @staticmethod
    def confusion_from_df_pandas(df, modelo, dataset, normalize=False):
        d = df[(df["Modelo"] == modelo) & (df["Dataset"] == dataset)].copy()
        d["y_true"] = d["y_test"].apply(list)
        d["y_pred"] = d["y_pred"].apply(list)
        d = d.explode(["y_true", "y_pred"], ignore_index=True)
        cm = pd.crosstab(d["y_true"], d["y_pred"])
        if normalize:
            cm = cm.div(cm.sum(axis=1), axis=0)
        return cm

    @staticmethod
    def plot_cm(cm_df, title="Matriz de Confusão", normalize=False):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
        plt.title(title)
        plt.ylabel("Real")
        plt.xlabel("Previsto")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def matrizConfusao(df):
        for (modelo, dataset), _ in df.groupby(["Modelo", "Dataset"]):
            cm_df = Utils.confusion_from_df_pandas(df, modelo, dataset, normalize=False)
            # se tiver label encoder correspondente, renomeie:
            # le = joblib.load(rf"..\datasets\model\le_{modelo}_{dataset.replace(' ', '_')}.pkl")
            # cm_df = rename_with_labels(cm_df, le)
            Utils.plot_cm(cm_df, title=f"{modelo} - {dataset}")


    @staticmethod
    def construirPipeline(modelo):
        return {
            "BASE": Pipeline([
                ("scaler", StandardScaler()),
                ("classificador", modelo)
            ]),
            "PCA": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=Utils.DEFAULT_PCA_COMPONENTS)),
                ("classificador", modelo)
            ]),
            "SMOTE": Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                ("classificador", modelo)
            ]),
            "SMOTE_PCA": Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=Utils.DEFAULT_PCA_COMPONENTS)),
                ("classificador", modelo)
            ])
        }
    
    @staticmethod
    def testarModelosPipelineCV(modelo, X, y, dataset):
        le = LabelEncoder()
        y_encoder = le.fit_transform(y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "balanced_accuracy": "balanced_accuracy",    
        }

        resultados = []

        for nome, modelo in Utils.getModelos(modelo):
            modelPipeline = Utils.construirPipeline(modelo)

            for pipe_nome, pipe in modelPipeline.items():
                cv_scores = cross_validate(
                    pipe, 
                    X, 
                    y_encoder,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=False
                )

                row = {
                    "Modelo": nome,
                    "Dataset": dataset,
                    "Pipeline": pipe_nome,
                    **{f"CV_{k}_mean": np.mean(v) for k, v in cv_scores.items() if k.startswith("test_")},
                    **{f"CV_{k}_std":  np.std(v)  for k, v in cv_scores.items() if k.startswith("test_")},
                }

                resultados.append(row)
        return pd.DataFrame(resultados)

    @staticmethod
    def testarModelosPipelineSplit(X, y, test_size, modelo, dataset):
        le = LabelEncoder()
        y_encoder = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoder, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoder
        )

        resultados = []

        for nome, modelo in Utils.getModelos(modelo):
            modelPipeline = Utils.construirPipeline(modelo)

            for pipe_nome, pipe in modelPipeline.items():
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                resultados.append({
                    "Modelo": nome,
                    "Dataset": dataset,
                    "Pipeline": pipe_nome,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "y_test": y_test, 
                    "y_pred": y_pred
                })
        return pd.DataFrame(resultados)
    
    @staticmethod
    def _get_output_dir():
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "ml_tradicional"))
        os.makedirs(base, exist_ok=True)
        return base

    @staticmethod
    def treinarMelhoresModelos(X, y, modelo, pipeline, dataset):
        le = LabelEncoder()
        y_encoder = le.fit_transform(y)
        output_dir = Utils._get_output_dir()
        
        for nome, modelo in Utils.getModelos(modelo):
            modelPipeline = Utils.construirPipeline(modelo)[pipeline]

            for pipe in modelPipeline:
                pipe.fit(X, y_encoder)

                joblib.dump(pipe, os.path.join(output_dir, f"{nome}_{dataset}_{pipeline}.pkl"))
                joblib.dump(le,   os.path.join(output_dir, f"le_{nome}_{dataset}_{pipeline}.pkl"))
       
    
    @staticmethod
    def classificarPipeline(modelo, dataset, pipeline, X_feat, img_path):
        resultado = []
        output_dir = Utils._get_output_dir()
        for nome, _ in Utils.getModelos(modelo):
            pipe = joblib.load(os.path.join(output_dir, f"{nome}_{dataset}_{pipeline}.pkl"))
            le   = joblib.load(os.path.join(output_dir, f"le_{nome}_{dataset}_{pipeline}.pkl"))

            x = np.array(X_feat)
            if x.ndim == 1: 
                x = x.reshape(1, -1)

            y_pred = pipe.predict(x)
            classe = le.inverse_transform(y_pred)[0]

            resultado.append({
                'Modelo': nome, 
                'Dataset': dataset,
                'Pipeline': pipeline,
                'Imagem': os.path.basename(img_path), 
                'Classe': classe, 
            })

        return pd.DataFrame(resultado)
    

    