
import os
import glob
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import argparse

VALID_LABELS_DEFAULT = ["Positive", "Negative", "Neutral"]

def _norm_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("positive", "pos", "positivo", "positiva"):
        return "Positive"
    if s in ("negative", "neg", "negativo", "negativa"):
        return "Negative"
    if s in ("neutral", "neutro", "neutra"):
        return "Neutral"
    if s in ("", "nan", "none", "indefinido"):
        return None
    return None 

def _confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def _precision_recall_f1_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1        = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return precision, recall, f1

def avaliar_saidas(output_dir: str, labels: List[str]) -> None:
    arquivos = sorted(glob.glob(os.path.join(output_dir, "*_analise.csv")))
    if not arquivos:
        print("Nenhum *_analise.csv encontrado em", output_dir)
        return

    linhas_all_true, linhas_all_pred = [], []
    registros_metricas = []

    for caminho in arquivos:
        base = os.path.splitext(os.path.basename(caminho))[0].replace("_analise", "")
        try:
            df = pd.read_csv(caminho, dtype=str)
        except Exception as e:
            print(f"Falha ao ler {caminho}: {e}")
            continue

        if "classificacao" not in df.columns or "sentimento_manual" not in df.columns:
            print(f"Pulando {caminho} — colunas 'classificacao' e/ou 'sentimento_manual' não encontradas.")
            continue

        df["y_pred"] = df["classificacao"].map(_norm_label)
        df["y_true"] = df["sentimento_manual"].map(_norm_label)

        aval = df.dropna(subset=["y_true"]).copy()
        aval = aval[aval["y_true"].isin(labels)]
        aval = aval[aval["y_pred"].isin(labels)]

        if aval.empty:
            print(f"Sem exemplos válidos para avaliar em {caminho}.")
            continue

        y_true = aval["y_true"].tolist()
        y_pred = aval["y_pred"].tolist()

        linhas_all_true.extend(y_true)
        linhas_all_pred.extend(y_pred)

        cm = _confusion_matrix(y_true, y_pred, labels)
        precision, recall, f1 = _precision_recall_f1_from_cm(cm)

        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        f1_macro = f1.mean()

        tp_total = np.trace(cm)
        total = cm.sum()
        f1_micro = tp_total / total if total > 0 else 0.0

        met_df = pd.DataFrame({
            "arquivo": [base],
            "amostras_avaliadas": [len(aval)],
            "accuracy": [round(float(accuracy), 4)],
            "f1_macro": [round(float(f1_macro), 4)],
            "f1_micro": [round(float(f1_micro), 4)],
            **{f"f1_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, f1)},
            **{f"precision_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, precision)},
            **{f"recall_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, recall)},
        })
        met_path = os.path.join(output_dir, f"{base}_metricas.csv")
        met_df.to_csv(met_path, index=False, encoding="utf-8-sig")
        print(f"Métricas salvas em '{met_path}'")

        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
        cm_path = os.path.join(output_dir, f"{base}_confusao.csv")
        cm_df.to_csv(cm_path, encoding="utf-8-sig")
        print(f"Matriz de confusão salva em '{cm_path}'")

        registros_metricas.append({
            "arquivo": base,
            "amostras_avaliadas": len(aval),
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_micro": round(float(f1_micro), 4),
            **{f"f1_{lab.lower()}": round(float(val), 4) for lab, val in zip(labels, f1)}
        })

    if linhas_all_true and linhas_all_pred:
        cm_all = _confusion_matrix(linhas_all_true, linhas_all_pred, labels)
        precision_all, recall_all, f1_all = _precision_recall_f1_from_cm(cm_all)
        acc_all = (np.array(linhas_all_true) == np.array(linhas_all_pred)).mean()
        f1_macro_all = f1_all.mean()
        tp_total = np.trace(cm_all)
        total = cm_all.sum()
        f1_micro_all = tp_total / total if total > 0 else 0.0

        geral_df = pd.DataFrame({
            "escopo": ["AGREGADO_TODOS_ARQUIVOS"],
            "amostras_avaliadas": [int(total)],
            "accuracy": [round(float(acc_all), 4)],
            "f1_macro": [round(float(f1_macro_all), 4)],
            "f1_micro": [round(float(f1_micro_all), 4)],
            **{f"f1_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, f1_all)},
            **{f"precision_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, precision_all)},
            **{f"recall_{lab.lower()}": [round(float(val), 4)] for lab, val in zip(labels, recall_all)},
        })

        por_arquivo_df = pd.DataFrame(registros_metricas)

        geral_path = os.path.join(output_dir, "metricas_geral.csv")
        geral_df.to_csv(geral_path, index=False, encoding="utf-8-sig")
        print(f"Métricas agregadas salvas em '{geral_path}'")

        por_arquivo_path = os.path.join(output_dir, "metricas_por_arquivo.csv")
        por_arquivo_df.to_csv(por_arquivo_path, index=False, encoding="utf-8-sig")
        print(f"Métricas por arquivo salvas em '{por_arquivo_path}'")
    else:
        print("Não foi possível calcular métricas agregadas (nenhuma linha válida encontrada).")

def main():
    parser = argparse.ArgumentParser(
        description="Avaliar acurácia e F1 a partir de *_analise.csv com colunas 'classificacao' e 'sentimento_manual'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Diretório onde estão os arquivos *_analise.csv."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="Positive,Negative,Neutral",
        help="Lista de classes separadas por vírgula. Ex: 'Positive,Negative,Neutral' (ou incluir Mixed)."
    )
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        labels = VALID_LABELS_DEFAULT

    avaliar_saidas(args.output_dir, labels)

if __name__ == "__main__":
    main()
