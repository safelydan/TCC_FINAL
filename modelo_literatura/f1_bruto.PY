import pandas as pd
from sklearn.metrics import classification_report
import glob
import os

INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_FINAL\modelo_literatura\resultados"   # pasta onde extraímos os CSVs


for file in glob.glob(os.path.join(INPUT_FOLDER, "*.csv")):
    df = pd.read_csv(file)


    y_true = df["sentimento_manual"]  
    y_pred = df["sentiment"]

    print(f"\n=== Relatório para: {os.path.basename(file)} ===")
    print(classification_report(y_true, y_pred, digits=2))
