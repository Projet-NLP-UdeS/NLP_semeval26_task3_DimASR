"""
Evaluate models on metrics.
"""
import math
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_prd(model, dataloder, type="dev"):
    if type == "dev":
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
                all_labels.append(labels)
        preds = np.vstack(all_preds)
        lables = np.vstack(all_labels)

        pred_v = preds[:,0]
        pred_a = preds[:,1]

        gold_v = lables[:,0]
        gold_a = lables[:,1]

        return pred_v, pred_a, gold_v, gold_a

    elif type == "pred":
        all_preds = []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
        preds = np.vstack(all_preds)

        pred_v = preds[:, 0]
        pred_a = preds[:, 1]

        return pred_v, pred_a

def predict_to_dataframe(model, dataloder, dataframe, pred_v_col="Pred_Valence", pred_a_col="Pred_Arousal"):
    """
    Run batched inference and append predictions to a copy of the input dataframe.

    Works with ID column check.
    """
    rows = []
    with torch.no_grad():
        for batch in dataloder:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask).cpu().numpy()

            batch_ids = batch.get("ID")
            rows.extend(
                {
                    "ID": id_,
                    pred_v_col: v, #round(v,2),
                    pred_a_col: a, #round(a,2),
                    # "Notes: VA outputs must be within [1, 9], rounded to two decimals."
                    # from the official SemEval GitHub
                }
                for id_, v, a in zip(batch_ids, outputs[:, 0], outputs[:, 1])
            )
    
    preds_df = pd.DataFrame(rows)
    df_with_preds = dataframe.copy()
    df_with_preds = dataframe.merge(
        preds_df[['ID', pred_v_col, pred_a_col]],
        on='ID', how='left'
    )

    if df_with_preds[pred_v_col].isna().any():
        ValueError("There are missing predictions.")

    return df_with_preds

def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v, is_norm=False):
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range.")

    # Calcul du PCC (Pearson Correlation Coefficient)
    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]

    # Calcul du RMSE séparé pour la Valence et l'Arousal avec Numpy
    rmse_v = np.sqrt(np.mean((gold_v - pred_v)**2))
    rmse_a = np.sqrt(np.mean((gold_a - pred_a)**2))

    # Calcul du RMSE global
    gold_va = np.concatenate((gold_v, gold_a))
    pred_va = np.concatenate((pred_v, pred_a))
    rmse_va_global = np.sqrt(np.mean((gold_va - pred_va)**2))

    # Appliquation de la logique de normalisation si is_norm est True
    if is_norm:
        rmse_v = rmse_v / math.sqrt(128)
        rmse_a = rmse_a / math.sqrt(128)
        rmse_va_global = rmse_va_global / math.sqrt(128)

    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_V': rmse_v,
        'RMSE_A': rmse_a,
        'RMSE_VA': rmse_va_global
    }
