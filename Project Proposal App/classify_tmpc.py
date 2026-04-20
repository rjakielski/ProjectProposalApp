import os
import re

import joblib
import numpy as np
import pandas as pd
from setfit import SetFitModel

MODEL_DIR = os.getenv(
    "TMPC_MODEL_DIR",
    "/dbfs/mnt/adls2-silver/Custom_Models/TMPC_Classifier/runs/setfit_run_c549125b/final_model",
)
NEW_DATA = os.getenv(
    "TMPC_NEW_DATA",
    "/dbfs/mnt/adls2-silver/SPP/PPT/TMPC_testing_objectives.csv",
)
OUTPUT_PATH = os.getenv(
    "TMPC_OUTPUT_PATH",
    "/dbfs/mnt/adls2-silver/SPP/PPT/TMPC_testing_results_SetFit.csv",
)

THRESHOLD = float(os.getenv("TMPC_THRESHOLD", "0.5"))

_MODEL = None


def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u00ad", "")
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def combine_project_fields(project_name: str, project_objective: str) -> str:
    parts = [str(project_name or "").strip(), str(project_objective or "").strip()]
    combined = " ".join(part for part in parts if part)
    return normalize_text(combined)


def load_model(model_dir: str = MODEL_DIR):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = SetFitModel.from_pretrained(model_dir)
    model.labels = joblib.load(os.path.join(model_dir, "labels.joblib"))
    model.model_head = joblib.load(os.path.join(model_dir, "model_head.joblib"))
    _MODEL = model
    return _MODEL


def score_texts(texts, model=None):
    if model is None:
        model = load_model()

    embeddings = model.model_body.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=False,
    )
    probs = model.model_head.predict_proba(embeddings)
    top2_idx = np.argsort(-probs, axis=1)[:, :2]

    top1_labels = [model.labels[i] for i in top2_idx[:, 0]]
    top2_labels = [model.labels[i] for i in top2_idx[:, 1]]
    top1_conf = probs[np.arange(len(probs)), top2_idx[:, 0]]
    top2_conf = probs[np.arange(len(probs)), top2_idx[:, 1]]

    results = []
    for idx, text in enumerate(texts):
        top1 = float(top1_conf[idx])
        predicted = top1_labels[idx] if top1 >= THRESHOLD else "Needs review"
        results.append(
            {
                "text": text,
                "predicted_TMPC": predicted,
                "prediction_confidence": top1,
                "top1_class": top1_labels[idx],
                "top1_confidence": top1,
                "top2_class": top2_labels[idx],
                "top2_confidence": float(top2_conf[idx]),
            }
        )
    return results


def classify_project(project_name: str, project_objective: str, model=None):
    text = combine_project_fields(project_name, project_objective)
    return score_texts([text], model=model)[0]


def classify_dataframe(df: pd.DataFrame, text_column: str = "name_and_objective", model=None):
    if text_column not in df.columns:
        raise ValueError(f"CSV must include a '{text_column}' column.")

    df = df.copy()
    df["text"] = df[text_column].fillna("").astype(str).str.strip().map(normalize_text)
    results = score_texts(df["text"].tolist(), model=model)

    results_df = pd.DataFrame(results)
    for column in results_df.columns:
        df[column] = results_df[column]
    return df


def classify_csv(input_path: str = NEW_DATA, output_path: str = OUTPUT_PATH, model=None):
    df = pd.read_csv(input_path, encoding="latin-1")
    df = classify_dataframe(df, model=model)

    df.to_csv(output_path, index=False)
    return output_path


def main():
    saved_to = classify_csv()
    print("saved predictions to:", saved_to)


if __name__ == "__main__":
    main()
