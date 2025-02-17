import pandas as pd

from lennart_epp.analysis.evaluate_ar_model import evaluate_ar_models
from lennart_epp.config import BLD, SRC


def task_evaluate_ar_models(
    script=SRC / "analysis" / "evaluate_ar_model.py",
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "models" / "ar_model_evaluation.pkl",
    max_p=15,
    criterion="aic",
):
    """Evaluate AR(p) models and save top models and metrics."""
    df = pd.read_pickle(data)

    evaluation_results = evaluate_ar_models(df, max_p=max_p, criterion=criterion)

    top_models_data = evaluation_results.get("top_models", [])
    metrics_data = evaluation_results.get("model_metrics", [])
    metadata_data = evaluation_results.get("metadata", {})

    top_models_df = pd.DataFrame(top_models_data) if top_models_data else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()
    metadata_df = pd.DataFrame([metadata_data]) if metadata_data else pd.DataFrame()

    results = {
        "top_models": top_models_df,
        "model_metrics": metrics_df,
        "metadata": metadata_df,
    }

    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, produces)

    assert produces.exists(), f"❌ Failed to produce {produces}"
