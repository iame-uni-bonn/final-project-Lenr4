import pandas as pd

from lennart_epp.analysis.evaluate_ar_model import evaluate_ar_models
from lennart_epp.config import BLD, SRC


def task_evaluate_ar_models(
    script=SRC / "analysis" / "evaluate_ar_model.py",
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=(
        BLD / "models" / "ar_model_evaluation.pkl",
        BLD / "models" / "top_models.tex",
    ),
    max_p=15,
    criterion="aic",
):
    df = pd.read_pickle(data)
    evaluation_results = evaluate_ar_models(df, max_p=max_p, criterion=criterion)

    produces[0].parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(evaluation_results, produces[0])
    assert produces[0].exists(), f"❌ Failed to produce {produces[0]}"

    top_models_df = pd.DataFrame(evaluation_results.get("top_models", []))

    if not top_models_df.empty:
        latex_table = top_models_df.drop(
            columns=["coefficients", "integrated_coefficients"], errors="ignore"
        )

        expected_columns = ["p", "aic", "bic", "p_value", "differenced"]
        latex_table = latex_table[
            [col for col in expected_columns if col in latex_table.columns]
        ]

        with produces[1].open("w", encoding="utf-8") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Top AR Models}\n")
            f.write("\\label{tab:top_models}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("p & AIC & BIC & p-value & Differenced \\\\\n")
            f.write("\\hline\n")

            for _, row in latex_table.iterrows():
                f.write(
                    f"{int(row['p'])} & {row['aic']:.3f} & {row['bic']:.3f} & "
                    f"{row['p_value']:.5f} & {row['differenced']!s} \\\\\n"
                )

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        assert produces[1].exists(), f"❌ Failed to produce {produces[1]}"
