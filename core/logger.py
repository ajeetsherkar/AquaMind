from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def save_run(
    dataframe: pd.DataFrame,
    experiment_id: str,
    scenario: str,
) -> Path:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{experiment_id}_{scenario}_simulated.csv"
    )

    output_path = DATA_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path
