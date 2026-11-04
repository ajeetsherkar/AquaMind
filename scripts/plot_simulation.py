from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

SCENARIOS = [
    "healthy",
    "prefilter_restriction",
    "membrane_performance_loss",
    "feed_tds_spike",
    "stuck_sensor",
]


def find_latest_csv(scenario: str) -> Path:
    matches = sorted(
        DATA_DIR.glob(f"*_{scenario}_simulated.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No generated CSV found for scenario: {scenario}. "
            "Run `python run_simulator.py` first."
        )
    return matches[-1]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(14, 12),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for index, scenario in enumerate(SCENARIOS):
        df = pd.read_csv(find_latest_csv(scenario))
        time_seconds = range(len(df))

        ax = axes[index]
        ax.plot(time_seconds, df["feed_tds"], label="Feed TDS")
        ax.plot(time_seconds, df["permeate_tds"], label="Permeate TDS")
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Sample")
        ax.set_ylabel("TDS")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].axis("off")

    output_path = REPORTS_DIR / "session1_scenario_overview.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
