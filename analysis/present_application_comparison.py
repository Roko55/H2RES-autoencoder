"""Present one frozen CVAE against the default hybrid; never fit or select models.

This is a retrospective application diagnostic. Complete comparison tables remain
unchanged. Episode lengths are averaged within each path, then over paths.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AE = "selected_CVAE"
HYBRID = "current_default_GPUfit_CPUpredict_seed42"


def read(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    annual = read("annual_mean_spread.csv")
    wind = {r["model"]: r for r in annual if r["channel"] == "wind"}
    yields = {r["model"]: r for r in read("renewable_yield_index_spread.csv")}
    paths = read("individual_path_metrics.csv")
    groups = {m: [r for r in paths if r["model"] == m and r["channel"] == "wind"]
              for m in (AE, HYBRID)}
    assert all(len(group) == 15 for group in groups.values())

    def episode(model, key):
        return sum(float(r[key]) for r in groups[model]) / len(groups[model])

    rows = [
        ["Annual wind-mean SD", "capacity factor", float(wind[AE]["observed_annual_mean_SD"]),
         float(wind[AE]["generated_annual_mean_SD"]), float(wind[HYBRID]["generated_annual_mean_SD"])],
        ["Annual renewable-yield index SD", "solar CF + wind CF", float(yields["observed_reporting"]["annual_yield_index_SD"]),
         float(yields[AE]["annual_yield_index_SD"]), float(yields[HYBRID]["annual_yield_index_SD"])],
        ["Mean low-wind episode duration", "hours", episode(AE, "real_low_wind_mean_hours"),
         episode(AE, "generated_low_wind_mean_hours"), episode(HYBRID, "generated_low_wind_mean_hours")],
        ["Mean high-wind episode duration", "hours", episode(AE, "real_high_wind_mean_hours"),
         episode(AE, "generated_high_wind_mean_hours"), episode(HYBRID, "generated_high_wind_mean_hours")],
    ]
    with (ROOT / "results/application_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "unit", "observed_reporting", AE, HYBRID])
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    colors = ["#535f69", "#d07a35", "#347b9c"]
    for ax, row, title in zip(axes, rows[:2], ["Annual wind-mean variability", "Annual renewable-yield variability"]):
        values = row[2:]
        bars = ax.bar(["Historical data\n(reference)", "Autoencoder\n(CVAE)", "Current pipeline\n(XGBoost + wind resampling)"], values, color=colors, width=.65)
        ax.set_title(title)
        ax.set_ylabel("Standard deviation of annual means")
        ax.set_ylim(0, max(values) * 1.27)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value + max(values)*.025,
                    f"{value:.5f}", ha="center", fontsize=10)
    fig.suptitle("Annual scenario diversity: autoencoder vs current H2RES pipeline")
    fig.text(.5, .025, "Bars show variation BETWEEN annual averages, not hourly error or training loss.\n"
             "Left: annual mean wind CF. Right: annual mean solar CF + wind CF.\n"
             "Reference: 5 historical years (2020–2024); each generator: 15 annual profiles.\n"
             "Retrospective pre-calibrated corpus; descriptive comparison, not a tail-probability estimate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, .93))
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/application_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
