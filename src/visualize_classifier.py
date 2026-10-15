
import json
from pathlib import Path

import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    ROOT / "reports" / "classification" / "evaluation_report.json"
)
OUTPUT_DIR = ROOT / "reports" / "figures"


def load_report() -> dict:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation report not found: {REPORT_PATH}\n"
            "Run src/evaluate_classifier.py first."
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_confusion_matrix(report: dict) -> None:
    class_names = report["class_names"]
    matrix = report["metrics"]["confusion_matrix"]

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}<extra></extra>"
            ),
            colorbar=dict(title="Samples"),
        )
    )

    figure.update_layout(
        title="ResNet18 — Validation Confusion Matrix",
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        template="plotly_white",
        width=1100,
        height=900,
        font=dict(size=13),
        margin=dict(l=130, r=80, t=90, b=110),
    )

    figure.update_yaxes(autorange="reversed")

    output_path = OUTPUT_DIR / "classifier_confusion_matrix.png"
    figure.write_image(
        str(output_path),
        format="png",
        width=1100,
        height=900,
        scale=2,
    )

    print(f"Saved: {output_path}")


def save_class_metrics(report: dict) -> None:
    class_names = report["class_names"]
    per_class = report["per_class"]

    metric_names = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1-score"]

    figure = go.Figure()

    for metric_name, metric_label in zip(metric_names, metric_labels):
        values = [
            float(per_class[class_name][metric_name])
            for class_name in class_names
        ]

        figure.add_trace(
            go.Bar(
                name=metric_label,
                x=class_names,
                y=values,
                text=[f"{value:.3f}" for value in values],
                textposition="auto",
                cliponaxis=False,
            )
        )

    figure.update_layout(
        title="ResNet18 — Per-Class Validation Performance",
        xaxis_title="Minecraft mob",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        barmode="group",
        template="plotly_white",
        legend_title="Metric",
        width=1400,
        height=800,
        font=dict(size=13),
        margin=dict(l=80, r=40, t=90, b=80),
    )

    output_path = OUTPUT_DIR / "classifier_class_metrics.png"
    figure.write_image(
        str(output_path),
        format="png",
        width=1400,
        height=800,
        scale=2,
    )

    print(f"Saved: {output_path}")


def main() -> None:
    report = load_report()

    required_keys = {"class_names", "metrics", "per_class"}
    missing_keys = required_keys - report.keys()

    if missing_keys:
        raise ValueError(
            f"Evaluation report is missing keys: {sorted(missing_keys)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(report)
    save_class_metrics(report)

    print(f"\nPNG charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
