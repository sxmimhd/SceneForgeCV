
import json
from pathlib import Path

import plotly.graph_objects as go
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
REPORT_PATH = ROOT / "reports" / "detection" / "evaluation_report.json"
OUTPUT_DIR = ROOT / "reports" / "figures"


def load_report() -> dict:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation report not found: {REPORT_PATH}\n"
            "Run src/evaluate_detector.py first."
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_class_order() -> list[str]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["dataset"]["classes"]


def create_comparison_chart(
    classes: list[str],
    per_class: dict,
    metric_keys: list[str],
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    figure = go.Figure()

    for metric_key, label in zip(metric_keys, labels):
        values = [
            float(per_class[class_name][metric_key])
            for class_name in classes
        ]

        figure.add_trace(
            go.Bar(
                name=label,
                x=classes,
                y=values,
                text=[f"{value:.3f}" for value in values],
                textposition="auto",
                cliponaxis=False,
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Minecraft mob",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        barmode="group",
        template="plotly_white",
        legend_title="Metric",
        font=dict(size=13),
        margin=dict(l=80, r=40, t=90, b=80),
        width=1400,
        height=800,
    )

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
    class_order = load_class_order()
    per_class = report["per_class"]

    missing_classes = [
        class_name
        for class_name in class_order
        if class_name not in per_class
    ]

    if missing_classes:
        raise ValueError(
            f"Evaluation report is missing classes: {missing_classes}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    create_comparison_chart(
        classes=class_order,
        per_class=per_class,
        metric_keys=["precision", "recall"],
        labels=["Precision", "Recall"],
        title="YOLOv8n — Precision vs Recall by Class",
        output_path=OUTPUT_DIR / "detection_precision_recall.png",
    )

    create_comparison_chart(
        classes=class_order,
        per_class=per_class,
        metric_keys=["map50", "map50_95"],
        labels=["mAP@50", "mAP@50–95"],
        title="YOLOv8n — Mean Average Precision by Class",
        output_path=OUTPUT_DIR / "detection_map_comparison.png",
    )

    print(f"\nPNG charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
