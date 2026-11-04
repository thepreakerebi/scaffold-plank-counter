"""Turn per-image results into the count-vs-truth table used in the report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    name: str
    predicted: int
    truth: int
    ms: float
    flagged: bool = False

    @property
    def error(self) -> int:
        return self.predicted - self.truth


def summary(rows: list[Row]) -> dict:
    total_truth = sum(r.truth for r in rows)
    abs_err = sum(abs(r.error) for r in rows)
    return {
        "mae": abs_err / len(rows),
        "total_accuracy": 1 - abs_err / total_truth,
        "exact": sum(r.error == 0 for r in rows),
        "mean_ms": sum(r.ms for r in rows) / len(rows),
    }


def markdown(rows: list[Row], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| Image | Predicted | True | Error | Time (ms) | Review flag |",
        "|---|---:|---:|---:|---:|:-:|",
        *[
            f"| {r.name} | {r.predicted} | {r.truth} | {r.error:+d} | {r.ms:.1f} | {'yes' if r.flagged else ''} |"
            for r in rows
        ],
    ]
    s = summary(rows)
    lines += [
        "",
        f"Mean absolute error {s['mae']:.1f} planks · total-count accuracy {s['total_accuracy']:.1%} · "
        f"exact on {s['exact']}/{len(rows)} images · mean time {s['mean_ms']:.1f} ms",
    ]
    return "\n".join(lines)
