def ranking_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return metrics["recall@20"], metrics["map"], metrics["ndcg@20"]


def ranking_progress(
    current: tuple[float, float, float],
    anchor: tuple[float, float, float],
    min_map_delta: float,
) -> bool:
    # ranking selection uses all three values; patience uses recall and MAP
    return (
        current[0] > anchor[0]
        or (
            current[0] == anchor[0]
            and current[1] >= anchor[1] + min_map_delta
        )
    )


def loss_progress(current: float, anchor: float, min_delta: float) -> bool:
    return current <= anchor - min_delta
