import runpod
from segmentation_engine import engine


def handler(job):
    """RunPod Serverless queue handler for KTP SAM2."""
    data = job.get("input") or {}
    action = str(data.get("action", "health")).lower()

    if action == "health":
        return engine.health()

    if action == "auto":
        return engine.auto(
            data["image"],
            int(data.get("max_masks", 40)),
            float(data.get("min_area_ratio", 0.0005)),
        )

    if action == "point":
        return engine.point(
            data["image"],
            data.get("points", []),
            data.get("labels", []),
        )

    if action == "box":
        return engine.box(data["image"], data["box"])

    if action == "refine":
        return engine.refine(
            data["image"],
            data.get("points", []),
            data.get("labels", []),
            data.get("box"),
        )

    raise ValueError(f"Unsupported action: {action}")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
