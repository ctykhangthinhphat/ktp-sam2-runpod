import runpod
from segmentation_engine import engine

def handler(job):
    d=job.get("input") or {}
    a=str(d.get("action","health")).lower()
    if a=="health": return engine.health()
    if a=="auto": return engine.auto(d["image"],int(d.get("max_masks",40)),float(d.get("min_area_ratio",0.0005)))
    if a=="point": return engine.point(d["image"],d.get("points",[]),d.get("labels",[]))
    if a=="box": return engine.box(d["image"],d["box"])
    if a=="refine": return engine.refine(d["image"],d.get("points",[]),d.get("labels",[]),d.get("box"))
    raise ValueError(f"Unsupported action: {a}")

runpod.serverless.start({"handler":handler})
