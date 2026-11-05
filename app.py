import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from stable_baselines3 import PPO

ORDER_DAYS = np.array([0, 1, 2, 3, 5, 7, 10, 14])

custom = {
    "learning_rate": 0.0003,
    "lr_schedule": lambda _: 0.0003,
    "clip_range": lambda _: 0.2,
}
model = PPO.load("ppo_inventory", device="cpu", custom_objects=custom)
params = pd.read_csv("materials_params.csv").set_index("material_id")

app = FastAPI(title="Inventory DRL API")


class Item(BaseModel):
    material_id: str
    current_stock: float
    pipeline: float = 0.0
    use_7d: Optional[float] = None
    use_30d: Optional[float] = None
    weekday: int = 0


class Req(BaseModel):
    items: List[Item]


@app.get("/")
def health():
    return {"status": "ok", "materials": len(params)}


@app.post("/predict")
def predict(req: Req):
    out = []
    for it in req.items:
        if it.material_id not in params.index:
            raise HTTPException(status_code=404, detail="unknown material_id: " + it.material_id)
        p = params.loc[it.material_id]
        u7 = it.use_7d if (it.use_7d and it.use_7d > 0) else float(p["use_7d"])
        u30 = it.use_30d if (it.use_30d and it.use_30d > 0) else float(p["use_30d"])
        u = u7
        lead = float(p["lead_time_days"])
        moq = float(p["moq"])
        safe = float(p["safety_stock"])
        shelf = float(p["shelf_life_days"])

        o = [it.current_stock / u, it.pipeline / u]
        o += [(it.current_stock + it.pipeline) / u, safe / u, lead / 14]
        o += [moq / u, min(shelf, 30) / 30, u7 / u, u30 / u]
        o += [(it.weekday % 7) / 6]
        obs = np.clip(np.array(o, dtype=np.float32), -50, 50)

        a, _ = model.predict(obs, deterministic=True)
        days = int(ORDER_DAYS[int(a)])
        qty = 0.0
        if days > 0:
            qty = float(np.ceil(max(days * u, moq) / moq) * moq)

        out.append({
            "material_id": it.material_id,
            "material_name": str(p["material_name"]),
            "supplier": str(p["supplier"]),
            "unit": str(p["unit"]),
            "reorder_required": qty > 0,
            "suggested_order_qty": qty,
            "order_days_of_use": days,
            "estimated_cost": round(qty * float(p["Unit Cost"]) / moq, 2),
        })
    return {"results": out}
