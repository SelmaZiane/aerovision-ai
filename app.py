"""
Aircraft Identification API — FastAPI backend
Run: uvicorn app:app --host 0.0.0.0 --port 7860
"""

import os, gc, io, traceback, base64, json
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# ── Config (same as your inference code) ──────────────────────
HF_REPO  = "selmamalak/aircraft-eva021"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP  = torch.cuda.is_available()

ROUTER_MODEL_NAME     = "tf_efficientnet_b0"
ROUTER_IMAGE_SIZE     = 224
ROUTER_FILE           = "router/router_model.pth"
SPECIALIST_MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
SPECIALIST_IMAGE_SIZE = 448
ROUTER_CHALLENGER_THRESHOLD = 85.0
LOW_CONF_THRESHOLD = 35.0

CHALLENGER_MAP = {"commercial_regional": "helicopters_drones"}

UAV_CLASS_NAMES = {
    "mq-9","mq9","rq-4","rq4","tb-2","tb2","bayraktar","akinci",
    "global hawk","reaper","predator","uav","ucav","drone","fixed-wing uav",
}

CATEGORY_TO_SPECIALIST = {
    "bombers_transport":   "specialists/bombers_transport_v2.pth",
    "commercial_regional": "aircraft_eva02_finetuned.pth",
    "helicopters_drones":  "specialists/helicopters_drones_v2.pth",
    "military_fighters":   "specialists/military_fighters_v2.pth",
}
CATEGORY_FALLBACK = {
    "bombers_transport":   "specialists/bombers_transport.pth",
    "commercial_regional": "aircraft_eva02_finetuned.pth",
    "helicopters_drones":  "specialists/helicopters_drones.pth",
    "military_fighters":   "specialists/military_fighters.pth",
}
CATEGORY_DISPLAY = {
    "bombers_transport":   "Bombers & Transport",
    "commercial_regional": "Commercial & Regional",
    "helicopters_drones":  "Helicopters & Drones",
    "military_fighters":   "Military Fighters",
}
CATEGORY_ACCURACY = {
    "bombers_transport":   "95.7%",
    "commercial_regional": "92%",
    "helicopters_drones":  "95.3%",
    "military_fighters":   "93.5%",
}
CATEGORY_ICON = {
    "bombers_transport":   "✈",
    "commercial_regional": "🛫",
    "helicopters_drones":  "🚁",
    "military_fighters":   "⚔️",
}

_specs_path = os.path.join(os.path.dirname(__file__), "specs.json")
with open(_specs_path, "r", encoding="utf-8") as _f:
    SPECS = json.load(_f)

# ── Model classes (same as your code) ──────────────────────────
class RouterNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.backbone(x)

class EVA02Classifier(nn.Module):
    def __init__(self, num_classes, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            SPECIALIST_MODEL_NAME, pretrained=False, num_classes=0, drop_rate=0.0)
        with torch.no_grad():
            feat_dim = self.backbone(
                torch.randn(1, 3, SPECIALIST_IMAGE_SIZE, SPECIALIST_IMAGE_SIZE)
            ).shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512), nn.GELU(),
            nn.LayerNorm(512), nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# ── Transforms ────────────────────────────────────────────────
router_tf = transforms.Compose([
    transforms.Resize((ROUTER_IMAGE_SIZE, ROUTER_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
NORMALIZE = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
base_tf = transforms.Compose([
    transforms.Resize((SPECIALIST_IMAGE_SIZE, SPECIALIST_IMAGE_SIZE),
                       interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(), NORMALIZE,
])
aug_tf = transforms.Compose([
    transforms.Resize((int(SPECIALIST_IMAGE_SIZE*1.15),)*2,
                       interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(SPECIALIST_IMAGE_SIZE, scale=(0.85,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), NORMALIZE,
])

# ── Cache ────────────────────────────────────────────────────
router_cache     = {"model": None, "classes": None}
specialist_cache = {}

def load_router():
    if router_cache["model"] is not None:
        return router_cache["model"], router_cache["classes"]
    path  = hf_hub_download(repo_id=HF_REPO, filename=ROUTER_FILE, token=HF_TOKEN)
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    cls   = ckpt["class_names"]
    model = RouterNet(ckpt.get("model_name", ROUTER_MODEL_NAME), num_classes=len(cls))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    router_cache.update({"model": model, "classes": cls})
    del ckpt; gc.collect()
    return model, cls

def load_specialist(category):
    if category in specialist_cache:
        return specialist_cache[category]["model"], specialist_cache[category]["classes"]
    if len(specialist_cache) >= 2:
        oldest = next(iter(specialist_cache))
        del specialist_cache[oldest]["model"]
        del specialist_cache[oldest]
        torch.cuda.empty_cache(); gc.collect()
    hf_file = CATEGORY_TO_SPECIALIST[category]
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=hf_file, token=HF_TOKEN)
    except Exception:
        hf_file = CATEGORY_FALLBACK[category]
        path = hf_hub_download(repo_id=HF_REPO, filename=hf_file, token=HF_TOKEN)
    ckpt    = torch.load(path, map_location="cpu", weights_only=False)
    classes = ckpt["class_names"]
    model   = EVA02Classifier(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model   = model.to(DEVICE).eval()
    specialist_cache[category] = {"model": model, "classes": classes}
    del ckpt; gc.collect(); torch.cuda.empty_cache()
    return model, classes

@torch.no_grad()
def predict_tta(model, img_pil, n=5):
    total = None
    for i in range(n):
        x = (base_tf if i == 0 else aug_tf)(img_pil).unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda") if USE_AMP else torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu()
        total = probs if total is None else total + probs
    return total / n

def _is_uav(name):
    return any(u in name.lower() for u in UAV_CLASS_NAMES)

@torch.no_grad()
def run_pipeline(img_pil):
    # Router
    router_model, router_classes = load_router()
    x = router_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda") if USE_AMP else torch.no_grad():
        rp = torch.softmax(router_model(x), dim=1)[0].cpu()
    r_conf, r_idx = rp.max(0)
    r_cat  = router_classes[r_idx.item()]
    r_pct  = r_conf.item() * 100

    # Primary specialist
    pm, pc = load_specialist(r_cat)
    pprobs = predict_tta(pm, img_pil)
    ptop   = pprobs.argmax().item()
    ptname = pc[ptop]
    ptconf = pprobs[ptop].item() * 100

    # Challenger
    challenger_info = None
    final_cat = r_cat
    final_probs  = pprobs
    final_classes = pc

    trigger_b = (r_cat == "commercial_regional" and _is_uav(ptname))
    trigger_a  = (r_cat in CHALLENGER_MAP and r_pct < ROUTER_CHALLENGER_THRESHOLD)

    if trigger_a or trigger_b:
        chal_cat = "helicopters_drones" if trigger_b else CHALLENGER_MAP[r_cat]
        reason = f"UAV name '{ptname}'" if trigger_b else f"Router conf {r_pct:.1f}%"
        cm, cc = load_specialist(chal_cat)
        cprobs = predict_tta(cm, img_pil)
        ctop   = cprobs.argmax().item()
        ctname = cc[ctop]
        ctconf = cprobs[ctop].item() * 100

        if ctconf > ptconf:
            final_cat     = chal_cat
            final_probs   = cprobs
            final_classes = cc
            challenger_info = {
                "triggered": True,
                "won": True,
                "reason": reason,
                "winner":  f"{CATEGORY_DISPLAY[chal_cat]} ({ctconf:.1f}%)",
                "loser":   f"{CATEGORY_DISPLAY[r_cat]} ({ptconf:.1f}%)",
            }
        else:
            challenger_info = {
                "triggered": True,
                "won": False,
                "reason": reason,
                "winner":  f"{CATEGORY_DISPLAY[r_cat]} ({ptconf:.1f}%)",
                "loser":   f"{CATEGORY_DISPLAY[chal_cat]} ({ctconf:.1f}%)",
            }

    top5v, top5i = final_probs.topk(min(5, len(final_classes)))
    top1_name = final_classes[top5i[0].item()]
    top1_conf = top5v[0].item() * 100

    router_scores = [
        {"category": CATEGORY_DISPLAY.get(router_classes[i], router_classes[i]),
         "conf": round(rp[i].item() * 100, 1),
         "selected": i == r_idx.item()}
        for i in range(len(router_classes))
    ]

    specs = SPECS.get(top1_name, {})

    return {
        "name":          top1_name,
        "confidence":    round(top1_conf, 1),
        "low_conf":      top1_conf < LOW_CONF_THRESHOLD,
        "category":      CATEGORY_DISPLAY.get(final_cat, final_cat),
        "category_acc":  CATEGORY_ACCURACY.get(final_cat, ""),
        "category_icon": CATEGORY_ICON.get(final_cat, "✈"),
        "challenger":    challenger_info,
        "top5": [
            {"name":  final_classes[top5i[i].item()],
             "conf":  round(top5v[i].item() * 100, 1)}
            for i in range(min(5, len(top5i)))
        ],
        "router_scores": router_scores,
        "specs":         specs,
    }

# ── FastAPI app ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading router at startup...")
    load_router()
    print("Ready!")
    yield

app = FastAPI(title="Aircraft ID API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = run_pipeline(img)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
