# app.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---- Constants ----
MAX_EXPECTED_CYCLES = 7500
MAX_STRAIN = 0.0012231
MAX_STRESS_PA = 250e6
E_STEEL = 210e9
WARNING_ZONE = 7000
BASE_TEMP = 310.0  # 37°C in Kelvin


class BarrelState:

    def __init__(self):
        self.cycles = 0
        self.health = 1.0
        self.initialized = False
        self.expired = False
        self.temp = BASE_TEMP

    def reset(self, start_cycles: int):
        self.cycles = start_cycles
        self.health = max(0.0, 1 - start_cycles / MAX_EXPECTED_CYCLES)
        self.initialized = True
        self.expired = False
        self.temp = BASE_TEMP


barrel_state = BarrelState()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class InitData(BaseModel):
    cycles: int


@app.post("/initialize")
def initialize(data: InitData):
    barrel_state.reset(data.cycles)
    return {"message": "Initialized"}


@app.get("/latest")
def latest():

    if not barrel_state.initialized:
        return JSONResponse({
            "stress": 0.0,
            "temp": BASE_TEMP,
            "strain": 0.0,
            "vibration": 0.0,
            "health": 1.0,
            "cycles": 0,
            "remaining": MAX_EXPECTED_CYCLES,
            "status": "Not Initialized"
        })

    if barrel_state.expired:
        return JSONResponse({
            "stress": 0.0,
            "temp": barrel_state.temp,
            "strain": 0.0,
            "vibration": 0.0,
            "health": 0.0,
            "cycles": barrel_state.cycles,
            "remaining": 0,
            "status": "Expired"
        })

    # ---- Advance Cycle ----
    barrel_state.cycles += 1
    degradation = barrel_state.cycles / MAX_EXPECTED_CYCLES

    # ---- Temperature (monotonic increase) ----
    barrel_state.temp += 0.08 + 0.35 * degradation
    temp = barrel_state.temp

    # ---- Vibration (non-negative) ----
    vibration = abs(np.random.normal(2 + 5 * degradation, 1.3))

    # ---- Stress ----
    raw_stress = vibration * (80e6 + 150e6 * degradation)
    stress_pa = min(raw_stress, MAX_STRESS_PA)
    stress_mpa = stress_pa / 1e6

    # ---- Strain (Hooke's Law) ----
    raw_strain = stress_pa / E_STEEL
    normalized_strain = min(raw_strain / 0.002, 1.0)
    strain = normalized_strain * MAX_STRAIN

    # ---- Health ----
    decay_rate = 0.0004 + 0.0025 * degradation
    barrel_state.health -= decay_rate
    barrel_state.health = max(0.0, barrel_state.health)

    health = barrel_state.health
    remaining = MAX_EXPECTED_CYCLES - barrel_state.cycles

    if barrel_state.cycles >= MAX_EXPECTED_CYCLES:
        barrel_state.expired = True
        status = "Expired"
    elif barrel_state.cycles >= WARNING_ZONE:
        status = "WARNING: Near End-of-Life"
    elif health > 0.6:
        status = "Fully Mission Capable"
    elif health > 0.4:
        status = "Serviceable"
    elif health > 0.2:
        status = "Accuracy Degrading"
    else:
        status = "Critical - Replace Soon"

    return JSONResponse({
        "stress": round(stress_mpa, 2),
        "temp": round(temp, 2),
        "strain": round(strain, 7),
        "vibration": round(vibration, 3),
        "health": round(health, 4),
        "cycles": barrel_state.cycles,
        "remaining": remaining,
        "status": status
    })