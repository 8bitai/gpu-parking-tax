import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import data_loader
from api.routers import (
    anomalies,
    correlations,
    distributions,
    gpus,
    overview,
    profiling,
    temporal,
    workloads,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_loader.load_all()
    yield


app = FastAPI(title="GPU Telemetry EDA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview.router, prefix="/api/overview", tags=["overview"])
app.include_router(workloads.router, prefix="/api/workloads", tags=["workloads"])
app.include_router(gpus.router, prefix="/api/gpus", tags=["gpus"])
app.include_router(temporal.router, prefix="/api/temporal", tags=["temporal"])
app.include_router(correlations.router, prefix="/api/correlations", tags=["correlations"])
app.include_router(distributions.router, prefix="/api/distributions", tags=["distributions"])
app.include_router(anomalies.router, prefix="/api/anomalies", tags=["anomalies"])
app.include_router(profiling.router, prefix="/api/profiling", tags=["profiling"])
