
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database import Base, engine
from app import models
from app.routers import users, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AttendAI Lightweight API",
    version="1.0.0",
    description="Lightweight online prototype for AttendAI",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(sessions.router)


@app.get("/api/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "service": "AttendAI Lightweight API",
        "mode": "prototype",
        "face_recognition": False,
    }


@app.get("/", tags=["health"])
def root():
    return {
        "message": "AttendAI API is running",
        "docs": "/docs",
    }
