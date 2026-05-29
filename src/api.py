"""
Movie Recommender REST API
Run with command:  uvicorn api:app --reload
"""
import os
# Fix: Anaconda bundles its own OpenMP runtime which conflicts with PyTorch's on Windows.
# This env var must be set BEFORE torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# ── Constants ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TOP_K = 10
MIN_RATINGS = 5

# ── Pydantic request / response models ───────────────────────────────────────
class RatingRequest(BaseModel):
    # Key (str) represents movie_id, value (float) represents movie rating
    ratings: Dict[str, float]

    @field_validator("ratings")
    @classmethod
    def check_ratings(cls, v):
        if len(v) < MIN_RATINGS: # Check if ratings dictionary contains at least `MIN_RATINGS`` movies
            raise ValueError(f"Please rate at least {MIN_RATINGS} movies.")
        return v

class MovieItem(BaseModel):
    movie_id: int
    title: str

class RecommendationItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]

# ── App state (loaded once at startup) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts into app.state once at server startup."""
    print("Loading artifacts...")

    with open(ARTIFACTS_DIR / "item_to_idx.pkl", "rb") as f:
        app.state.item_to_idx = pickle.load(f)          # movie_id (int) → model_idx

    with open(ARTIFACTS_DIR / "idx_to_item.pkl", "rb") as f:
        app.state.idx_to_item = pickle.load(f)          # model_idx → movie_id (int)

    with open(ARTIFACTS_DIR / "movie_id_to_title.pkl", "rb") as f:
        app.state.movie_id_to_title = pickle.load(f)    # movie_id (int) → title str

    # item_embeddings: numpy array shape (num_items, 64)
    item_emb_tensor = torch.load(ARTIFACTS_DIR / "item_embeddings.pt", map_location="cpu")
    app.state.item_embeddings = item_emb_tensor.numpy()  # keep as numpy for fast matmul

    print(f"  Loaded {len(app.state.item_to_idx)} movies, "
          f"embedding dim={app.state.item_embeddings.shape[1]}")
    print("API ready.")
    yield  # server runs here
    print("Shutting down.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommender API",
    description="Trained Two-Tower model with cold-start embedding averaging.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (the HTML UI) from ./static/
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the favicon.ico directly to the url."""
    return FileResponse(STATIC_DIR / "favicon.png")


@app.get("/movies", response_model=list[MovieItem])
async def get_movies():
    """Return all movies available for rating, sorted alphabetically."""
    movies = [
        MovieItem(movie_id=mid, title=title)
        for mid, title in app.state.movie_id_to_title.items()
        if mid in app.state.item_to_idx            # only movies the model knows
    ]
    movies.sort(key=lambda m: m.title)
    return movies


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RatingRequest):
    """
    Accept a dict of {movie_id: rating} pairs (min 5) and return
    the top-10 personalised movie recommendations.
    """
    item_to_idx: dict = app.state.item_to_idx
    idx_to_item: dict = app.state.idx_to_item
    movie_id_to_title: dict = app.state.movie_id_to_title
    item_embeddings: np.ndarray = app.state.item_embeddings  # (num_items, 64)

    # Map submitted movie IDs → model indices, skip unknowns
    rated_indices = []
    rating_weights = []
    for movie_id_str, rating in request.ratings.items():
        movie_id = int(movie_id_str)
        if movie_id in item_to_idx:
            rated_indices.append(item_to_idx[movie_id])
            rating_weights.append(float(rating))

    if len(rated_indices) < MIN_RATINGS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_RATINGS} of the submitted movies must be in the dataset."
        )

    # ── Cold-start: compute user embeddings as weighted average of rated item embeddings
    weights = np.array(rating_weights, dtype=np.float32)
    weights /= weights.sum()                              # normalise weights

    rated_embs = item_embeddings[rated_indices]           # (num_rated, 64)
    user_embeddings = (weights[:, None] * rated_embs).sum(axis=0)  # (64,)

    # ── Score all items
    scores = item_embeddings @ user_embeddings             # (num_items,) dot product

    # ── Mask out movies the user has already rated
    scores[rated_indices] = -np.inf

    # ── Rank and return top-10
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    recommendations = []
    for rank, model_idx in enumerate(top_indices, start=1):
        movie_id = idx_to_item[int(model_idx)]
        title = movie_id_to_title.get(movie_id, f"Movie {movie_id}")
        recommendations.append(RecommendationItem(
            rank=rank,
            movie_id=movie_id,
            title=title,
            score=round(float(scores[model_idx]), 4),
        ))

    return RecommendResponse(recommendations=recommendations)
