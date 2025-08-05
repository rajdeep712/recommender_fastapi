from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import requests
import io
import numpy as np

app = FastAPI()

# Replace the following URLs with your actual Hugging Face direct URLs:
NN_MODEL_URL = "https://huggingface.co/rajdeeppa53/recommender-files/resolve/main/nn_model.pkl"
MOVIES_DF_URL = "https://huggingface.co/rajdeeppa53/recommender-files/resolve/main/dataframe.pkl"
REDUCED_MATRIX_URL = "https://huggingface.co/rajdeeppa53/recommender-files/resolve/main/reduced_matrix.pkl"

def load_joblib_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    # Load the joblib object from the in-memory bytes
    return pickle.load(io.BytesIO(response.content))

# Global variables for the preloaded data
nn_model = None           # Your NearestNeighbors model
movies_df = None          # DataFrame with at least a "code" column
reduced_matrix = None     # Array with low-dimensional representations

@app.on_event("startup")
def startup_event():
    global nn_model, movies_df, reduced_matrix
    print("Loading nn_model from Hugging Face...")
    nn_model = load_joblib_from_url(NN_MODEL_URL)
    print("Loading movies_df from Hugging Face...")
    movies_df = load_joblib_from_url(MOVIES_DF_URL)
    print("Loading reduced_matrix from Hugging Face...")
    reduced_matrix = load_joblib_from_url(REDUCED_MATRIX_URL)
    print("All data loaded successfully.")

# Request body model
class MovieRequest(BaseModel):
    fav_movies: List[str]

@app.post("/recommend")
def recommend_movies(request: MovieRequest):
    fav_codes = request.fav_movies
    if not fav_codes:
        raise HTTPException(status_code=400, detail="No favorite movie codes provided.")

    # Find indices in movies_df for each favorite code.
    indices = []
    for code in fav_codes:
        idx_series = movies_df[movies_df['code'] == code].index
        if not idx_series.empty:
            indices.append(idx_series[0])
    
    if not indices:
        raise HTTPException(status_code=404, detail="None of the provided movie codes were found.")

    # Compute the average vector from the reduced matrix corresponding to the favorite movies.
    avg_vector = reduced_matrix[indices].mean(axis=0).reshape(1, -1)

    # Query the nn_model for neighbors.
    # Here, we ask for (top_n + len(fav_codes)) neighbors to ensure that if some favorites are returned, we can skip them.
    top_n = 200
    distances, neighbor_indices = nn_model.kneighbors(avg_vector, n_neighbors=top_n + len(indices))
    neighbor_indices = neighbor_indices[0]

    # Build the recommendations list (exclude input movie codes).
    recommended = []
    for idx in neighbor_indices:
        candidate_code = movies_df.iloc[idx]['code']
        if candidate_code not in fav_codes:
            recommended.append(candidate_code)
        if len(recommended) >= top_n:
            break

    return {"recommended_movies": recommended}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
