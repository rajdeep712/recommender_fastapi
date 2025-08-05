from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# Load data
try:
    movie_dataset = pickle.load(open("movies_list_upd.pkl", "rb"))
    similarities = pickle.load(open("similarity_upd.pkl", "rb"))
except Exception as e:
    raise RuntimeError("Error loading pickle files: " + str(e))

app = FastAPI()

class RecommendationRequest(BaseModel):
    favorite_movies: list[str]

@app.post("/recommend")
def recommend_movies(data: RecommendationRequest):
    fav_movies = data.favorite_movies
    indices = []

    try:
        for movie in fav_movies:
            index_list = movie_dataset[movie_dataset['code'] == movie].index
            if not index_list.empty:
                indices.append(index_list[0])
            else:
                raise HTTPException(status_code=404, detail=f"Movie '{movie}' not found")
        
        avg_similarity_score = sum(similarities[index] for index in indices) / len(indices)
        sorted_recc_movies_idxs = sorted(
            list(enumerate(avg_similarity_score)),
            key=lambda vector: vector[1],
            reverse=True
        )

        recommended_movies = []
        for movie_idx, similarity_score in sorted_recc_movies_idxs:
            if movie_idx not in indices:
                recommended_movies.append(movie_dataset.iloc[movie_idx, 0])

        return {"recommendations": recommended_movies[:1000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
