from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
import numpy as np
import re
import itertools
from createFeaturesSet import create_feature_set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from create_necessary_outputs import (
    create_necessary_outputs_function,
    generate_playlist_feature,
)
from generate_recoSystem_function import generate_playlist_recos
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from preprocess import preprocess
from filter_artists import filter_artists
import os
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.secret_key = "your_secret_key"  


SPOTIPY_CLIENT_ID =  os.getenv("SPOTIFY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET =  os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'
SCOPE = "playlist-modify-public playlist-read-private playlist-read-collaborative user-library-read user-read-playback-state user-read-recently-played user-read-private user-read-email"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True, 
    cache_path=None,  
)


@app.route("/", methods=["GET"])
def index():
    if not session.get("token_info"):
        return redirect(url_for("login"))
    
    try:
        # Check if token needs refresh
        token_info = session.get("token_info")
        if sp_oauth.is_token_expired(token_info):
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session["token_info"] = token_info

        sp = spotipy.Spotify(auth=token_info["access_token"])
        
        # Get playlists
        id_name = get_playlists(sp)
        session["id_name"] = id_name
        
        return render_template("index.html", 
                             playlists=id_name, 
                             recommendations=None,
                             user_name=session.get("user_name"))
    except Exception as e:
        print(f"Error in index: {e}")
        # If there's any error, clear session and redirect to login
        session.clear()
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    # Clear session data
    session.clear()
    cache_file = ".cache"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    return redirect(url_for("login"))

@app.route("/login")
def login():
    # Clear any existing session data
    session.clear()
    
    # Remove cache file if it exists
    cache_file = ".cache"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    auth_url = sp_oauth.get_authorize_url()
    return render_template("login.html", auth_url=auth_url)


@app.route("/callback")
def callback():
    code = request.args.get("code")
    try:
        # Exchange the authorization code for an access token
        token_info = sp_oauth.get_access_token(code)
        session["token_info"] = token_info

        # Get user information right after authentication
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user_info = sp.me()
        
        # Store user information in session
        session["user_id"] = user_info["id"]
        session["user_name"] = user_info["display_name"]
        
        return redirect(url_for("index"))

    except Exception as e:
        print(f"Error in callback: {e}")
        return redirect(url_for("login"))


# ======== PRE-LOAD DATA HERE ========
# Load once when the app starts
SPOTIFY_DF = pd.read_csv("tracks.csv")
DATA_W_GENRE = filter_artists(
    pd.read_csv("artists.csv"),
    target_genres=[
        "classical", "jazz", "reggae", "rock", "pop",
        "electronics", "hip_hop", "hip hop", "rap"
    ]
)
PREPROCESSED_DF, FLOAT_COLS = preprocess(SPOTIFY_DF, DATA_W_GENRE)

# ====================================

@app.route("/recommendations", methods=["POST"])
def recommendations():
    token_info = session.get("token_info")
    sp = spotipy.Spotify(
        auth=token_info["access_token"],
        requests_timeout=20,
        retries=3
    )
    id_name = session.get("id_name")

    # Get the selected playlist and number of recommendations
    selected_playlist_id = request.form.get("playlist")
    num_recommendations = int(request.form.get("num_recommendations", 20))  # Default to 20 if not specified
    
    # Ensure num_recommendations is within bounds (changed max to 150)
    num_recommendations = max(1, min(200, num_recommendations))

    selected_playlist_name = [
        name for name, id in id_name.items() if id == selected_playlist_id
    ][0]

    # Preprocessing
    spotify_df = SPOTIFY_DF.copy()
    data_w_genre = DATA_W_GENRE.copy()
    # spotify_df, float_cols = preprocess(spotify_df, data_w_genre)
    spotify_df = PREPROCESSED_DF.copy()
    float_cols = FLOAT_COLS.copy()

    # Extract playlist and generate recommendations
    current_playlist, filtered_playlist = create_necessary_outputs_function(
        selected_playlist_name, id_name, spotify_df, sp
    )

    # Feature Extraction
    complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)

    # Summarize playlist into a single vector
    complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM = generate_playlist_feature(
        complete_feature_set, filtered_playlist, 1.09
    )

    # Generate recommendations
    edm_top40 = generate_playlist_recos(
        spotify_df,
        complete_feature_set_playlist_vector_EDM,
        complete_feature_set_nonplaylist_EDM,
        sp,
        num_recommendations
    )

    # After generating recommendations (edm_top40)
    # Create a temporary playlist dataframe for the recommended songs
    recommended_playlist_df = edm_top40[['id']].copy()
    recommended_playlist_df['date_added'] = pd.to_datetime('today')  # Add dummy date

    # Generate summary for the recommended playlist
    recommended_complete_feature_set_playlist_vector_EDM, recommended_complete_feature_set_nonplaylist_EDM = generate_playlist_feature(
        complete_feature_set, 
        recommended_playlist_df,  # Use the recommended songs
        1.09
    )

    recommendations = edm_top40.to_dict(orient="records")
    current_playlist_tracks = current_playlist.to_dict(orient="records")

    track_ids = edm_top40["id"].tolist()
    session["tracks"] = track_ids

    return render_template(
        "index.html", 
        playlists=id_name, 
        current_playlist=current_playlist_tracks,
        recommendations=recommendations,
        complete_feature_set_playlist_vector_EDM=complete_feature_set_playlist_vector_EDM[complete_feature_set_playlist_vector_EDM != 0],
        recommended_complete_feature_set_playlist_vector_EDM=recommended_complete_feature_set_playlist_vector_EDM[recommended_complete_feature_set_playlist_vector_EDM != 0]
    )


def get_playlists(sp):
    try:
        playlists = sp.current_user_playlists()
        print(f"Playlists response: {playlists}")
        #Filter out any None items: BUG fix
        id_name = {
            item["name"]: item["id"] for item in playlists["items"] if item is not None
        }
        return id_name
    except Exception as e:
        print(f"Error in get_playlists: {e}")
        return {}



@app.route("/save_playlist", methods=["POST"])
def save_playlist():
    #boilerPlate, get tokenInfo
    token_info = session.get("token_info")
    sp = spotipy.Spotify(auth=token_info["access_token"])

    #Retrieve the playlist name and track IDs
    playlist_name = request.form.get("playlist_name")
    track_ids = session.get("tracks") #FROM /recommend

    if not track_ids:
        return "No tracks to save. Please generate recommendations first.", 400

    # Create the playlist and add tracks
    try:
        user_id = sp.me()["id"]
            #create blank playlist here
        playlist = sp.user_playlist_create(user_id, name=playlist_name, public=True)

        sp.user_playlist_add_tracks(user_id, playlist["id"], track_ids)
        return redirect(url_for("index"))
    except Exception as e:
        print(f"Error creating playlist: {e}")
        return "An error occurred while saving the playlist.", 500


def create_and_add_to_playlist(sp, edm_top40, playlist_name):
    user_id = sp.me()["id"]

    #create blank playlist here
    playlist = sp.user_playlist_create(user_id, name=playlist_name, public=True)

    #to list 
    track_ids = edm_top40["id"].tolist()

    #input the listed songs
    sp.user_playlist_add_tracks(user_id, playlist["id"], track_ids)


@app.route("/get_playlist_tracks/<playlist_id>")
def get_playlist_tracks(playlist_id):
    try:
        token_info = session.get("token_info")
        sp = spotipy.Spotify(auth=token_info["access_token"])
        id_name = session.get("id_name")
        
        # Get playlist name from id
        selected_playlist_name = [name for name, id in id_name.items() if id == playlist_id][0]
        
        # Get playlist tracks
        current_playlist, _ = create_necessary_outputs_function(
            selected_playlist_name, id_name, PREPROCESSED_DF, sp
        )
        
        return jsonify({
            'tracks': current_playlist.to_dict(orient="records"),
            'playlist_name': selected_playlist_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)