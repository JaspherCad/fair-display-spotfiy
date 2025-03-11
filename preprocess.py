import re
import pandas as pd
import numpy as np
from itertools import chain
from functools import lru_cache

# Pre-compiled regex patterns
ARTIST_EXTRACT_V1 = re.compile(r"'([^']*)'")
ARTIST_EXTRACT_V2 = re.compile(r'"(.*?)"')

def preprocess(spotify_df, data_w_genre):
    # Create artist lists
    spotify_df = spotify_df.copy()
    data_w_genre = data_w_genre.copy().drop('id', axis=1)
    
    # Vectorized artist extraction
    spotify_df['artists_upd_v1'] = spotify_df['artists'].str.findall(ARTIST_EXTRACT_V1)
    spotify_df['artists_upd_v2'] = spotify_df['artists'].str.findall(ARTIST_EXTRACT_V2)
    
    # Optimized conditional selection
    spotify_df['artists_upd'] = np.where(
        spotify_df['artists_upd_v1'].str.len() == 0,
        spotify_df['artists_upd_v2'],
        spotify_df['artists_upd_v1']
    )
    
    # Create unique identifier
    spotify_df['name'] = spotify_df['name'].fillna('')
    spotify_df['artists_song'] = spotify_df['artists_upd'].str[0] + spotify_df['name']
    
    # Deduplication
    spotify_df = spotify_df.sort_values(['artists_song', 'release_date'], ascending=False)
    spotify_df = spotify_df.drop_duplicates('artists_song', keep='first')
    
    # Genre processing
    genre_lookup = data_w_genre.set_index('name')['genres_upd'].to_dict()
    artists_exploded = spotify_df[['artists_upd', 'id']].explode('artists_upd')
    artists_exploded['genres'] = artists_exploded['artists_upd'].map(genre_lookup)
    
    # Consolidated genres
    artists_genres_consolidated = (
        artists_exploded[~artists_exploded['genres'].isna()]
        .groupby('id')['genres']
        .agg(list)
        .reset_index()
    )
    artists_genres_consolidated['consolidates_genre_lists'] = [
        list(set(chain.from_iterable(x))) for x in artists_genres_consolidated['genres']
    ]
    
    # Merge back
    spotify_df = spotify_df.merge(
        artists_genres_consolidated[['id', 'consolidates_genre_lists']],
        on='id',
        how='left'
    )
    
    # Feature engineering
    # Fix for year extraction
    spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
    spotify_df['popularity_red'] = spotify_df['popularity'] // 5
    float_cols = spotify_df.select_dtypes(include='float64').columns.tolist()
    
    # Handle genre lists
    spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    return spotify_df, float_cols