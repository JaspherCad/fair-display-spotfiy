# Spotify Recommendation System

A personalized music recommendation engine that analyzes your Spotify listening habits and suggests new tracks you'll actually enjoy. Built with Python, Flask, and the Spotify API.

##  Video Demos

**Full Demo + Backend Explained (2.5 min):**  
Watch how the recommendation system works under the hood  
https://www.youtube.com/watch?v=R4kFx7N2gXg

**Quick Demo (1 min):**  
See the app in action without the technical details  
https://www.youtube.com/watch?v=E_4ve8gynhI

## What It Does

This app connects to your Spotify account, analyzes your playlists and listening history, then recommends songs based on audio features like tempo, energy, danceability, and more. Instead of just looking at genres or what's popular, it digs into the actual characteristics of music you like.

The recommendation engine uses:
- **Audio feature analysis** - Breaking down tracks by tempo, key, loudness, acousticness, etc.
- **TF-IDF vectorization** - Smart text analysis for artist and genre information
- **Cosine similarity** - Finding songs that match your taste profile
- **Recency weighting** - Prioritizing your recent listening habits

## Getting Started

### Prerequisites

- Python 3.7 or higher
- A Spotify account (free or premium)
- Spotify Developer credentials (Client ID & Secret)

### Installation

1. Clone this repo:
```bash
git clone <your-repo-url>
cd spotifyReco
```

2. Install dependencies:
```bash
pip install flask spotipy pandas numpy scikit-learn matplotlib
```

3. Set up your Spotify API credentials:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy your Client ID and Client Secret
   - Add `http://localhost:5000/callback` to your Redirect URIs
   - Update the credentials in [app.py](app.py)

4. Run the app:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## How It Works

1. **Login** - Authenticate with your Spotify account
2. **Data Collection** - The app pulls your playlists and listening history
3. **Feature Extraction** - Each song is analyzed for its audio characteristics
4. **Profile Building** - Your listening habits are summarized into a taste profile
5. **Recommendation** - The system finds similar songs you haven't heard yet

The backend processes hundreds of audio features per track, normalizes the data, and uses machine learning to find patterns in what you like. Check out the video above for a deeper dive into how the algorithm works.

## Project Structure

```
spotifyReco/
├── app.py                          # Main Flask application
├── recommendation.py               # Core recommendation algorithm
├── createFeaturesSet.py           # Feature extraction logic
├── generate_recoSystem_function.py # Playlist recommendation generator
├── create_necessary_outputs.py    # Helper functions for feature generation
├── preprocess.py                  # Data preprocessing utilities
├── filter_artists.py              # Artist filtering logic
├── spotify_api.py                 # Spotify API wrapper
├── tracks.csv                     # Track dataset
├── artists.csv                    # Artist dataset
├── templates/                     # HTML templates
│   ├── base.html
│   ├── index.html
│   └── login.html
└── static/                        # CSS and static files
    └── css/
        └── styles.css
```

## Features

- **Spotify OAuth Integration** - Secure login with your Spotify account
- **Playlist Analysis** - Analyzes all your playlists and saved songs
- **Smart Recommendations** - ML-powered suggestions based on audio features
- **Recency Bias** - Gives more weight to what you've been listening to lately
- **Artist Filtering** - Option to discover completely new artists or stick with familiar ones

## Technical Notes

The recommendation system uses scikit-learn for ML operations and cosine similarity to measure how close songs are to your taste profile. Audio features are normalized using MinMaxScaler to ensure fair comparison across different metrics.

The weighting factor for recency can be adjusted to control how much your recent listening habits influence recommendations versus your overall taste profile.

## Known Issues

- Sessions are temporary (token caching is disabled for demo purposes)
- Large playlists may take a moment to process
- API rate limits may apply for heavy usage

## Future Improvements

- Save user preferences and recommendation history
- Add mood-based filtering
- Implement collaborative filtering
- Export recommendations as new Spotify playlists
- Add more visualization of your listening patterns

## License

This project is for educational purposes. Make sure to comply with Spotify's terms of service when using their API.

---

Built with  and 
