from sklearn.metrics.pairwise import cosine_similarity


def generate_playlist_recos(df, features, nonplaylist_features, sp, num_recommendations=20):
        """ 
        Generate top song recommendations for a specific playlist.

        Parameters: 
            df (pandas DataFrame): Spotify dataset.
            features (pandas Series): Summarized playlist feature vector.
            nonplaylist_features (pandas DataFrame): Feature set of songs not in the selected playlist.
            sp: Spotify client instance
            num_recommendations (int): Number of recommendations to generate (default: 20)
            
        Returns: 
            non_playlist_df_top_40: Top N recommendations for the playlist.
        """
        # Filter the non-playlist songs
        non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)].copy()
        
        # Compute similarity scores
        non_playlist_df['sim'] = cosine_similarity(
            nonplaylist_features.drop('id', axis=1).values, 
            features.values.reshape(1, -1)
        )[:, 0]
        
        # Sort by similarity and get top N
        non_playlist_df_top_n = non_playlist_df.sort_values('sim', ascending=False).head(num_recommendations)
        
        # Add URL column for visualization
        non_playlist_df_top_n['url'] = non_playlist_df_top_n['id'].apply(
            lambda x: sp.track(x)['album']['images'][1]['url']
        )
        
        return non_playlist_df_top_n