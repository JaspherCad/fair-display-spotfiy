import pandas as pd



def create_necessary_outputs_function(playlist_name, id_dic, df, sp):
    """
    Pull songs from a specific playlist.
    Returns both the complete playlist and filtered playlist (songs available in dataset)
    """
    # Generate playlist dataframe
    playlist = pd.DataFrame()
    
    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        if i['track']: # Check if track exists
            playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
            playlist.loc[ix, 'name'] = i['track']['name']
            playlist.loc[ix, 'id'] = i['track']['id']
            playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
            playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])
    
    # Create two versions: complete and filtered
    complete_playlist = playlist.copy()
    filtered_playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added', ascending=False)
    
    return complete_playlist, filtered_playlist


def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
        """ 
        Summarize a user's playlist into a single vector

        Parameters: 
            complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
            playlist_df (pandas dataframe): playlist dataframe
            weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
            
        Returns: 
            playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
            complete_feature_set_nonplaylist (pandas dataframe): 
        """
        
        complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
        complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
        complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
        
        playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

        most_recent_date = playlist_feature_set.iloc[0,-1]
        
        for ix, row in playlist_feature_set.iterrows():
            playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
            
        playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
        
        playlist_feature_set_weighted = playlist_feature_set.copy()
        #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
        playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
        playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
        #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
        
        return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist