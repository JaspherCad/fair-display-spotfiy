import re 


def filter_artists(data_w_genre, target_genres):
    def genre_filter(genres):
        return any(genre in target_genres for genre in genres)
    
    #turn data_w_genre into real arrays
    data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
    
    #apply filter here
    data_w_genre = data_w_genre[data_w_genre['genres_upd'].apply(genre_filter)]
    return data_w_genre