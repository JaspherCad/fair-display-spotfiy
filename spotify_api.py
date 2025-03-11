import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
import sys
    
#redirect_uri = 'http://localhost:8881/'

# scope = 'user-library-read'

def authenticate_spotify(client_id, client_secret, scope, redirect_uri ): #return sp (authenticated user)

    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        print("Usage: %s username" % (sys.argv[0],))
        sys.exit()


    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri=redirect_uri)

    sp = spotipy.Spotify(auth=token)
    return sp



def get_playlists(sp):

    #gather playlist names and images. 
    #images aren't going to be used until I start building a UI
    id_name = {}
    list_photo = {}
    for i in sp.current_user_playlists()['items']:

        id_name[i['name']] = i['uri'].split(':')[2]
        list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']

    return id_name
    # strings = id_name CORRECT HERE