from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# API id and secret
clientId = "224f76769eaf40f8b2276cc315f4dad3"
secret = "15f09542e9de45dd881c71acb9eaf6f7"

# Set up spotify object
client_credentials_manager = SpotifyClientCredentials(client_id=clientId, client_secret=secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# get track ids from playlist
def getPlaylistTrackIDs(user, playlist_id):
    ids = []
    playlist = spotify.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids


# get songs from a playlist
song_features = []
ids = getPlaylistTrackIDs('benovard', '2cpFKC4U25CAXegtW9SJvI')
for id in ids:
    track = spotify.track(id)
    track_features = spotify.audio_features(id)[0]
    track_features['title'] = track['name']
    track_features['artist'] = track['artists'][0]['name']
    song_features.append(track_features)

# convert to pandas dataframe
df = pd.DataFrame(song_features)
df = df.drop(['type', 'uri', 'track_href', 'analysis_url'], axis=1)

# scale attributes between 0 and 1
scaler = preprocessing.MinMaxScaler()
df['key'] = pd.DataFrame(scaler.fit_transform(df[['key']].values))
df['loudness'] = pd.DataFrame(scaler.fit_transform(df[['loudness']].values))
df['tempo'] = pd.DataFrame(scaler.fit_transform(df[['tempo']].values))
df['duration_ms'] = pd.DataFrame(scaler.fit_transform(df[['duration_ms']].values))
df['time_signature'] = pd.DataFrame(scaler.fit_transform(df[['time_signature']].values))
