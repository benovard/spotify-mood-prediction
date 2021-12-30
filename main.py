from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from api_key import ApiKey


# API id and secret
clientId = ApiKey.CLIENT_ID
secret = ApiKey.SECRET

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
songs = []
ids = getPlaylistTrackIDs('benovard', '4ZOvYXdxuxjhxSOsGavzpc')
for id in ids:
    track = spotify.track(id)
    track_features = spotify.audio_features(id)[0]
    track_features['title'] = track['name']
    track_features['artist'] = track['artists'][0]['name']
    songs.append(track_features)

# convert to pandas dataframe
df = pd.DataFrame(songs)
df = df.drop(['type', 'uri', 'track_href', 'analysis_url'], axis=1)

# scale attributes between 0 and 1
scaler = preprocessing.MinMaxScaler()
df['key'] = pd.DataFrame(scaler.fit_transform(df[['key']].values))
df['loudness'] = pd.DataFrame(scaler.fit_transform(df[['loudness']].values))
df['tempo'] = pd.DataFrame(scaler.fit_transform(df[['tempo']].values))
df['duration_ms'] = pd.DataFrame(scaler.fit_transform(df[['duration_ms']].values))
df['time_signature'] = pd.DataFrame(scaler.fit_transform(df[['time_signature']].values))

song_features = df.copy()
song_features = song_features.drop(['title', 'artist', 'id'], axis=1)

# run with up to 15 clusters
SSE = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(song_features)
    SSE.append(km.inertia_)

# plot SSE vs K
plt.plot(K, SSE, 'gx-')
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()

# cluster with kmeans
k_means = KMeans(n_clusters=5)
k_means.fit(song_features)

# dimension reduction with pca
y_kmeans = k_means.predict(song_features)
pca = PCA(n_components=2)
components = pca.fit_transform(song_features)

pc = pd.DataFrame(components)
pc['label'] = y_kmeans
pc.columns = ['x', 'y', 'label']

# plot pca reduction
plt.scatter(pc[pc.label == 0].x, pc[pc.label == 0].y)
plt.scatter(pc[pc.label == 1].x, pc[pc.label == 1].y)
plt.scatter(pc[pc.label == 2].x, pc[pc.label == 2].y)
plt.scatter(pc[pc.label == 3].x, pc[pc.label == 3].y)
plt.scatter(pc[pc.label == 4].x, pc[pc.label == 4].y)
plt.scatter(pc[pc.label == 5].x, pc[pc.label == 5].y)
plt.show()

# add cluster labels to songs
df['cluster'] = y_kmeans

# add labels to clusters
labels = ["Angry", "Sad", "Vibin", "Dancey", "Chill"]

# create train and test sets
x = song_features
y = y_kmeans
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# classify with knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

print(classification_report(y_test, knn_pred, target_names=labels))
