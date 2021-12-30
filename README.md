# spotify-mood-prediction

This project predicts what your mood is based on the music you are listening to. It uses Spotify's API to pull attribute data from the songs.
You input a playlist from Spotify, and a K-Means clustering algorithm groups the songs into categories. I then name the categories to relate them to real moods.
Next, new songs are classified using K Nearest Neighbors, and related to one of the mood categories. Your current mood is therefore predicted by the song you are listening to.
