import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    try:
        df = df.dropna()  # Drop rows with any missing values
        df = df.drop_duplicates() #drop duplicates

        # One-hot encode 'mode' and 'genre'
        df = pd.get_dummies(df, columns=['mode', 'genre'], drop_first=True)

        # Select features for similarity calculation
        feature_list = ['danceability', 'energy', 'key', 'loudness',
                          'speechiness', 'acousticness', 'instrumentalness',
                          'liveness', 'valence', 'tempo']
        if not all(feature in df.columns for feature in feature_list):
            st.error("Error: Not all required features are present in the DataFrame.")
            return None

        df_features = df[feature_list]
        return df, df_features
    except KeyError as e:
        st.error(f"Error: Key not found in DataFrame: {e}")
        return None
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def calculate_similarity(df_features):
    try:
        similarity_matrix = cosine_similarity(df_features)
        return similarity_matrix
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return None

def recommend_songs(song_name, artist_name, df, similarity_matrix, num_recommendations=10):
    try:
        processed_song_name = song_name.lower()
        processed_artist_name = artist_name.lower() if artist_name else ""

        if artist_name:
            comparison_result = (df['song'].str.lower().str.contains(processed_song_name)) & (
                df['artist'].str.lower().str.contains(processed_artist_name)
            )
        else:
            comparison_result = df['song'].str.lower().str.contains(processed_song_name)

        matching_songs = df[comparison_result]

        if matching_songs.empty:
            st.warning(f"Song '{song_name}' by '{artist_name}' not found in the dataset.")
            return None

        # If multiple matches, use the first one.
        selected_song_index = matching_songs.index[0]
        selected_song = matching_songs.iloc[0]['song']
        selected_artist = matching_songs.iloc[0]['artist']

        # Get the similarity scores for the song
        similarity_scores = similarity_matrix[selected_song_index]

        # Sort the songs by similarity (most similar first)
        similar_song_indices = np.argsort(similarity_scores)[::-1]

        # Exclude the input song itself
        similar_song_indices = similar_song_indices[1:]

        # Get the top N recommendations
        top_n_indices = similar_song_indices[:num_recommendations]
        recommended_songs = df.iloc[top_n_indices]['song'].tolist()
        recommended_artists = df.iloc[top_n_indices]['artist'].tolist()
        return selected_song, selected_artist, list(zip(recommended_songs, recommended_artists))
    except Exception as e:
        st.error(f"Error recommending songs: {e}")
        return None

def main():
    st.title("Music Recommendation System")
    st.write("This app recommends similar songs based on your input!")
    
    file_path = 'songs_normalize.csv'
    df = load_data(file_path)
    if df is None:
        return

    df, df_features = preprocess_data(df)
    if df_features is None:
        return

    similarity_matrix = calculate_similarity(df_features)
    if similarity_matrix is None:
        return

    song_name = st.text_input("Enter a song name:")
    artist_name = st.text_input("Enter artist name (optional):")

    if st.button("Get Recommendations"):
        if song_name:
            result = recommend_songs(song_name, artist_name, df, similarity_matrix)
            if result:
                selected_song, selected_artist, recommended_songs = result
                st.subheader(f"Selected Song: {selected_song} by {selected_artist}")
                st.subheader("Recommended Songs:")
                for song, artist in recommended_songs:
                    st.write(f"- {song} by {artist}")
        else:
            st.warning("Please enter a song name.")

if __name__ == "__main__":
    main()
