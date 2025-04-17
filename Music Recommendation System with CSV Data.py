import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import urllib.parse

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

def get_youtube_search_url(song_name, artist_name):
    """Generate a YouTube Music search URL for a song"""
    query = f"{song_name} {artist_name} official audio"
    encoded_query = urllib.parse.quote(query)
    return f"https://music.youtube.com/search?q={encoded_query}"

def get_music_icon_url(index):
    """Return a music-themed icon URL based on the index"""
    # Use a set of predefined music-themed icons or colors that rotate based on index
    icon_list = [
        "https://img.icons8.com/color/96/000000/musical-notes.png",
        "https://img.icons8.com/color/96/000000/music.png",
        "https://img.icons8.com/color/96/000000/musical.png",
        "https://img.icons8.com/color/96/000000/audio-wave.png",
        "https://img.icons8.com/color/96/000000/electronic-music.png"
    ]
    return icon_list[index % len(icon_list)]

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
        
        recommendations = []
        for i, idx in enumerate(top_n_indices):
            rec_song = df.iloc[idx]['song']
            rec_artist = df.iloc[idx]['artist']
            
            # Generate YouTube Music search link
            yt_link = get_youtube_search_url(rec_song, rec_artist)
            
            # Get a music icon based on index
            icon_url = get_music_icon_url(i)
            
            recommendations.append({
                'song': rec_song,
                'artist': rec_artist,
                'youtube_link': yt_link,
                'icon_url': icon_url,
                'similarity_score': similarity_scores[idx]
            })
        
        # Also get YouTube link for the selected song
        selected_yt_link = get_youtube_search_url(selected_song, selected_artist)
        
        return {
            'selected': {
                'song': selected_song,
                'artist': selected_artist,
                'youtube_link': selected_yt_link,
                'icon_url': get_music_icon_url(0)
            },
            'recommendations': recommendations
        }
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
    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

    if st.button("Get Recommendations"):
        if song_name:
            with st.spinner("Finding recommendations..."):
                result = recommend_songs(song_name, artist_name, df, similarity_matrix, num_recommendations)
                
            if result:
                selected = result['selected']
                recommendations = result['recommendations']
                
                # Display selected song with icon and link
                st.subheader(f"Selected Song: {selected['song']} by {selected['artist']}")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(selected['icon_url'], width=80)
                with col2:
                    st.markdown(f"[Listen on YouTube Music]({selected['youtube_link']})")
                    st.markdown("Your seed song for recommendations")
                
                # Display recommendations
                st.subheader("Recommended Songs:")
                
                # Custom CSS for better card layout
                st.markdown("""
                <style>
                .song-card {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                }
                .song-card img {
                    border-radius: 5px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Use columns to display recommendations in a grid
                cols = st.columns(2)
                for i, rec in enumerate(recommendations):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="song-card">
                            <h3>{rec['song']} by {rec['artist']}</h3>
                            <p>Similarity: {rec['similarity_score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(rec['icon_url'], width=60)
                        st.markdown(f"[Listen on YouTube Music]({rec['youtube_link']})")
                        st.markdown("---")
        else:
            st.warning("Please enter a song name.")

if __name__ == "__main__":
    main()