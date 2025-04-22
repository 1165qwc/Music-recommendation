import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import urllib.parse
import requests
import json
from sklearn.preprocessing import MinMaxScaler
import traceback


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
        original_len = len(df)

        num_duplicates_before = df.duplicated().sum()
        st.sidebar.write(f"🔁 Duplicates before cleaning: {num_duplicates_before}")

        df = df.dropna()
        df = df.drop_duplicates()

        cleaned_len = len(df)
        st.sidebar.write(f"🧼 Cleaned dataset: {cleaned_len} songs")
        st.sidebar.write(f"🚮 Removed rows during cleaning: {original_len - cleaned_len}")

        df = df.reset_index(drop=True)

        df['genres'] = df['genre'].str.split(',').apply(lambda x: [g.strip() for g in x])
        all_genres = sorted(list(set([genre for genres in df['genres'] for genre in genres])))

        df = pd.get_dummies(df, columns=['mode'], drop_first=True)

        feature_list = ['danceability', 'energy', 'key', 'loudness',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']

        if not all(feature in df.columns for feature in feature_list):
            st.error("Error: Not all required features are present in the DataFrame.")
            return None

        df_features = df[feature_list]
        scaler = MinMaxScaler()
        df_features = pd.DataFrame(scaler.fit_transform(df_features), columns=feature_list)

        return df, df_features, all_genres

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
    query = f"{song_name} {artist_name} official audio"
    encoded_query = urllib.parse.quote(query)
    return f"https://music.youtube.com/search?q={encoded_query}"


def get_itunes_artwork(song_name, artist_name):
    try:
        query = f"{song_name} {artist_name}"
        encoded_query = urllib.parse.quote(query)
        url = f"https://itunes.apple.com/search?term={encoded_query}&media=music&entity=song&limit=1"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = json.loads(response.text)
            if data["resultCount"] > 0:
                artwork_url = data["results"][0]["artworkUrl100"].replace('100x100', '300x300')
                return artwork_url

        return "https://img.icons8.com/fluency/96/000000/musical-notes.png"
    except Exception as e:
        return "https://img.icons8.com/fluency/96/000000/musical-notes.png"


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

        selected_song_index = matching_songs.index[0]

        if selected_song_index >= similarity_matrix.shape[0]:
            st.error(f"Index error: Selected song index {selected_song_index} is out of bounds")
            return None

        selected_song = matching_songs.iloc[0]['song']
        selected_artist = matching_songs.iloc[0]['artist']

        similarity_scores = similarity_matrix[selected_song_index]
        similar_song_indices = np.argsort(similarity_scores)[::-1]
        similar_song_indices = [idx for idx in similar_song_indices if idx != selected_song_index and idx < len(df)]

        top_n_indices = similar_song_indices[:min(num_recommendations, len(similar_song_indices))]

        selected_artwork = get_itunes_artwork(selected_song, selected_artist)
        selected_yt_link = get_youtube_search_url(selected_song, selected_artist)

        st.sidebar.write("Debug: Similarity score range")
        unique_scores = sorted(set([similarity_scores[idx] for idx in top_n_indices]))
        st.sidebar.write(f"Min: {min(unique_scores):.4f}, Max: {max(unique_scores):.4f}")
        st.sidebar.write(f"Unique scores: {len(unique_scores)}")

        recommendations = []
        for idx in top_n_indices:
            rec_song = df.iloc[idx]['song']
            rec_artist = df.iloc[idx]['artist']
            artwork_url = get_itunes_artwork(rec_song, rec_artist)
            yt_link = get_youtube_search_url(rec_song, rec_artist)
            score = similarity_scores[idx]
            recommendations.append({
                'song': rec_song,
                'artist': rec_artist,
                'youtube_link': yt_link,
                'artwork_url': artwork_url,
                'similarity_score': round(float(score), 4),
                'raw_score': score
            })

        return {
            'selected': {
                'song': selected_song,
                'artist': selected_artist,
                'youtube_link': selected_yt_link,
                'artwork_url': selected_artwork
            },
            'recommendations': recommendations
        }
    except Exception as e:
        st.error(f"Error recommending songs: {e}")
        st.error(traceback.format_exc())
        return None


def get_popular_songs_by_genre(df, genre, num_recommendations=10):
    try:
        genre_songs = df[df['genres'].apply(lambda x: genre in x)].copy()

        if genre_songs.empty:
            st.warning(f"No songs found for genre: {genre}")
            return None

        scaler = MinMaxScaler()
        popularity_features = ['popularity', 'danceability', 'energy', 'valence']

        if not all(feature in genre_songs.columns for feature in popularity_features):
            st.error(f"Missing required features for popularity calculation")
            return None

        normalized_features = scaler.fit_transform(genre_songs[popularity_features])
        weights = [0.4, 0.2, 0.2, 0.2]
        popularity_scores = np.sum(normalized_features * weights, axis=1)
        genre_songs['popularity_score'] = popularity_scores

        top_songs = genre_songs.nlargest(num_recommendations, 'popularity_score')

        recommendations = []
        for _, song in top_songs.iterrows():
            song_name = song['song']
            artist_name = song['artist']
            artwork_url = get_itunes_artwork(song_name, artist_name)
            yt_link = get_youtube_search_url(song_name, artist_name)

            recommendations.append({
                'song': song_name,
                'artist': artist_name,
                'youtube_link': yt_link,
                'artwork_url': artwork_url,
                'popularity_score': round(float(song['popularity_score']), 4),
                'genre': genre
            })

        return recommendations

    except Exception as e:
        st.error(f"Error getting popular songs by genre: {e}")
        st.error(traceback.format_exc())
        return None


def main():
    st.title("Music Recommendation System")
    st.write("This app recommends similar songs based on your input!")

    file_path = 'songs_normalize.csv'
    df = load_data(file_path)
    if df is None:
        return

    st.sidebar.subheader("Dataset Information")
    st.sidebar.write(f"Number of songs: {len(df)}")
    st.sidebar.write(f"Dataset shape: {df.shape}")

    with st.spinner("Preprocessing data..."):
        result = preprocess_data(df)
        if result is None:
            return
        df, df_features, all_genres = result
        st.sidebar.write(f"After preprocessing: {len(df)} songs")

        similarity_matrix = calculate_similarity(df_features)
        if similarity_matrix is None:
            return

        st.sidebar.write(f"Similarity matrix shape: {similarity_matrix.shape}")

    tab1, tab2 = st.tabs(["Similar Songs", "Popular by Genre"])

    with tab1:
        st.subheader("Find Similar Songs")
        song_query = st.text_input("Enter a song name:")

        if song_query:
            matching_titles = df[df['song'].str.lower().str.startswith(song_query.lower())]['song'].unique()
            matching_titles = sorted(matching_titles)[:10]

            if matching_titles.any():
                st.markdown("**Suggestions:**")
                for title in matching_titles:
                    st.markdown(f"- {title}")

            artist_name = st.text_input("Enter artist name (optional):")
            num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

            if st.button("Get Similar Songs"):
                if song_query:
                    with st.spinner("Finding recommendations and artwork..."):
                        result = recommend_songs(song_query, artist_name, df, similarity_matrix, num_recommendations)

                    if result:
                        selected = result['selected']
                        recommendations = result['recommendations']

                        st.subheader("Selected Song")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(selected['artwork_url'], width=150)
                        with col2:
                            st.markdown(f"## {selected['song']} by {selected['artist']}")
                            st.markdown(f"[Listen on YouTube Music]({selected['youtube_link']})")

                        st.markdown("---")
                        st.subheader("Recommended Songs")

                        st.markdown("""
                        <style>
                        .song-card {
                            background-color: #f8f9fa;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        cols = st.columns(2)
                        for i, rec in enumerate(recommendations):
                            with cols[i % 2]:
                                st.markdown("<div class='song-card'>", unsafe_allow_html=True)
                                st.image(rec['artwork_url'], width=150)
                                st.markdown(f"### {rec['song']}")
                                st.markdown(f"**Artist:** {rec['artist']}")
                                st.markdown(f"[Listen on YouTube Music]({rec['youtube_link']})")
                                st.markdown("</div>", unsafe_allow_html=True)
                                st.markdown(f"**Similarity Score:** {rec['similarity_score']:.5f}")
                else:
                    st.warning("Please enter a song name.")

    with tab2:
        st.subheader("Popular Songs by Genre")
        selected_genre = st.selectbox("Select a genre:", ["All Genres"] + all_genres)
        num_genre_recommendations = st.slider("Number of recommendations:", 5, 20, 10, key="genre_slider")

        if st.button("Get Popular Songs"):
            with st.spinner("Finding popular songs..."):
                if selected_genre == "All Genres":
                    all_recommendations = []
                    for genre in all_genres:
                        genre_recs = get_popular_songs_by_genre(df, genre, num_genre_recommendations // len(all_genres) + 1)
                        if genre_recs:
                            all_recommendations.extend(genre_recs)

                    all_recommendations.sort(key=lambda x: x['popularity_score'], reverse=True)
                    recommendations = all_recommendations[:num_genre_recommendations]
                else:
                    recommendations = get_popular_songs_by_genre(df, selected_genre, num_genre_recommendations)

                if recommendations:
                    st.subheader(f"Popular {selected_genre} Songs")
                    cols = st.columns(2)
                    for i, rec in enumerate(recommendations):
                        with cols[i % 2]:
                            st.markdown("<div class='song-card'>", unsafe_allow_html=True)
                            st.image(rec['artwork_url'], width=150)
                            st.markdown(f"### {rec['song']}")
                            st.markdown(f"**Artist:** {rec['artist']}")
                            st.markdown(f"**Genre:** {rec['genre']}")
                            st.markdown(f"[Listen on YouTube Music]({rec['youtube_link']})")
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown(f"**Popularity Score:** {rec['popularity_score']:.5f}")
                else:
                    st.warning(f"No recommendations found for {selected_genre}.")


if __name__ == "__main__":
    main()
