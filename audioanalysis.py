import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_audio_features(song_data):
    """
    Perform deep analysis of audio features for a song
    """
    try:
        # Extract basic features
        features = {
            'tempo': song_data['tempo'],
            'key': song_data['key'],
            'mode': song_data['mode'],
            'danceability': song_data['danceability'],
            'energy': song_data['energy'],
            'loudness': song_data['loudness'],
            'speechiness': song_data['speechiness'],
            'acousticness': song_data['acousticness'],
            'instrumentalness': song_data['instrumentalness'],
            'liveness': song_data['liveness'],
            'valence': song_data['valence']
        }
        
        # Calculate derived features
        features['rhythm_complexity'] = calculate_rhythm_complexity(song_data)
        features['harmonic_complexity'] = calculate_harmonic_complexity(song_data)
        features['dynamic_range'] = calculate_dynamic_range(song_data)
        
        return features
    except Exception as e:
        st.error(f"Error analyzing audio features: {e}")
        return None

def calculate_rhythm_complexity(song_data):
    """
    Calculate rhythm complexity based on tempo and time signature
    """
    try:
        # Basic rhythm complexity calculation
        tempo = song_data['tempo']
        time_signature = song_data.get('time_signature', 4)  # Default to 4/4
        
        # Higher tempo and more complex time signatures increase complexity
        complexity = (tempo / 120) * (time_signature / 4)
        return min(complexity, 1.0)  # Normalize to 0-1 range
    except Exception as e:
        st.error(f"Error calculating rhythm complexity: {e}")
        return 0.5  # Default value

def calculate_harmonic_complexity(song_data):
    """
    Calculate harmonic complexity based on key, mode, and other features
    """
    try:
        # Combine various features to estimate harmonic complexity
        key = song_data['key']
        mode = song_data['mode']
        acousticness = song_data['acousticness']
        instrumentalness = song_data['instrumentalness']
        
        # More complex harmonies often have:
        # - Minor keys (mode = 0)
        # - Higher instrumentalness
        # - Lower acousticness
        complexity = (
            (1 - mode) * 0.4 +  # Minor keys are more complex
            instrumentalness * 0.4 +
            (1 - acousticness) * 0.2
        )
        return min(complexity, 1.0)  # Normalize to 0-1 range
    except Exception as e:
        st.error(f"Error calculating harmonic complexity: {e}")
        return 0.5  # Default value

def calculate_dynamic_range(song_data):
    """
    Calculate dynamic range based on loudness and energy
    """
    try:
        # Dynamic range is the difference between loudest and quietest parts
        loudness = song_data['loudness']
        energy = song_data['energy']
        
        # Normalize loudness to 0-1 range (assuming -60 to 0 dB range)
        normalized_loudness = (loudness + 60) / 60
        
        # Combine with energy for overall dynamic range
        dynamic_range = (normalized_loudness + energy) / 2
        return min(dynamic_range, 1.0)  # Normalize to 0-1 range
    except Exception as e:
        st.error(f"Error calculating dynamic range: {e}")
        return 0.5  # Default value

def visualize_audio_features(features, song_name):
    """
    Create visualizations for audio features
    """
    try:
        # Create a radar chart for audio features
        categories = ['Tempo', 'Danceability', 'Energy', 'Valence', 
                     'Rhythm Complexity', 'Harmonic Complexity', 'Dynamic Range']
        
        values = [
            features['tempo'] / 200,  # Normalize tempo
            features['danceability'],
            features['energy'],
            features['valence'],
            features['rhythm_complexity'],
            features['harmonic_complexity'],
            features['dynamic_range']
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Close the loop
        angles = np.concatenate((angles, [angles[0]]))  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f'Audio Features Analysis: {song_name}')
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def get_song_characteristics(features):
    """
    Generate human-readable characteristics of the song
    """
    try:
        characteristics = []
        
        # Tempo characteristics
        if features['tempo'] < 60:
            characteristics.append("Slow tempo")
        elif features['tempo'] < 100:
            characteristics.append("Moderate tempo")
        else:
            characteristics.append("Fast tempo")
        
        # Energy level
        if features['energy'] < 0.3:
            characteristics.append("Low energy")
        elif features['energy'] < 0.7:
            characteristics.append("Medium energy")
        else:
            characteristics.append("High energy")
        
        # Mood
        if features['valence'] < 0.3:
            characteristics.append("Sad/melancholic")
        elif features['valence'] < 0.7:
            characteristics.append("Neutral mood")
        else:
            characteristics.append("Happy/upbeat")
        
        # Instrumentation
        if features['instrumentalness'] > 0.7:
            characteristics.append("Mostly instrumental")
        elif features['instrumentalness'] > 0.3:
            characteristics.append("Mix of vocals and instruments")
        else:
            characteristics.append("Vocal-focused")
        
        # Acoustic quality
        if features['acousticness'] > 0.7:
            characteristics.append("Acoustic sound")
        elif features['acousticness'] > 0.3:
            characteristics.append("Mix of acoustic and electronic")
        else:
            characteristics.append("Electronic production")
        
        return characteristics
    except Exception as e:
        st.error(f"Error generating song characteristics: {e}")
        return []

def find_similar_songs_by_audio_features(df, target_features, num_recommendations=5):
    """
    Find similar songs based on audio feature analysis
    """
    try:
        # Select relevant features for comparison
        feature_columns = [
            'tempo', 'danceability', 'energy', 'valence',
            'acousticness', 'instrumentalness', 'liveness'
        ]
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df[feature_columns])
        
        # Calculate similarity scores
        target_values = np.array([target_features[col] for col in feature_columns])
        target_normalized = scaler.transform(target_values.reshape(1, -1))
        
        similarities = np.dot(normalized_features, target_normalized.T).flatten()
        
        # Get top similar songs
        similar_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        return df.iloc[similar_indices]
    except Exception as e:
        st.error(f"Error finding similar songs: {e}")
        return None 
