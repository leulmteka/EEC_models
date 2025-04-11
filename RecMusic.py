import pandas as pd
import numpy as np
import kagglehub
import random
import os
import time
import scipy.sparse as sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.manifold import TSNE
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional, Attention
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
# try:
# from codecarbon import EmissionsTracker
ENERGY_TRACKING_AVAILABLE = False
# except ImportError:
    # ENERGY_TRACKING_AVAILABLE = False

# Download dataset if not already downloaded
def download_dataset():
    dataset_path = kagglehub.dataset_download("joebeachcapital/30000-spotify-songs")
    return dataset_path

# Load the dataset
def load_data(dataset_path=""):
    # Try different possible file paths
    possible_paths = [
        os.path.join(dataset_path, "spotify_songs.csv"),
        "spotify_songs.csv",
        os.path.join(dataset_path, "30000-spotify-songs", "spotify_songs.csv"),
        "30000-spotify-songs/spotify_songs.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError("Could not find spotify_songs.csv. Please download it manually.")

# Preprocess the data with enhanced feature engineering
# Preprocess the data with enhanced feature engineering and null removal
def preprocess_data(df):
    # Count initial rows
    initial_rows = len(df)
    
    # Remove rows with null values instead of filling them
    null_counts = df.isnull().sum()
    
    # Drop rows with any null values
    df = df.dropna()
    
    # Extract relevant features for content-based filtering
    base_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms', 'popularity']
    
    # Ensure all base features exist in the dataframe
    existing_features = [f for f in base_features if f in df.columns]
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[existing_features] = scaler.fit_transform(df[existing_features])
    
    # Feature engineering - create polynomial features for more complex relationships
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[existing_features])
    
    # Create feature names for polynomial features
    poly_feature_names = []
    for i, feat1 in enumerate(existing_features):
        for feat2 in existing_features[i:]:
            poly_feature_names.append(f"{feat1}_{feat2}")
    
    # Truncate to match actual number of features created
    poly_feature_names = poly_feature_names[:poly_features.shape[1] - len(existing_features)]
    
    # Add polynomial features to dataframe
    for i, name in enumerate(poly_feature_names):
        df[name] = poly_features[:, len(existing_features) + i]
    
    # Create additional features
    if 'track_popularity' in df.columns:
        df['popularity_scaled'] = MinMaxScaler().fit_transform(df[['track_popularity']])
    
    # Create genre and artist embeddings if possible
    if 'playlist_genre' in df.columns:
        genre_dummies = pd.get_dummies(df['playlist_genre'], prefix='genre')
        df = pd.concat([df, genre_dummies], axis=1)
    
    # Create a clean version for display
    display_columns = ['track_id', 'track_name', 'track_artist', 'track_album_name', 
                     'playlist_genre', 'playlist_subgenre']
    available_display_columns = [col for col in display_columns if col in df.columns]
    
    display_df = df[available_display_columns].copy()
    
    # Combine original and engineered features
    all_features = existing_features + poly_feature_names
    if 'popularity_scaled' in df.columns:
        all_features.append('popularity_scaled')
    
    # Add genre features if they exist
    genre_features = [col for col in df.columns if col.startswith('genre_')]
    all_features.extend(genre_features)
    
    return df, display_df, all_features

# Select diverse songs for initial rating with enhanced diversity algorithm
def select_initial_songs(df, n=25):
    
    # Ensure we get a mix of genres and popularity
    if 'playlist_genre' in df.columns:
        genres = df['playlist_genre'].unique()
        
        # Calculate genre distribution for balanced sampling
        genre_counts = df['playlist_genre'].value_counts(normalize=True)
        
        # Calculate songs per genre proportionally
        songs_per_genre = {genre: max(1, int(n * genre_counts[genre])) for genre in genres}
        
        # Adjust to ensure we get exactly n songs
        total = sum(songs_per_genre.values())
        if total < n:
            # Add remaining to largest genres
            for genre in genre_counts.index:
                if total >= n:
                    break
                songs_per_genre[genre] += 1
                total += 1
        elif total > n:
            # Remove from smallest genres
            for genre in genre_counts.index[::-1]:
                if total <= n:
                    break
                if songs_per_genre[genre] > 1:
                    songs_per_genre[genre] -= 1
                    total -= 1
        
        selected_songs = []
        
        # Use K-means clustering within each genre to ensure diversity
        for genre in genres:
            genre_df = df[df['playlist_genre'] == genre]
            if len(genre_df) <= songs_per_genre[genre]:
                # If we have fewer songs than needed, take all
                selected_songs.extend(genre_df['track_id'].tolist())
                continue
                
            # Select features for clustering
            if 'acousticness' in df.columns and 'energy' in df.columns:
                features_for_clustering = ['acousticness', 'danceability', 'energy', 'tempo']
                features_for_clustering = [f for f in features_for_clustering if f in genre_df.columns]
                
                if not features_for_clustering:
                    # Fallback if no audio features available
                    selected_songs.extend(genre_df.sample(songs_per_genre[genre])['track_id'].tolist())
                    continue
                
                # Cluster songs within genre
                n_clusters = min(songs_per_genre[genre], len(genre_df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                try:
                    clusters = kmeans.fit_predict(genre_df[features_for_clustering])
                    genre_df['cluster'] = clusters
                    
                    # Select one song from each cluster
                    for cluster in range(n_clusters):
                        cluster_songs = genre_df[genre_df['cluster'] == cluster]
                        if len(cluster_songs) > 0:
                            selected_songs.append(cluster_songs.sample(1)['track_id'].iloc[0])
                except:
                    # Fallback if clustering fails
                    selected_songs.extend(genre_df.sample(songs_per_genre[genre])['track_id'].tolist())
            else:
                # Fallback if no audio features available
                selected_songs.extend(genre_df.sample(songs_per_genre[genre])['track_id'].tolist())
    else:
        # If no genre information, use clustering on the whole dataset
        if len(df) <= n:
            return df['track_id'].tolist()
            
        # Select features for clustering
        features_for_clustering = [col for col in ['acousticness', 'danceability', 'energy', 'tempo'] 
                                  if col in df.columns]
        
        if not features_for_clustering:
            # Random selection if no suitable features
            return df.sample(n)['track_id'].tolist()
        
        # Cluster all songs
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df[features_for_clustering])
        df['cluster'] = clusters
        
        selected_songs = []
        for cluster in range(n):
            cluster_songs = df[df['cluster'] == cluster]
            if len(cluster_songs) > 0:
                selected_songs.append(cluster_songs.sample(1)['track_id'].iloc[0])
    
    # Ensure we have exactly n songs
    if len(selected_songs) > n:
        selected_songs = selected_songs[:n]
    elif len(selected_songs) < n:
        # Add more random songs if needed
        remaining = n - len(selected_songs)
        additional_df = df[~df['track_id'].isin(selected_songs)]
        if len(additional_df) >= remaining:
            additional = additional_df.sample(remaining)['track_id'].tolist()
            selected_songs.extend(additional)
        else:
            # If we don't have enough unique songs, just use what we have
            selected_songs.extend(additional_df['track_id'].tolist())
    
    return selected_songs

# Energy measurement and reporting functions
class EnergyTracker:
    def __init__(self):
        self.start_time = None
        # Default power values
        self.cpu_power_watts = 40.0  # From codecarbon output
        self.ram_power_watts = 6.26  # From codecarbon output
        self.gpu_power_watts = 30.0  # Estimated for Quadro P1000 (typical TDP)
        self.total_energy_kwh = 0
        self.component_energy = {}
        self.has_codecarbon = ENERGY_TRACKING_AVAILABLE
        self.tracker = None
        self.gpu_measurement_failed = False
        
    def start(self):
        self.start_time = time.time()
        if self.has_codecarbon:
            try:
                self.tracker = EmissionsTracker(project_name="spotify_recommender", 
                                               output_dir=".", 
                                               tracking_mode="process", 
                                               save_to_file=False)
                self.tracker.start()
            except Exception as e:
                print(f"Error starting CodeCarbon tracker: {e}")
                self.has_codecarbon = False
        
    def stop(self):
        duration = time.time() - self.start_time
        
        if self.has_codecarbon and self.tracker:
            try:
                emissions = self.tracker.stop()
                
                # Check if GPU measurement failed and add an estimate if needed
                if self.gpu_measurement_failed:
                    # Calculate estimated GPU energy
                    gpu_energy_kwh = (self.gpu_power_watts * duration) / 3600
                    print(f"Adding estimated GPU energy: {gpu_energy_kwh:.6f} kWh")
                    
                    # Add to total energy (assuming codecarbon's emissions are based on CPU+RAM only)
                    self.total_energy_kwh = gpu_energy_kwh
                    
                    # Recalculate emissions with GPU included
                    additional_emissions = gpu_energy_kwh * 0.4  # 0.4 kg CO2 per kWh
                    emissions += additional_emissions
                
                return emissions
            except Exception as e:
                print(f"Error stopping CodeCarbon tracker: {e}")
                self.has_codecarbon = False
        
        # Fallback to estimate if codecarbon failed
        if not self.has_codecarbon:
            # Calculate energy in kWh
            total_power = self.cpu_power_watts + self.ram_power_watts + self.gpu_power_watts
            self.total_energy_kwh = (total_power * duration) / 3600
            return self.total_energy_kwh * 0.4  # Estimate (0.4 kg CO2 per kWh)
    
    def log_component(self, component_name, duration):
        """Log energy consumption for a specific component"""
        # Use the power values from codecarbon when available
        if component_name == "Content-based Model":
            # Mostly CPU-bound
            power = self.cpu_power_watts * 0.7 + self.ram_power_watts * 0.2
        elif component_name == "Clustering Model":
            # CPU and RAM intensive
            power = self.cpu_power_watts * 0.8 + self.ram_power_watts * 0.5
        elif component_name == "Matrix Factorization Model" or component_name == "Pure Collaborative Filtering Model":
            # CPU intensive
            power = self.cpu_power_watts * 0.9 + self.ram_power_watts * 0.3
        elif component_name == "Deep Learning Model":
            # GPU, CPU, and RAM intensive
            power = self.cpu_power_watts * 0.6 + self.ram_power_watts * 0.4 + self.gpu_power_watts * 0.9
        elif "Recommendation" in component_name:
            # Mixed workload
            power = self.cpu_power_watts * 0.5 + self.ram_power_watts * 0.2 + self.gpu_power_watts * 0.3
        else:
            # Default
            power = self.cpu_power_watts * 0.5 + self.ram_power_watts * 0.2 + self.gpu_power_watts * 0.2
        
        energy_kwh = (power * duration) / 3600
        self.component_energy[component_name] = energy_kwh
        
    def generate_report(self, emissions=None):
        """Generate a comprehensive energy report with relatable metrics"""
        # Check for GPU measurement failure in codecarbon logs
        # if "Failed to retrieve gpu total energy consumption" in open("emissions.csv").read():
        self.gpu_measurement_failed = True
        #     print("Detected that GPU energy measurement failed - will add estimates")
        
        if emissions is None:
            emissions = self.stop()
        
        # Get the total energy from component measurements if available
        if self.component_energy:
            component_total = sum(self.component_energy.values())
            if component_total > 0:
                self.total_energy_kwh = component_total
        
        # Calculate equivalent metrics
        smartphone_charges = self.total_energy_kwh / 0.0127  # kWh per full smartphone charge
        lightbulb_hours = self.total_energy_kwh / 0.01  # 10W LED bulb
        car_miles = emissions * 2.5  # miles per kg of CO2
        
        report = "\n=== ENERGY CONSUMPTION REPORT ===\n"
        report += f"Total Energy Consumption: {self.total_energy_kwh:.6f} kWh\n"
        report += f"{'Measured' if self.has_codecarbon else 'Estimated'} CO2 Emissions: {emissions:.6f} kg CO2eq\n"
        
        if self.gpu_measurement_failed:
            report += "(Note: GPU energy was estimated as direct measurement failed)\n"
        
        report += "\nThis is equivalent to:\n"
        report += f"- Charging a smartphone {smartphone_charges:.1f} times\n"
        report += f"- Keeping a 10W LED light bulb on for {lightbulb_hours:.1f} hours\n"
        report += f"- Driving a car for {car_miles:.2f} miles\n"
        
        if self.component_energy:
            report += "\nEnergy breakdown by component:\n"
            for component, energy in sorted(self.component_energy.items(), key=lambda x: x[1], reverse=True):
                percentage = (energy / self.total_energy_kwh) * 100 if self.total_energy_kwh > 0 else 0
                report += f"- {component}: {energy:.6f} kWh ({percentage:.1f}%)\n"
        
        return report

# Enhanced Content-based filtering model with multiple similarity metrics
class ContentBasedRecommender:
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.similarity_matrix = None
        self.rbf_similarity_matrix = None
        self.feature_importance = None
        
    def fit(self):
        # Compute similarity matrices using different metrics
        feature_matrix = self.df[self.features].values
        
        # Cosine similarity
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # RBF kernel similarity (captures non-linear relationships)
        self.rbf_similarity_matrix = rbf_kernel(feature_matrix)
        
        # Initialize feature importance (will be updated based on user preferences)
        self.feature_importance = np.ones(len(self.features))
        
        return self
    
    def update_feature_importance(self, liked_ids, disliked_ids):
        """Update feature importance based on user preferences"""
        if not liked_ids or not self.features:
            return
            
        # Get feature values for liked and disliked songs
        liked_features = self.df[self.df['track_id'].isin(liked_ids)][self.features].values
        
        if len(liked_features) == 0:
            return
            
        # Calculate feature variance in liked songs
        # Features with low variance are more important (user has consistent preference)
        if len(liked_features) > 1:
            feature_variance = np.var(liked_features, axis=0)
            # Convert variance to importance (lower variance = higher importance)
            raw_importance = 1.0 / (feature_variance + 0.1)  # Add small constant to avoid division by zero
            # Normalize importance
            self.feature_importance = raw_importance / np.sum(raw_importance)
        
        # If we have disliked songs, adjust importance to maximize separation
        if disliked_ids:
            disliked_features = self.df[self.df['track_id'].isin(disliked_ids)][self.features].values
            
            if len(disliked_features) > 0:
                # Calculate separation power of each feature (difference between liked and disliked)
                liked_means = np.mean(liked_features, axis=0)
                disliked_means = np.mean(disliked_features, axis=0)
                separation = np.abs(liked_means - disliked_means)
                
                # Combine with existing importance
                combined_importance = self.feature_importance * separation
                self.feature_importance = combined_importance / np.sum(combined_importance)
        
    def recommend(self, liked_ids, disliked_ids, n=10):
        if not liked_ids:
            return []
            
        # Update feature importance based on preferences
        self.update_feature_importance(liked_ids, disliked_ids)
            
        # Get indices of liked and disliked songs
        liked_indices = [self.df[self.df['track_id'] == song_id].index[0] for song_id in liked_ids if song_id in self.df['track_id'].values]
        disliked_indices = [self.df[self.df['track_id'] == song_id].index[0] for song_id in disliked_ids if song_id in self.df['track_id'].values]
        
        if not liked_indices:
            # Fallback to random recommendations if no liked songs are found
            rated_ids = liked_ids + disliked_ids
            unrated_df = self.df[~self.df['track_id'].isin(rated_ids)]
            if len(unrated_df) >= n:
                return unrated_df.sample(n)['track_id'].tolist()
            else:
                return unrated_df['track_id'].tolist()
        
        #  Weighted combination of similarity matrices
        cosine_weight = 0.7
        rbf_weight = 0.3
        
        # Calculate weighted average similarity with liked songs
        similarity_scores = np.zeros(len(self.df))
        for idx in liked_indices:
            # Weighted combination of similarity metrics
            combined_similarity = (cosine_weight * self.similarity_matrix[idx] + 
                                  rbf_weight * self.rbf_similarity_matrix[idx])
            
            # Apply feature importance weighting
            weighted_features = self.df[self.features].values * self.feature_importance
            song_features = weighted_features[idx]
            all_features = weighted_features
            
            # Calculate weighted feature similarity
            feature_similarity = np.zeros(len(self.df))
            for i in range(len(self.df)):
                feature_similarity[i] = 1.0 / (1.0 + np.sum((song_features - all_features[i])**2))
            
            # Combine different similarity metrics
            final_similarity = 0.5 * combined_similarity + 0.5 * feature_similarity
            similarity_scores += final_similarity
        
        if liked_indices:
            similarity_scores /= len(liked_indices)
            
        # Penalize similarity with disliked songs (with stronger penalty)
        for idx in disliked_indices:
            combined_similarity = (cosine_weight * self.similarity_matrix[idx] + 
                                  rbf_weight * self.rbf_similarity_matrix[idx])
            similarity_scores -= 0.8 * combined_similarity
            
        # Get top recommendations excluding rated songs
        rated_indices = liked_indices + disliked_indices
        similarity_scores[rated_indices] = -np.inf
        
        top_indices = similarity_scores.argsort()[-n:][::-1]
        recommendations = self.df.iloc[top_indices]['track_id'].tolist()
        
        return recommendations

# Enhanced Clustering-based recommender with multiple clustering algorithms
class ClusteringRecommender:
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.kmeans = None
        self.gmm = None
        self.hierarchical = None
        self.df_with_clusters = None
        self.n_clusters = min(20, len(df) // 100)  # More clusters for finer granularity
        
    def fit(self):
        # Create a copy of the dataframe for clustering
        self.df_with_clusters = self.df.copy()
        
        # Select a subset of features for clustering to avoid curse of dimensionality
        if len(self.features) > 20:
            # Use SVD to reduce dimensionality while preserving variance
            n_components = min(20, len(self.features))
            svd = TruncatedSVD(n_components=n_components)
            reduced_features = svd.fit_transform(self.df[self.features])
            
            # Create new feature names
            reduced_feature_names = [f'svd_{i}' for i in range(n_components)]
            
            # Add reduced features to dataframe
            for i, name in enumerate(reduced_feature_names):
                self.df_with_clusters[name] = reduced_features[:, i]
                
            clustering_features = reduced_feature_names
        else:
            clustering_features = self.features
        
        # 1. K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.df_with_clusters['kmeans_cluster'] = self.kmeans.fit_predict(self.df_with_clusters[clustering_features])
        
        # 2. Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42, n_init=3)
        self.df_with_clusters['gmm_cluster'] = self.gmm.fit_predict(self.df_with_clusters[clustering_features])
        
        # 3. Hierarchical Clustering (on a sample if dataset is large)
        if len(self.df) > 10000:
            # Sample for hierarchical clustering which is computationally expensive
            sample_indices = np.random.choice(len(self.df), size=10000, replace=False)
            sample_features = self.df_with_clusters.iloc[sample_indices][clustering_features]
            
            self.hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
            sample_clusters = self.hierarchical.fit_predict(sample_features)
            
            # Initialize all with -1
            self.df_with_clusters['hierarchical_cluster'] = -1
            
            # Assign clusters to sampled points
            for i, idx in enumerate(sample_indices):
                self.df_with_clusters.loc[idx, 'hierarchical_cluster'] = sample_clusters[i]
                
            # For non-sampled points, assign to nearest centroid
            # First, compute centroids from sample
            centroids = {}
            for cluster in range(self.n_clusters):
                cluster_samples = sample_indices[sample_clusters == cluster]
                if len(cluster_samples) > 0:
                    centroids[cluster] = self.df_with_clusters.iloc[cluster_samples][clustering_features].mean().values
            
            # Assign remaining points to nearest centroid
            unassigned = self.df_with_clusters['hierarchical_cluster'] == -1
            for idx in self.df_with_clusters[unassigned].index:
                point = self.df_with_clusters.loc[idx, clustering_features].values
                min_dist = float('inf')
                best_cluster = 0
                
                for cluster, centroid in centroids.items():
                    dist = np.sum((point - centroid) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster
                        
                self.df_with_clusters.loc[idx, 'hierarchical_cluster'] = best_cluster
        else:
            # If dataset is small enough, run hierarchical on full dataset
            self.hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
            self.df_with_clusters['hierarchical_cluster'] = self.hierarchical.fit_predict(
                self.df_with_clusters[clustering_features])
        
        return self
        
    def recommend(self, liked_ids, disliked_ids, n=10):
        if not liked_ids:
            return []
            
        # Define rated_ids right at the beginning
        rated_ids = liked_ids + disliked_ids
            
        # Find clusters of liked songs for each clustering algorithm
        liked_songs = self.df_with_clusters[self.df_with_clusters['track_id'].isin(liked_ids)]
        if len(liked_songs) == 0:
            return []
            
        # Count occurrences of each cluster in liked songs for each algorithm
        kmeans_counts = liked_songs['kmeans_cluster'].value_counts()
        gmm_counts = liked_songs['gmm_cluster'].value_counts()
        hierarchical_counts = liked_songs['hierarchical_cluster'].value_counts()
        
        # Get disliked clusters to avoid for each algorithm
        disliked_clusters = {'kmeans': [], 'gmm': [], 'hierarchical': []}
        if disliked_ids:
            disliked_songs = self.df_with_clusters[self.df_with_clusters['track_id'].isin(disliked_ids)]
            disliked_clusters['kmeans'] = disliked_songs['kmeans_cluster'].unique().tolist()
            disliked_clusters['gmm'] = disliked_songs['gmm_cluster'].unique().tolist()
            disliked_clusters['hierarchical'] = disliked_songs['hierarchical_cluster'].unique().tolist()
        
        # Get recommendations from the most common liked clusters for each algorithm
        kmeans_recommendations = self._get_recommendations_from_clusters(
            'kmeans_cluster', kmeans_counts, disliked_clusters['kmeans'], rated_ids, n)
        
        gmm_recommendations = self._get_recommendations_from_clusters(
            'gmm_cluster', gmm_counts, disliked_clusters['gmm'], rated_ids, n)
        
        hierarchical_recommendations = self._get_recommendations_from_clusters(
            'hierarchical_cluster', hierarchical_counts, disliked_clusters['hierarchical'], rated_ids, n)
        
        # Combine recommendations with weights
        all_recommendations = (
            kmeans_recommendations * 0.4 + 
            gmm_recommendations * 0.3 + 
            hierarchical_recommendations * 0.3
        )
        
        # Sort and get top N
        top_recommendations = all_recommendations.sort_values(ascending=False).head(n).index.tolist()
        
        return top_recommendations
    
    def _get_recommendations_from_clusters(self, cluster_col, cluster_counts, disliked_clusters, rated_ids, n):
        """Helper method to get recommendations from a specific clustering algorithm"""
        # Initialize recommendation scores
        recommendations = pd.Series(0.0, index=self.df_with_clusters['track_id'])
        
        # Handle case where cluster_counts might be empty
        if len(cluster_counts) == 0:
            return recommendations
        
        # Calculate total count for normalization
        total_count = cluster_counts.sum()
        
        # Get recommendations from each cluster, weighted by cluster popularity
        for cluster, count in cluster_counts.items():
            if cluster in disliked_clusters:
                continue
                
            # Weight by the proportion of liked songs in this cluster
            cluster_weight = count / total_count
            
            # Get songs from this cluster that weren't rated
            cluster_songs = self.df_with_clusters[
                (self.df_with_clusters[cluster_col] == cluster) & 
                (~self.df_with_clusters['track_id'].isin(rated_ids))
            ]
            
            # Add weighted scores for songs in this cluster
            if len(cluster_songs) > 0:
                for song_id in cluster_songs['track_id']:
                    recommendations[song_id] += cluster_weight
        
        return recommendations

# Enhanced Matrix Factorization with multiple models
class AdvancedMatrixFactorization:
    def __init__(self, df, features, factors=100, regularization=0.01, iterations=30):
        self.df = df
        self.features = features
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.als_model = None
        self.song_mapping = None
        self.reverse_mapping = None
        self.user_item_matrix = None
        self.n_dummy_users = 500  # More users for better collaborative patterns
        self.is_transposed = False  # Track whether the matrix is transposed
        
    def fit(self):
        # Create a mapping of song IDs to indices
        unique_songs = self.df['track_id'].unique()
        self.song_mapping = {song: i for i, song in enumerate(unique_songs)}
        self.reverse_mapping = {i: song for song, i in self.song_mapping.items()}
        
        n_songs = len(unique_songs)
        n_users = self.n_dummy_users + 1
        
        # Create synthetic user-item matrix with diverse patterns
        self._create_stratified_synthetic_data(n_songs)
        
        # Initialize the ALS model
        self.als_model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False
        )
        
        # Fit the model with synthetic data
        try:
            # The implicit library expects (items x users) format for input
            # But we'll keep track of the original orientation
            item_user_matrix = self.user_item_matrix.T.tocsr()
            self.is_transposed = True
            
            self.als_model.fit(item_user_matrix)
        except Exception as e:
            print(f"Error during initial model fitting: {e}")
            import traceback
            traceback.print_exc()
        
        return self
    
    def _create_stratified_synthetic_data(self, n_songs):
        """Create synthetic data with stratified sampling to ensure genre balance"""
        n_users = self.n_dummy_users + 1  # +1 for the actual user (user 0)
        
        # Create a sparse matrix directly
        data = []
        row = []
        col = []
        
        # Group songs by genre if available
        songs_by_genre = {}
        if 'playlist_genre' in self.df.columns:
            for genre in self.df['playlist_genre'].unique():
                genre_songs = self.df[self.df['playlist_genre'] == genre]['track_id'].values
                song_indices = [self.song_mapping[song] for song in genre_songs if song in self.song_mapping]
                if song_indices:
                    songs_by_genre[genre] = song_indices
            
        else:
            # If no genre info, create one group with all songs
            songs_by_genre['all'] = list(range(n_songs))
        
        # Create synthetic users with stratified preferences
        for user_id in range(1, n_users):
            # Each user rates songs from 2-4 genres (or all if fewer genres)
            n_genres_to_rate = min(np.random.randint(2, 5), len(songs_by_genre))
            genres_to_rate = np.random.choice(list(songs_by_genre.keys()), 
                                            size=n_genres_to_rate, 
                                            replace=False)
            
            # Rate songs from each selected genre
            for genre in genres_to_rate:
                genre_songs = songs_by_genre[genre]
                
                # Rate 10-30 songs from this genre (or all if fewer)
                n_songs_to_rate = min(np.random.randint(10, 31), len(genre_songs))
                songs_to_rate = np.random.choice(genre_songs, 
                                               size=n_songs_to_rate, 
                                               replace=False)
                
                # Add ratings (80% positive, 20% negative)
                for song_idx in songs_to_rate:
                    if np.random.random() < 0.8:  # 80% positive
                        rating = np.random.uniform(1, 5)
                    else:  # 20% negative
                        rating = np.random.uniform(-5, 0)
                    
                    data.append(rating)
                    row.append(user_id)
                    col.append(song_idx)
        
        # Add initial ratings for user 0 (actual user)
        # Rate some songs from each genre for balance
        for genre, genre_songs in songs_by_genre.items():
            # Rate 3-5 songs from each genre
            n_to_rate = min(np.random.randint(3, 6), len(genre_songs))
            songs_to_rate = np.random.choice(genre_songs, size=n_to_rate, replace=False)
            
            for song_idx in songs_to_rate:
                # 50/50 positive/negative for initial ratings
                if np.random.random() < 0.5:
                    rating = np.random.uniform(1, 5)
                else:
                    rating = np.random.uniform(-5, 0)
                
                data.append(rating)
                row.append(0)  # User 0
                col.append(song_idx)
        
        # Create the sparse matrix
        self.user_item_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_users, n_songs))
        # Verify that user 0 has ratings
        user0_ratings = self.user_item_matrix[0].nnz
        
        # Verify genre distribution in synthetic data
        if 'playlist_genre' in self.df.columns:
            rated_songs = []
            for i, j, v in zip(row, col, data):
                if i > 0:  # Skip user 0
                    rated_songs.append(j)
            
            rated_genres = {}
            for song_idx in set(rated_songs):  # Use set to count unique songs
                if song_idx in self.reverse_mapping:
                    song_id = self.reverse_mapping[song_idx]
                    genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                    rated_genres[genre] = rated_genres.get(genre, 0) + 1
    
    def update_preferences(self, liked_ids, disliked_ids):
        """Update the user-item matrix with the actual user's preferences"""
        if self.user_item_matrix is None:
            return
            
        n_songs = self.user_item_matrix.shape[1]
        
        # Create a new row for user 0
        user0_data = np.zeros(n_songs)
        
        # Set liked songs to high rating (4-5)
        for song_id in liked_ids:
            if song_id in self.song_mapping:
                song_idx = self.song_mapping[song_id]
                user0_data[song_idx] = np.random.uniform(4, 5)  # High rating
                
        # Set disliked songs to low rating (-5 to -1)
        for song_id in disliked_ids:
            if song_id in self.song_mapping:
                song_idx = self.song_mapping[song_id]
                user0_data[song_idx] = np.random.uniform(-5, -1)  # Low rating
        
        # Update the user-item matrix
        # Replace the first row (user 0) with the new preferences
        self.user_item_matrix[0] = sparse.csr_matrix(user0_data)
        
        # Refit the model with updated user preferences
        try:
            # The implicit library expects (items x users) format
            item_user_matrix = self.user_item_matrix.T.tocsr()
            self.is_transposed = True
            
            self.als_model.fit(item_user_matrix)
        except Exception as e:
            print(f"Error refitting model: {e}")
            import traceback
            traceback.print_exc()
    
    def recommend(self, liked_ids, disliked_ids, n=10):
        """Get recommendations with strict filtering of disliked genres"""
        if not liked_ids:
            return []
            
        # Update the model with current preferences
        self.update_preferences(liked_ids, disliked_ids)
        
        # Get the indices of rated songs
        rated_ids = liked_ids + disliked_ids
        rated_indices = [self.song_mapping[song_id] for song_id in rated_ids 
                        if song_id in self.song_mapping]
        
        # Identify completely disliked genres (genres where user has only dislikes, no likes)
        completely_disliked_genres = set()
        if 'playlist_genre' in self.df.columns:
            # Get genres of liked songs
            liked_genres = set()
            for song_id in liked_ids:
                if song_id in self.df['track_id'].values:
                    genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                    liked_genres.add(genre)
            
            # Get genres of disliked songs
            disliked_genres = set()
            for song_id in disliked_ids:
                if song_id in self.df['track_id'].values:
                    genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                    disliked_genres.add(genre)
            
            # Genres that are disliked but not liked
            completely_disliked_genres = disliked_genres - liked_genres
        try:
            if self.is_transposed:
                # When transposed, user_factors are item factors and item_factors are user factors
                item_factors = self.als_model.user_factors  # These are actually item factors
                user_factors = self.als_model.item_factors  # These are actually user factors
            else:
                # Normal orientation
                user_factors = self.als_model.user_factors
                item_factors = self.als_model.item_factors
            
            if user_factors is None or item_factors is None:
                return self._fallback_recommendations(rated_ids, n, completely_disliked_genres)
            
            # Get the factors for our user (user 0)
            user_vec = user_factors[0]
            
            # Calculate scores for all items
            scores = np.dot(item_factors, user_vec)
            
            # Set scores of rated items to -inf
            for idx in rated_indices:
                if 0 <= idx < len(scores):
                    scores[idx] = -np.inf
            
            # Also set scores of items from completely disliked genres to -inf
            filtered_count = 0
            if completely_disliked_genres and 'playlist_genre' in self.df.columns:
                for idx, song_id in self.reverse_mapping.items():
                    if idx < len(scores) and song_id in self.df['track_id'].values:
                        genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                        if genre in completely_disliked_genres:
                            scores[idx] = -np.inf
                            filtered_count += 1
            
            # Get top N indices
            top_indices = np.argsort(scores)[-n*3:][::-1]  # Get 3x more candidates
            # Convert to song IDs
            candidate_recs = [self.reverse_mapping[idx] for idx in top_indices 
                             if idx in self.reverse_mapping]
            # Check genre distribution of candidates
            if 'playlist_genre' in self.df.columns and candidate_recs:
                genre_counts = {}
                for rec in candidate_recs:
                    genre = self.df[self.df['track_id'] == rec]['playlist_genre'].values[0]
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Apply genre diversity to final recommendations
            final_recs = self._enforce_genre_diversity(candidate_recs, liked_ids, disliked_ids, n)
            
            return final_recs
        except Exception as e:
            print(f"Error in recommendation: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_recommendations(rated_ids, n, completely_disliked_genres)
    
    def _enforce_genre_diversity(self, candidate_recs, liked_ids, disliked_ids, n):
        """Enforce genre diversity in recommendations with strict filtering of disliked genres"""
        
        if not candidate_recs or 'playlist_genre' not in self.df.columns:
            return candidate_recs[:n]
        
        # Get genre preferences from user ratings
        liked_genres = set()
        for song_id in liked_ids:
            if song_id in self.df['track_id'].values:
                genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                liked_genres.add(genre)
        
        disliked_genres = set()
        for song_id in disliked_ids:
            if song_id in self.df['track_id'].values:
                genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                disliked_genres.add(genre)
        
        # Completely disliked genres (disliked but not liked)
        completely_disliked_genres = disliked_genres - liked_genres
        
        # Group candidates by genre
        genre_candidates = {}
        for rec in candidate_recs:
            genre = self.df[self.df['track_id'] == rec]['playlist_genre'].values[0]
            # Skip completely disliked genres
            if genre in completely_disliked_genres:
                continue
            if genre not in genre_candidates:
                genre_candidates[genre] = []
            genre_candidates[genre].append(rec)
        
        # If no genres left after filtering, try a different approach
        if not genre_candidates:
            # Return random songs from non-disliked genres
            return self._fallback_recommendations(liked_ids + disliked_ids, n, completely_disliked_genres)
        
        # Allocate recommendations across genres
        final_recs = []
        genres = list(genre_candidates.keys())
        
        # Prioritize liked genres if available
        if liked_genres:
            genres.sort(key=lambda g: g in liked_genres, reverse=True)
        
        # Take recommendations from each genre in a round-robin fashion
        while len(final_recs) < n and genres:
            for genre in genres[:]:
                if genre_candidates[genre]:
                    final_recs.append(genre_candidates[genre].pop(0))
                    if len(final_recs) >= n:
                        break
                else:
                    genres.remove(genre)
        
        return final_recs[:n]
    
    def _fallback_recommendations(self, rated_ids, n, completely_disliked_genres=None):
        """Fallback to random recommendations with genre filtering"""
        
        # Get unrated songs
        unrated_songs = self.df[~self.df['track_id'].isin(rated_ids)]
        
        # Filter out completely disliked genres
        if completely_disliked_genres and 'playlist_genre' in self.df.columns:
            for genre in completely_disliked_genres:
                unrated_songs = unrated_songs[unrated_songs['playlist_genre'] != genre]
            
        
        if len(unrated_songs) == 0:
            # If no songs left, relax the genre constraints
            unrated_songs = self.df[~self.df['track_id'].isin(rated_ids)]
        
        if 'playlist_genre' in self.df.columns and len(unrated_songs) > 0:
            # Group by genre
            genre_groups = {}
            for genre in unrated_songs['playlist_genre'].unique():
                genre_songs = unrated_songs[unrated_songs['playlist_genre'] == genre]['track_id'].tolist()
                if genre_songs:
                    genre_groups[genre] = genre_songs
            
            
            # Take songs from each genre in a round-robin fashion
            recommendations = []
            genres = list(genre_groups.keys())
            
            while len(recommendations) < n and genres:
                for genre in genres[:]:
                    if genre_groups[genre]:
                        # Take a random song from this genre
                        song_idx = np.random.randint(0, len(genre_groups[genre]))
                        recommendations.append(genre_groups[genre].pop(song_idx))
                        if len(recommendations) >= n:
                            break
                    else:
                        genres.remove(genre)
            
            return recommendations
        else:
            # If no genre info or no songs left, just return random songs
            if len(unrated_songs) >= n:
                return unrated_songs.sample(n)['track_id'].tolist()
            else:
                return unrated_songs['track_id'].tolist()

class PureCollaborativeFiltering:
    def __init__(self, df, features, factors=100, regularization=0.01, iterations=30):
        self.df = df
        self.features = features
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.als_model = None
        self.song_mapping = None
        self.reverse_mapping = None
        self.user_item_matrix = None
        self.n_dummy_users = 500  # More users for better collaborative patterns
        self.is_transposed = False  # Track whether the matrix is transposed
        
    def fit(self):
        # Create a mapping of song IDs to indices
        unique_songs = self.df['track_id'].unique()
        self.song_mapping = {song: i for i, song in enumerate(unique_songs)}
        self.reverse_mapping = {i: song for song, i in self.song_mapping.items()}
        
        n_songs = len(unique_songs)
        n_users = self.n_dummy_users + 1
        
        # Create synthetic user-item matrix with diverse patterns
        self._create_synthetic_data(n_songs)
        
        # Initialize the ALS model
        self.als_model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False
        )
        
        # Fit the model with synthetic data
        try:
            # The implicit library expects (items x users) format for input
            # But we'll keep track of the original orientation
            item_user_matrix = self.user_item_matrix.T.tocsr()
            self.is_transposed = True
            
            self.als_model.fit(item_user_matrix)
            
            # Verify factor shapes
        except Exception as e:
            print(f"Error during initial model fitting: {e}")
            import traceback
            traceback.print_exc()
        
        return self
    
    def _create_synthetic_data(self, n_songs):
        n_users = self.n_dummy_users + 1  # +1 for the actual user (user 0)
        
        # Create a sparse matrix directly
        data = []
        row = []
        col = []
        
        # Create synthetic users with diverse preferences
        for user_id in range(1, n_users):
            # Each user rates a random subset of songs
            n_songs_to_rate = np.random.randint(50, 200)  # Rate 50-200 songs per user
            songs_to_rate = np.random.choice(n_songs, 
                                           size=min(n_songs_to_rate, n_songs), 
                                           replace=False)
            
            # Add ratings (80% positive, 20% negative)
            for song_idx in songs_to_rate:
                if np.random.random() < 0.8:  # 80% positive
                    rating = np.random.uniform(1, 5)
                else:  # 20% negative
                    rating = np.random.uniform(-5, 0)
                
                data.append(rating)
                row.append(user_id)
                col.append(song_idx)
        
        # Add initial ratings for user 0 (actual user)
        # Rate a random subset of songs
        n_initial_ratings = 20
        initial_songs = np.random.choice(n_songs, size=n_initial_ratings, replace=False)
        
        for song_idx in initial_songs:
            # 50/50 positive/negative for initial ratings
            if np.random.random() < 0.5:
                rating = np.random.uniform(1, 5)
            else:
                rating = np.random.uniform(-5, 0)
            
            data.append(rating)
            row.append(0)  # User 0
            col.append(song_idx)
        
        # Create the sparse matrix
        self.user_item_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_users, n_songs))
        
        # Verify that user 0 has ratings
        user0_ratings = self.user_item_matrix[0].nnz
        
        # Print genre distribution of initial ratings if available
        if 'playlist_genre' in self.df.columns:
            user0_rated_songs = [self.reverse_mapping[j] for i, j, v in zip(row, col, data) if i == 0]
            genre_counts = {}
            for song_id in user0_rated_songs:
                if song_id in self.df['track_id'].values:
                    genre = self.df[self.df['track_id'] == song_id]['playlist_genre'].values[0]
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    def update_preferences(self, liked_ids, disliked_ids):
        """Update the user-item matrix with the actual user's preferences"""
        if self.user_item_matrix is None:
            print("Error: User-item matrix not initialized")
            return
            
        n_songs = self.user_item_matrix.shape[1]
        
        # Create a new row for user 0
        user0_data = np.zeros(n_songs)
        
        # Set liked songs to high rating (4-5)
        for song_id in liked_ids:
            if song_id in self.song_mapping:
                song_idx = self.song_mapping[song_id]
                user0_data[song_idx] = np.random.uniform(4, 5)  # High rating
                
        # Set disliked songs to low rating (-5 to -1)
        for song_id in disliked_ids:
            if song_id in self.song_mapping:
                song_idx = self.song_mapping[song_id]
                user0_data[song_idx] = np.random.uniform(-5, -1)  # Low rating
        
        # Update the user-item matrix
        # Replace the first row (user 0) with the new preferences
        self.user_item_matrix[0] = sparse.csr_matrix(user0_data)
        
        # Refit the model with updated user preferences
        try:
            # The implicit library expects (items x users) format
            item_user_matrix = self.user_item_matrix.T.tocsr()
            self.is_transposed = True
            
            self.als_model.fit(item_user_matrix)
        except Exception as e:
            print(f"Error refitting model: {e}")
            import traceback
            traceback.print_exc()
    
    def recommend(self, liked_ids, disliked_ids, n=10):
        """Get recommendations using pure collaborative filtering without genre filtering"""
        if not liked_ids:
            return []
            
        # Update the model with current preferences
        self.update_preferences(liked_ids, disliked_ids)
        
        # Get the indices of rated songs
        rated_ids = liked_ids + disliked_ids
        rated_indices = [self.song_mapping[song_id] for song_id in rated_ids 
                        if song_id in self.song_mapping]
        
        try:
            if self.is_transposed:
                # When transposed, user_factors are item factors and item_factors are user factors
                item_factors = self.als_model.user_factors  # These are actually item factors
                user_factors = self.als_model.item_factors  # These are actually user factors
            else:
                # Normal orientation
                user_factors = self.als_model.user_factors
                item_factors = self.als_model.item_factors
            
            if user_factors is None or item_factors is None:
                return self._fallback_recommendations(rated_ids, n)
            
            # Get the factors for our user (user 0)
            user_vec = user_factors[0]
            
            # Calculate scores for all items
            scores = np.dot(item_factors, user_vec)
            # Set scores of rated items to -inf
            for idx in rated_indices:
                if 0 <= idx < len(scores):
                    scores[idx] = -np.inf
            
            # Get top N indices
            top_indices = np.argsort(scores)[-n:][::-1]
            # Convert to song IDs
            recommendations = [self.reverse_mapping[idx] for idx in top_indices 
                              if idx in self.reverse_mapping]
            
            return recommendations
        except Exception as e:
            print(f"Error in recommendation: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_recommendations(rated_ids, n)
    
    def _fallback_recommendations(self, rated_ids, n):
        """Simple fallback to random recommendations"""
        
        # Get unrated songs
        unrated_songs = self.df[~self.df['track_id'].isin(rated_ids)]
        
        if len(unrated_songs) >= n:
            return unrated_songs.sample(n)['track_id'].tolist()
        else:
            return unrated_songs['track_id'].tolist()

# Advanced Deep Learning Recommender (Complex Neural Architecture)
class DeepLearningRecommender:
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.model = None
        self.song_features = None
        self.song_mapping = None
        self.reverse_mapping = None
        self.scaler = None
        self.history = None
        
    def fit(self):
        # Create a mapping of song IDs to indices
        unique_songs = self.df['track_id'].unique()
        self.song_mapping = {song: i for i, song in enumerate(unique_songs)}
        self.reverse_mapping = {i: song for song, i in self.song_mapping.items()}
        
        # Extract song features
        self.song_features = self.df[self.features].values
        
        # Normalize features
        self.scaler = MinMaxScaler()
        self.song_features = self.scaler.fit_transform(self.song_features)
        
        # Build a complex deep learning model
        
        # Input layer
        input_features = Input(shape=(len(self.features),), name='input_features')
        
        # 1. Convolutional branch (reshape 1D features to 2D)
        reshaped = tf.reshape(input_features, (-1, len(self.features), 1))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(reshaped)
        conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
        global_pool = GlobalMaxPooling1D()(conv3)
        
        # 2. Deep dense branch
        dense1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_features)
        batch_norm1 = BatchNormalization()(dense1)
        dropout1 = Dropout(0.3)(batch_norm1)
        
        dense2 = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dropout1)
        batch_norm2 = BatchNormalization()(dense2)
        dropout2 = Dropout(0.3)(batch_norm2)
        
        dense3 = Dense(128, activation='relu')(dropout2)
        
        # 3. LSTM branch for sequential patterns
        reshaped_lstm = tf.reshape(input_features, (-1, len(self.features) // 2, 2))
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(reshaped_lstm)
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        
        # Combine all branches
        concat = Concatenate()([global_pool, dense3, lstm2])
        
        # Final dense layers
        combined1 = Dense(128, activation='relu')(concat)
        dropout3 = Dropout(0.2)(combined1)
        combined2 = Dense(64, activation='relu')(dropout3)
        
        # Output layer (preference score)
        output = Dense(1, activation='sigmoid')(combined2)
        
        # Create model
        self.model = Model(inputs=input_features, outputs=output)
        
        # Compile with advanced optimizer settings
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.model.compile(optimizer=optimizer, loss='binary_focal_crossentropy', metrics=['accuracy'])
        
        return self
    
    def update_preferences(self, liked_ids, disliked_ids):
        if not liked_ids and not disliked_ids:
            return
            
        # Create training data from user preferences
        train_features = []
        train_labels = []
        
        # Add liked songs (label 1)
        for song_id in liked_ids:
            if song_id in self.df['track_id'].values:
                idx = self.df[self.df['track_id'] == song_id].index[0]
                train_features.append(self.song_features[idx])
                train_labels.append(1)
                
        # Add disliked songs (label 0)
        for song_id in disliked_ids:
            if song_id in self.df['track_id'].values:
                idx = self.df[self.df['track_id'] == song_id].index[0]
                train_features.append(self.song_features[idx])
                train_labels.append(0)
        
        if len(train_features) < 2:
            return  # Need at least 2 examples for training
            
        # Convert to numpy arrays
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Data augmentation for small datasets
        if len(train_features) < 20:
            # Create synthetic examples by adding small noise
            augmented_features = []
            augmented_labels = []
            
            for i in range(len(train_features)):
                # Create 5 variations of each song
                for _ in range(10):
                    noise_levels = [0.03, 0.05, 0.07, 0.1]
                    noise_level = random.choice(noise_levels)
                    noise = np.random.normal(0, noise_level, train_features[i].shape)
                    augmented_features.append(train_features[i] + noise)
                    augmented_labels.append(train_labels[i])
            
            # Combine original and augmented data
            train_features = np.vstack([train_features, np.array(augmented_features)])
            train_labels = np.concatenate([train_labels, np.array(augmented_labels)])
        
        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )

        self.history = self.model.fit(
            train_features, train_labels,
            epochs=1000,  # More epochs for complex model
            batch_size=min(32, len(train_features)),
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
    
    def recommend(self, liked_ids, disliked_ids, n=10):
        if not liked_ids:
            return []
            
        # Update the model with current preferences
        self.update_preferences(liked_ids, disliked_ids)
        
        # Get all songs except rated ones
        rated_ids = liked_ids + disliked_ids
        unrated_indices = [i for i, song_id in enumerate(self.df['track_id']) 
                          if song_id not in rated_ids]
        
        if not unrated_indices:
            return []
            
        # Get features for unrated songs
        unrated_features = self.song_features[unrated_indices]
        
        # If model wasn't properly trained (not enough data), return random recommendations
        if self.history is None or len(self.history.epoch) < 2:
            # Fallback to random recommendations
            sample_indices = np.random.choice(len(unrated_indices), size=min(n, len(unrated_indices)), replace=False)
            return [self.df.iloc[unrated_indices[i]]['track_id'] for i in sample_indices]
        
        # Predict preferences
        predictions = self.model.predict(unrated_features, verbose=0).flatten()
        
        # Get top N recommendations
        top_indices = predictions.argsort()[-n:][::-1]
        recommendations = [self.df.iloc[unrated_indices[i]]['track_id'] for i in top_indices]
        
        return recommendations

# Ensemble recommender that combines all models with adaptive weighting
class AdaptiveEnsembleRecommender:
    def __init__(self, models, initial_weights=None):
        self.models = models
        self.initial_weights = initial_weights if initial_weights else [1.0/len(models)] * len(models)
        self.weights = self.initial_weights.copy()
        self.performance_history = [[] for _ in range(len(models))]
        self.recommendation_history = {}  # Track which model recommended which songs
        
    def update_weights(self, feedback):
        """Update model weights based on feedback"""
        # feedback is a dict mapping song_id to a rating (1 for liked, -1 for disliked)
        if not feedback or not self.recommendation_history:
            return
        
        # Calculate success rate for each model
        success_rates = []
        
        for i in range(len(self.models)):
            # Get songs recommended by this model
            recommended_songs = self.recommendation_history.get(i, [])
            
            if not recommended_songs:
                success_rates.append(0.5)  # Neutral if no recommendations
                continue
                
            # Calculate success rate (proportion of liked songs)
            total = 0
            success = 0
            
            for song_id in recommended_songs:
                if song_id in feedback:
                    total += 1
                    if feedback[song_id] == 1:  # Liked
                        success += 1
            
            # Calculate rate with smoothing to avoid extremes
            rate = (success + 1) / (total + 2) if total > 0 else 0.5
            success_rates.append(rate)
        
        # Update weights using softmax to ensure they sum to 1
        raw_weights = np.array(success_rates) * np.array(self.weights)
        self.weights = raw_weights / np.sum(raw_weights)
        
        # Reset recommendation history
        self.recommendation_history = {}
        
    def recommend(self, liked_ids, disliked_ids, n=10):
        if not liked_ids:
            return []
            
        # Get recommendations from each model
        all_recommendations = []
        for i, model in enumerate(self.models):
            try:
                recs = model.recommend(liked_ids, disliked_ids, n=n)
                all_recommendations.append(recs)
                
                # Track which model recommended which songs
                self.recommendation_history[i] = recs
            except Exception as e:
                print(f"Error in model {i}: {e}")
                all_recommendations.append([])  # Add empty list for failed model
        
        # If all models failed, return empty list
        if all(not recs for recs in all_recommendations):
            return []
        
        # Count occurrences of each recommendation, weighted by model weights
        rec_scores = {}
        for i, recs in enumerate(all_recommendations):
            for j, rec in enumerate(recs):
                # Score decreases with position in the list
                position_weight = 1.0 - (j / len(recs)) if len(recs) > 0 else 0
                score = self.weights[i] * position_weight
                rec_scores[rec] = rec_scores.get(rec, 0) + score
        
        # Sort by score and return top N
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:n]]

# Function to get user ratings
def get_user_ratings(df, song_ids):
    liked_ids = []
    disliked_ids = []
    neutral_ids = []
    
    print("\n=== SONG RATING PHASE ===")
    print("Please rate each song as 'like' (l), 'dislike' (d), or 'neutral' (n)")
    
    for song_id in song_ids:
        song = df[df['track_id'] == song_id].iloc[0]
        print(f"\nSong: {song['track_name']}")
        print(f"Artist: {song['track_artist']}")
        print(f"Album: {song['track_album_name']}")
        print(f"Genre: {song['playlist_genre']} - {song['playlist_subgenre']}")
        
        while True:
            rating = input("Your rating (l/d/n): ").lower()
            if rating in ['l', 'd', 'n']:
                break
            print("Invalid input. Please enter 'l' for like, 'd' for dislike, or 'n' for neutral.")
        
        if rating == '1' or rating == 'l':
            liked_ids.append(song_id)
        elif rating == '0' or rating == 'd':
            disliked_ids.append(song_id)
        else:
            neutral_ids.append(song_id)
    
    return liked_ids, disliked_ids, neutral_ids

# Display recommendations
def display_recommendations(df, recommendations, title="Recommended Songs"):
    print(f"\n=== {title} ===")
    
    if not recommendations:
        print("No recommendations available. Please rate more songs.")
        return
    
    for i, song_id in enumerate(recommendations, 1):
        song = df[df['track_id'] == song_id].iloc[0]
        print(f"{i}. {song['track_name']} - {song['track_artist']} ({song['playlist_genre']})")

# Main function
def main():
    print("Welcome to the Spotify Song Recommender System!")
    print("This system will recommend songs based on your preferences.")
    
    # Download and load data
    dataset_path = download_dataset()
    df = load_data(dataset_path)
    df, display_df, features = preprocess_data(df)
    
    # Select initial songs for rating
    initial_songs = select_initial_songs(df, n=100)
    
    # Get user ratings
    liked_ids, disliked_ids, neutral_ids = get_user_ratings(display_df, initial_songs)
    
    print("\nTraining recommendation models based on your preferences...")
    print("Energy measurement will begin now...")
    
    # Initialize energy tracker
    energy_tracker = EnergyTracker()
    energy_tracker.start()
    
    # Initialize and train models with timing
    start_time = time.time()
    content_model = ContentBasedRecommender(df, features).fit()
    content_time = time.time() - start_time
    energy_tracker.log_component("Content-based Model", content_time)
    print(f"Content-based model training completed in {content_time:.2f} seconds")
    
    start_time = time.time()
    cluster_model = ClusteringRecommender(df, features).fit()
    cluster_time = time.time() - start_time
    energy_tracker.log_component("Clustering Model", cluster_time)
    print(f"Clustering model training completed in {cluster_time:.2f} seconds")
    
    start_time = time.time()
    pure_model = PureCollaborativeFiltering(df, features).fit()
    pure_time = time.time() - start_time
    energy_tracker.log_component("Pure Collaborative Filtering Model", pure_time)
    print(f"Matrix factorization model training completed in {pure_time:.2f} seconds")

    start_time = time.time()
    matrix_model = AdvancedMatrixFactorization(df, features).fit()
    matrix_time = time.time() - start_time
    energy_tracker.log_component("Matrix Factorization Model", matrix_time)
    print(f"Matrix factorization model training completed in {matrix_time:.2f} seconds")
    
    start_time = time.time()
    deep_model = DeepLearningRecommender(df, features).fit()
    deep_time = time.time() - start_time
    energy_tracker.log_component("Deep Learning Model", deep_time)
    print(f"Deep learning model training completed in {deep_time:.2f} seconds")
    
    # Create adaptive ensemble
    ensemble = AdaptiveEnsembleRecommender(
        # [deep_model], initial_weights=[1]
        [content_model, cluster_model, pure_model, matrix_model, deep_model],
        initial_weights=[0.20, 0.20, 0.20, 0.20, 0.20]  # Initial weights
        # initial_weights=[1, 0, 0, 0]
        # initial_weights=[0, 1, 0, 0]
        # initial_weights=[0, 0, 1, 0]
        # initial_weights=[0, 0, 0, 1]
    )
    
    # Get recommendations with timing
    start_time = time.time()
    recommendations = ensemble.recommend(liked_ids, disliked_ids, n=15)
    recommend_time = time.time() - start_time
    energy_tracker.log_component("Initial Recommendation", recommend_time)
    print(f"Recommendation generation completed in {recommend_time:.2f} seconds")
    
    # Display recommendations
    display_recommendations(display_df, recommendations)
    
    # Feedback loop
    feedback_count = 0
    total_recommendation_time = recommend_time
    
    while True:
        print("\nOptions:")
        print("1. Rate more songs")
        print("2. Get more recommendations")
        print("3. Exit and show energy report")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            # Select more songs for rating
            more_songs = select_initial_songs(df[~df['track_id'].isin(liked_ids + disliked_ids + neutral_ids)], n=15)
            new_liked, new_disliked, new_neutral = get_user_ratings(display_df, more_songs)
            
            # Update preferences
            liked_ids.extend(new_liked)
            disliked_ids.extend(new_disliked)
            neutral_ids.extend(new_neutral)
            
            # Update ensemble weights based on feedback
            feedback = {}
            for song_id in new_liked:
                feedback[song_id] = 1
            for song_id in new_disliked:
                feedback[song_id] = -1
                
            ensemble.update_weights(feedback)
            
            # Get new recommendations
            start_time = time.time()
            recommendations = ensemble.recommend(liked_ids, disliked_ids, n=15)
            rec_time = time.time() - start_time
            total_recommendation_time += rec_time
            energy_tracker.log_component(f"Recommendation Round {feedback_count+1}", rec_time)
            print(f"Recommendation generation completed in {rec_time:.2f} seconds")
            
            display_recommendations(display_df, recommendations, "Updated Recommendations")
            feedback_count += 1
            
        elif choice == '2':
            # Get more recommendations
            start_time = time.time()
            recommendations = ensemble.recommend(liked_ids, disliked_ids, n=15)
            rec_time = time.time() - start_time
            total_recommendation_time += rec_time
            energy_tracker.log_component(f"Recommendation Round {feedback_count+1}", rec_time)
            print(f"Recommendation generation completed in {rec_time:.2f} seconds")
            
            display_recommendations(display_df, recommendations, "More Recommendations")
            feedback_count += 1
            
        elif choice == '3':
            print("\nThank you for using the Spotify Song Recommender System!")
            
            # Print performance report
            print("\n=== PERFORMANCE REPORT ===")
            print(f"Content-Based Model Training: {content_time:.2f} seconds")
            print(f"Clustering Model Training: {cluster_time:.2f} seconds")
            print(f"Pure Collaborative Filtering Training: {pure_time:.2f} seconds")
            print(f"Matrix Factorization Training: {matrix_time:.2f} seconds")
            print(f"Deep Learning Model Training: {deep_time:.2f} seconds")
            print(f"Total Training Time: {content_time + cluster_time + pure_time + matrix_time + deep_time:.2f} seconds")
            print(f"Average Recommendation Generation Time: {total_recommendation_time/(feedback_count+1):.2f} seconds")
            print(f"Number of Feedback Iterations: {feedback_count}")
            
            # Generate and print energy report
            emissions = energy_tracker.stop()
            energy_report = energy_tracker.generate_report(emissions)
            print(energy_report)
            
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()