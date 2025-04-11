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
import warnings
warnings.filterwarnings('ignore')

# For energy measurement
try:
    from codecarbon import EmissionsTracker
    ENERGY_TRACKING_AVAILABLE = True
except ImportError:
    print("CodeCarbon not available. Install with: pip install codecarbon")
    ENERGY_TRACKING_AVAILABLE = False

# Download dataset if not already downloaded
def download_dataset():
    print("Downloading Spotify dataset...")
    try:
        # Download and get the path to the dataset
        dataset_path = kagglehub.dataset_download("joebeachcapital/30000-spotify-songs")
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Trying alternative approach...")
        
        # Alternative: If kagglehub download fails, check if file exists locally
        if os.path.exists("spotify_songs.csv"):
            print("Found dataset locally.")
            return ""
        else:
            print("Please download the dataset manually from Kaggle:")
            print("https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs")
            print("Save the spotify_songs.csv file in the current directory.")
            exit(1)

# Load the dataset
def load_data(dataset_path=""):
    print("Loading Spotify dataset...")
    
    # Try different possible file paths
    possible_paths = [
        os.path.join(dataset_path, "spotify_songs.csv"),
        "spotify_songs.csv",
        os.path.join(dataset_path, "30000-spotify-songs", "spotify_songs.csv"),
        "30000-spotify-songs/spotify_songs.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading from: {path}")
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} songs")
            return df
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError("Could not find spotify_songs.csv. Please download it manually.")

# Preprocess the data with enhanced feature engineering
def preprocess_data(df):
    print("Preprocessing data with advanced feature engineering...")
    
    # Handle missing values with more sophisticated approach
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            # Fill missing values with median for numerical columns
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
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
    display_df = df[['track_id', 'track_name', 'track_artist', 'track_album_name', 
                     'playlist_genre', 'playlist_subgenre']].copy() if all(col in df.columns for col in 
                     ['track_id', 'track_name', 'track_artist', 'track_album_name', 
                     'playlist_genre', 'playlist_subgenre']) else df[['track_id']].copy()
    
    # Combine original and engineered features
    all_features = existing_features + poly_feature_names
    if 'popularity_scaled' in df.columns:
        all_features.append('popularity_scaled')
    
    # Add genre features if they exist
    genre_features = [col for col in df.columns if col.startswith('genre_')]
    all_features.extend(genre_features)
    
    print(f"Created {len(all_features)} features through feature engineering")
    
    return df, display_df, all_features

# Select diverse songs for initial rating with enhanced diversity algorithm
def select_initial_songs(df, n=25):
    print(f"Selecting {n} diverse songs using advanced diversity algorithm...")
    
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
        self.cpu_power_watts = 65  # Estimated average CPU power consumption
        self.gpu_power_watts = 0   # Set to appropriate value if GPU is used
        self.total_energy_kwh = 0
        self.component_energy = {}
        self.has_codecarbon = ENERGY_TRACKING_AVAILABLE
        self.tracker = None
        
    def start(self):
        self.start_time = time.time()
        if self.has_codecarbon:
            self.tracker = EmissionsTracker(project_name="spotify_recommender", 
                                           output_dir=".", 
                                           tracking_mode="process", 
                                           save_to_file=True)
            self.tracker.start()
        
    def stop(self):
        duration = time.time() - self.start_time
        # Calculate energy in kWh
        self.total_energy_kwh = (self.cpu_power_watts + self.gpu_power_watts) * duration / 3600000
        
        if self.has_codecarbon:
            emissions = self.tracker.stop()
            return emissions
        else:
            # Estimate emissions if codecarbon not available (0.4 kg CO2 per kWh as average)
            return self.total_energy_kwh * 0.4
    
    def log_component(self, component_name, duration):
        """Log energy consumption for a specific component"""
        energy_kwh = (self.cpu_power_watts + self.gpu_power_watts) * duration / 3600000
        self.component_energy[component_name] = energy_kwh
        self.total_energy_kwh += energy_kwh
        
    def generate_report(self, emissions=None):
        """Generate a comprehensive energy report with relatable metrics"""
        if emissions is None and self.has_codecarbon:
            emissions = self.tracker.stop()
        elif emissions is None:
            emissions = self.total_energy_kwh * 0.4  # Estimate if not provided
        
        # Calculate equivalent metrics
        smartphone_charges = self.total_energy_kwh / 0.0127  # kWh per full smartphone charge
        lightbulb_hours = self.total_energy_kwh / 0.01  # 10W LED bulb
        car_miles = emissions * 2.5  # miles per kg of CO2
        
        report = "\n=== ENERGY CONSUMPTION REPORT ===\n"
        report += f"Total Energy Consumption: {self.total_energy_kwh:.6f} kWh\n"
        report += f"Total CO2 Emissions: {emissions:.6f} kg CO2eq\n\n"
        
        report += "This is equivalent to:\n"
        report += f"- Charging a smartphone {smartphone_charges:.1f} times\n"
        report += f"- Keeping a 10W LED light bulb on for {lightbulb_hours:.1f} hours\n"
        report += f"- Driving a car for {car_miles:.2f} miles\n"
        
        if self.component_energy:
            report += "\nEnergy breakdown by component:\n"
            for component, energy in self.component_energy.items():
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
        print("Training enhanced content-based recommender with multiple similarity metrics...")
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
        print("Training advanced clustering recommender with multiple clustering algorithms...")
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
        self.bpr_model = None
        self.lmf_model = None
        self.user_items = None
        self.song_mapping = None
        self.reverse_mapping = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self):
        print("Training advanced matrix factorization with multiple models...")
        # Create a mapping of song IDs to indices
        unique_songs = self.df['track_id'].unique()
        self.song_mapping = {song: i for i, song in enumerate(unique_songs)}
        self.reverse_mapping = {i: song for song, i in self.song_mapping.items()}
        
        # We'll create a dummy user-item matrix for initial training
        n_songs = len(unique_songs)
        
        # Create a sparse matrix with some dummy data for initial training
        data = np.zeros(1)
        row = np.zeros(1)
        col = np.zeros(1)
        self.user_items = sparse.csr_matrix((data, (row, col)), shape=(1, n_songs))
        
        # Initialize the ALS model with more factors for higher complexity
        self.als_model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False  # Set to True if GPU is available
        )
        
        # Initialize BPR model
        self.bpr_model = BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False
        )
        
        # Initialize LMF model
        self.lmf_model = LogisticMatrixFactorization(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False
        )
        
        # Fit the models with dummy data
        self.als_model.fit(self.user_items.T)
        self.bpr_model.fit(self.user_items.T)
        self.lmf_model.fit(self.user_items.T)
        
        return self
    
    def update_preferences(self, liked_ids, disliked_ids):
        # Update the user-item matrix with the user's preferences
        n_songs = len(self.song_mapping)
        
        # Create data for sparse matrix
        data = []
        row = []
        col = []
        
        # Set liked songs to 1
        for song_id in liked_ids:
            if song_id in self.song_mapping:
                data.append(1.0)
                row.append(0)  # Single user (index 0)
                col.append(self.song_mapping[song_id])
                
        # Set disliked songs to -1
        for song_id in disliked_ids:
            if song_id in self.song_mapping:
                data.append(-1.0)
                row.append(0)  # Single user (index 0)
                col.append(self.song_mapping[song_id])
        
        # Create sparse matrix
        if data:  # Only if we have some preferences
            self.user_items = sparse.csr_matrix(
                (data, (row, col)), 
                shape=(1, n_songs)
            )
            
            # For BPR and LMF, we need positive-only feedback
            positive_data = []
            positive_row = []
            positive_col = []
            
            for song_id in liked_ids:
                if song_id in self.song_mapping:
                    positive_data.append(1.0)
                    positive_row.append(0)
                    positive_col.append(self.song_mapping[song_id])
            
            positive_matrix = sparse.csr_matrix(
                (positive_data, (positive_row, positive_col)),
                shape=(1, n_songs)
            )
            
            # Retrain the models with updated preferences
            self.als_model.fit(self.user_items.T)
            
            if positive_data:  # BPR and LMF need positive feedback
                self.bpr_model.fit(positive_matrix.T)
                self.lmf_model.fit(positive_matrix.T)
        else:
            # Create empty sparse matrix if no preferences
            self.user_items = sparse.csr_matrix(([], ([], [])), shape=(1, n_songs))
    
    def get_als_recommendations(self, n, rated_ids):
        """Get recommendations from ALS model"""
        try:
            recommendations, _ = self.als_model.recommend(
                0, self.user_items, N=n+len(rated_ids), 
                filter_already_liked_items=True
            )
            
            # Convert indices to song IDs
            rec_song_ids = [self.reverse_mapping[idx] for idx in recommendations 
                            if idx in self.reverse_mapping and self.reverse_mapping[idx] not in rated_ids]
            
            return rec_song_ids[:n]
        except Exception as e:
            print(f"Error in ALS recommendations: {e}")
            return []
    
    def get_bpr_recommendations(self, n, rated_ids):
        """Get recommendations from BPR model"""
        try:
            recommendations, _ = self.bpr_model.recommend(
                0, self.user_items, N=n+len(rated_ids), 
                filter_already_liked_items=True
            )
            
            # Convert indices to song IDs
            rec_song_ids = [self.reverse_mapping[idx] for idx in recommendations 
                            if idx in self.reverse_mapping and self.reverse_mapping[idx] not in rated_ids]
            
            return rec_song_ids[:n]
        except Exception as e:
            print(f"Error in BPR recommendations: {e}")
            return []
    
    def get_lmf_recommendations(self, n, rated_ids):
        """Get recommendations from LMF model"""
        try:
            recommendations, _ = self.lmf_model.recommend(
                0, self.user_items, N=n+len(rated_ids), 
                filter_already_liked_items=True
            )
            
            # Convert indices to song IDs
            rec_song_ids = [self.reverse_mapping[idx] for idx in recommendations 
                            if idx in self.reverse_mapping and self.reverse_mapping[idx] not in rated_ids]
            
            return rec_song_ids[:n]
        except Exception as e:
            print(f"Error in LMF recommendations: {e}")
            return []
        
    def recommend(self, liked_ids, disliked_ids, n=10):
        if not liked_ids:
            return []
            
        # Update the models with current preferences
        self.update_preferences(liked_ids, disliked_ids)
        
        # Combine recommendations from all models
        rated_ids = liked_ids + disliked_ids
        
        # Get recommendations from each model
        als_recs = self.get_als_recommendations(n, rated_ids)
        bpr_recs = self.get_bpr_recommendations(n, rated_ids)
        lmf_recs = self.get_lmf_recommendations(n, rated_ids)
        
        # If all models failed, fallback to random recommendations
        if not als_recs and not bpr_recs and not lmf_recs:
            unrated_songs = self.df[~self.df['track_id'].isin(rated_ids)]
            if len(unrated_songs) >= n:
                return unrated_songs.sample(n)['track_id'].tolist()
            else:
                return unrated_songs['track_id'].tolist()
        
        # Combine recommendations with weights
        # ALS is more stable, so give it higher weight
        all_recs = {}
        
        for i, rec in enumerate(als_recs):
            score = 0.5 * (1.0 - i/len(als_recs) if len(als_recs) > 0 else 0)
            all_recs[rec] = all_recs.get(rec, 0) + score
            
        for i, rec in enumerate(bpr_recs):
            score = 0.3 * (1.0 - i/len(bpr_recs) if len(bpr_recs) > 0 else 0)
            all_recs[rec] = all_recs.get(rec, 0) + score
            
        for i, rec in enumerate(lmf_recs):
            score = 0.2 * (1.0 - i/len(lmf_recs) if len(lmf_recs) > 0 else 0)
            all_recs[rec] = all_recs.get(rec, 0) + score
        
        # Sort by score and return top N
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:n]]

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
        print("Training advanced deep learning recommender with complex neural architecture...")
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
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
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
        if len(train_features) < 10:
            # Create synthetic examples by adding small noise
            augmented_features = []
            augmented_labels = []
            
            for i in range(len(train_features)):
                # Create 5 variations of each song
                for _ in range(5):
                    noise = np.random.normal(0, 0.05, train_features[i].shape)
                    augmented_features.append(train_features[i] + noise)
                    augmented_labels.append(train_labels[i])
            
            # Combine original and augmented data
            train_features = np.vstack([train_features, np.array(augmented_features)])
            train_labels = np.concatenate([train_labels, np.array(augmented_labels)])
        
        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        self.history = self.model.fit(
            train_features, train_labels,
            epochs=100,  # More epochs for complex model
            batch_size=min(32, len(train_features)),
            callbacks=[early_stopping],
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
    initial_songs = select_initial_songs(df, n=25)
    
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
        [content_model, cluster_model, matrix_model, deep_model],
        # initial_weights=[0.3, 0.2, 0.3, 0.2]  # Initial weights
        # initial_weights=[1, 0, 0, 0]
        initial_weights=[0, 1, 0, 0]
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
            more_songs = select_initial_songs(df[~df['track_id'].isin(liked_ids + disliked_ids + neutral_ids)], n=10)
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
            print(f"Matrix Factorization Training: {matrix_time:.2f} seconds")
            print(f"Deep Learning Model Training: {deep_time:.2f} seconds")
            print(f"Total Training Time: {content_time + cluster_time + matrix_time + deep_time:.2f} seconds")
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