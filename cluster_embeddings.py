#!/usr/bin/env python3
"""
cluster_embeddings.py - Production clustering for YC startup map

Uses Leiden community detection on KNN graph of embeddings, with HDBSCAN fallback.
Do NOT use k-means on 2-D map - cluster in original embedding space for semantic coherence.

Required packages:
    pip install numpy scikit-learn hdbscan igraph leidenalg

Optional fallback (if igraph/leidenalg unavailable):
    pip install numpy scikit-learn hdbscan
"""

import json
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies() -> Tuple[bool, bool]:
    """Check which clustering libraries are available"""
    leiden_available = False
    hdbscan_available = False
    
    try:
        import igraph
        import leidenalg
        leiden_available = True
        logger.info("✅ Leiden dependencies available (igraph + leidenalg)")
    except ImportError:
        logger.warning("⚠️ Leiden dependencies not available (igraph + leidenalg)")
    
    try:
        import hdbscan
        hdbscan_available = True
        logger.info("✅ HDBSCAN available")
    except ImportError:
        logger.warning("⚠️ HDBSCAN not available")
    
    return leiden_available, hdbscan_available

def load_data(points_path: str, embeddings_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load points and embeddings data"""
    logger.info(f"Loading data from {points_path} and {embeddings_path}")
    
    # Load points
    if not Path(points_path).exists():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    
    with open(points_path, 'r', encoding='utf-8') as f:
        points = json.load(f)
    
    # Load embeddings
    if not Path(embeddings_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    
    # Validate dimensions
    if len(points) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(points)} points vs {embeddings.shape[0]} embeddings")
    
    # Ensure L2 normalization (required for cosine similarity via euclidean)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        logger.info("Normalizing embeddings (L2 norm)")
        embeddings = embeddings / norms[:, np.newaxis]
    
    logger.info(f"Loaded {len(points)} points with {embeddings.shape[1]}-dim embeddings")
    return points, embeddings

def build_knn_graph(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build KNN graph with cosine similarity weights"""
    from sklearn.neighbors import NearestNeighbors
    
    logger.info(f"Building KNN graph with k={k}")
    start_time = time.time()
    
    # Use cosine metric for KNN
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    knn.fit(embeddings)
    
    # Get distances and indices
    distances, indices = knn.kneighbors(embeddings)
    
    # Convert cosine distances to cosine similarities
    # cosine_similarity = 1 - cosine_distance
    similarities = 1 - distances
    
    # Build edge list (source, target, weight)
    edges = []
    weights = []
    
    for i in range(len(embeddings)):
        for j in range(1, k):  # Skip self (j=0)
            neighbor = indices[i, j]
            similarity = similarities[i, j]
            
            # Add undirected edge (both directions)
            edges.extend([(i, neighbor), (neighbor, i)])
            weights.extend([similarity, similarity])
    
    # Remove duplicates and keep max weight
    edge_dict = {}
    for (src, tgt), weight in zip(edges, weights):
        key = (min(src, tgt), max(src, tgt))  # Canonical ordering
        edge_dict[key] = max(edge_dict.get(key, 0), weight)
    
    # Convert back to arrays
    final_edges = []
    final_weights = []
    for (src, tgt), weight in edge_dict.items():
        final_edges.extend([(src, tgt), (tgt, src)])  # Undirected
        final_weights.extend([weight, weight])
    
    logger.info(f"Built KNN graph in {time.time() - start_time:.2f}s: {len(final_edges)} edges")
    return np.array(final_edges), np.array(final_weights)

def leiden_clustering(embeddings: np.ndarray, k: int, resolution: float, 
                     min_cluster_size: int, seed: int) -> np.ndarray:
    """Perform Leiden community detection on KNN graph"""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("Leiden clustering requires: pip install igraph leidenalg")
    
    logger.info(f"Running Leiden clustering (resolution={resolution}, min_size={min_cluster_size})")
    start_time = time.time()
    
    # Build KNN graph
    edges, weights = build_knn_graph(embeddings, k)
    
    # Create igraph
    n_nodes = len(embeddings)
    g = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    g.es['weight'] = weights.tolist()
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=seed
    )
    
    # Get cluster assignments
    cluster_labels = np.array(partition.membership)
    
    # Filter small clusters (mark as noise = -1)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    small_clusters = unique[counts < min_cluster_size]
    
    for small_cluster in small_clusters:
        cluster_labels[cluster_labels == small_cluster] = -1
    
    # Renumber clusters to be contiguous (0, 1, 2, ...)
    valid_clusters = unique[counts >= min_cluster_size]
    cluster_mapping = {old: new for new, old in enumerate(valid_clusters)}
    cluster_mapping[-1] = -1  # Keep noise as -1
    
    final_labels = np.array([cluster_mapping.get(label, -1) for label in cluster_labels])
    
    logger.info(f"Leiden clustering completed in {time.time() - start_time:.2f}s")
    return final_labels

def hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int, 
                       min_samples: int, seed: int) -> np.ndarray:
    """Perform HDBSCAN clustering on normalized embeddings"""
    try:
        import hdbscan
    except ImportError:
        raise ImportError("HDBSCAN clustering requires: pip install hdbscan")
    
    logger.info(f"Running HDBSCAN clustering (min_cluster_size={min_cluster_size})")
    start_time = time.time()
    
    # Use euclidean metric on L2-normalized embeddings (equivalent to cosine)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=0.0
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    logger.info(f"HDBSCAN clustering completed in {time.time() - start_time:.2f}s")
    return cluster_labels

def compute_cluster_summary(cluster_labels: np.ndarray) -> Dict[str, Any]:
    """Compute clustering summary statistics"""
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    # Separate noise from real clusters
    noise_mask = unique == -1
    noise_count = counts[noise_mask][0] if noise_mask.any() else 0
    
    # Real clusters (excluding noise)
    real_clusters = unique[~noise_mask]
    real_counts = counts[~noise_mask]
    
    # Sort by size (descending)
    sorted_idx = np.argsort(real_counts)[::-1]
    top_10_sizes = real_counts[sorted_idx][:10].tolist()
    
    summary = {
        "num_clusters": len(real_clusters),
        "noise_count": int(noise_count),
        "total_points": len(cluster_labels),
        "clustered_points": int(len(cluster_labels) - noise_count),
        "clustering_rate": float((len(cluster_labels) - noise_count) / len(cluster_labels)),
        "top_10_cluster_sizes": top_10_sizes,
        "largest_cluster_size": int(top_10_sizes[0]) if top_10_sizes else 0,
        "smallest_cluster_size": int(top_10_sizes[-1]) if top_10_sizes else 0
    }
    
    return summary

def save_results(points: List[Dict], cluster_labels: np.ndarray, 
                output_path: str, summary: Dict[str, Any]) -> None:
    """Save clustered points and summary"""
    logger.info(f"Saving results to {output_path}")
    
    # Add cluster_id to each point
    clustered_points = []
    for i, point in enumerate(points):
        new_point = point.copy()
        new_point['cluster_id'] = int(cluster_labels[i])
        clustered_points.append(new_point)
    
    # Save clustered points
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clustered_points, f, indent=2)
    
    # Save summary
    summary_path = output_path.replace('.json', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved: {output_path} and {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Cluster YC startups using Leiden or HDBSCAN")
    parser.add_argument('--points', default='out_full/points.json',
                       help='Input points JSON file')
    parser.add_argument('--output', default='out_full/points_with_clusters.json',
                       help='Output points with cluster_id JSON file')
    parser.add_argument('--embeddings', default='out_full/embeddings.npy',
                       help='Input embeddings numpy file')
    parser.add_argument('--algo', choices=['leiden', 'hdbscan'], default='leiden',
                       help='Clustering algorithm')
    parser.add_argument('--k', type=int, default=20,
                       help='K for KNN graph (Leiden only)')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Resolution parameter (Leiden only)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum cluster size (post-filter for Leiden, min_cluster_size for HDBSCAN)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Min samples parameter (HDBSCAN only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for determinism')
    
    args = parser.parse_args()
    
    try:
        # Check dependencies
        leiden_available, hdbscan_available = check_dependencies()
        
        # Auto-select algorithm if preferred is not available
        if args.algo == 'leiden' and not leiden_available:
            if hdbscan_available:
                logger.warning("Leiden not available, falling back to HDBSCAN")
                args.algo = 'hdbscan'
            else:
                raise ImportError("Neither Leiden nor HDBSCAN dependencies available")
        elif args.algo == 'hdbscan' and not hdbscan_available:
            if leiden_available:
                logger.warning("HDBSCAN not available, falling back to Leiden")
                args.algo = 'leiden'
            else:
                raise ImportError("Neither Leiden nor HDBSCAN dependencies available")
        
        # Load data
        points, embeddings = load_data(args.points, args.embeddings)
        
        # Run clustering
        start_time = time.time()
        if args.algo == 'leiden':
            cluster_labels = leiden_clustering(
                embeddings, args.k, args.resolution, args.min_cluster_size, args.seed
            )
        else:  # hdbscan
            cluster_labels = hdbscan_clustering(
                embeddings, args.min_cluster_size, args.min_samples, args.seed
            )
        
        total_time = time.time() - start_time
        
        # Compute summary
        summary = compute_cluster_summary(cluster_labels)
        summary['algorithm'] = args.algo
        summary['parameters'] = vars(args)
        summary['runtime_seconds'] = round(total_time, 2)
        
        # Log summary
        logger.info(f"Clustering completed in {total_time:.2f}s")
        logger.info(f"Found {summary['num_clusters']} clusters")
        logger.info(f"Clustered {summary['clustered_points']}/{summary['total_points']} points ({summary['clustering_rate']:.1%})")
        logger.info(f"Noise points: {summary['noise_count']}")
        logger.info(f"Top 5 cluster sizes: {summary['top_10_cluster_sizes'][:5]}")
        
        # Save results
        save_results(points, cluster_labels, args.output, summary)
        
        # Acceptance check
        if summary['clustering_rate'] < 0.8:
            logger.warning(f"⚠️ Only {summary['clustering_rate']:.1%} of points clustered (target: ≥80%)")
        else:
            logger.info(f"✅ Acceptance check passed: {summary['clustering_rate']:.1%} clustered")
        
        if total_time > 60:
            logger.warning(f"⚠️ Runtime {total_time:.2f}s exceeds 60s target")
        else:
            logger.info(f"✅ Performance check passed: {total_time:.2f}s < 60s")
        
        logger.info("🎉 Clustering pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Clustering failed: {e}")
        raise

if __name__ == "__main__":
    main()