#!/usr/bin/env python3
"""
Reposition nodes based on cluster assignments for better visualization
Creates a circular layout where clusters are grouped together
"""

import json
import numpy as np
import math
from collections import defaultdict

def reposition_by_clusters():
    """Reposition nodes to group clusters together"""
    
    # Load clustered points
    with open('out_full/points_with_clusters.json', 'r', encoding='utf-8') as f:
        points = json.load(f)
    
    print(f"Repositioning {len(points)} points based on clusters...")
    
    # Group points by cluster
    clusters = defaultdict(list)
    noise_points = []
    
    for i, point in enumerate(points):
        cluster_id = point.get('cluster_id', -1)
        if cluster_id == -1:
            noise_points.append((i, point))
        else:
            clusters[cluster_id].append((i, point))
    
    print(f"Found {len(clusters)} clusters and {len(noise_points)} noise points")
    
    # Calculate cluster positions in a circular layout
    cluster_radius = 50  # Radius of the main circle
    cluster_inner_radius = 8  # Radius within each cluster
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Position clusters in a circle
    repositioned_points = points.copy()
    
    for cluster_idx, (cluster_id, cluster_points) in enumerate(sorted_clusters):
        # Angle for this cluster in the main circle
        angle = 2 * math.pi * cluster_idx / len(sorted_clusters)
        
        # Center position for this cluster
        cluster_center_x = cluster_radius * math.cos(angle)
        cluster_center_y = cluster_radius * math.sin(angle)
        
        # Arrange points within the cluster in a smaller circle or grid
        n_points = len(cluster_points)
        
        if n_points == 1:
            # Single point at cluster center
            idx, point = cluster_points[0]
            repositioned_points[idx]['x'] = cluster_center_x
            repositioned_points[idx]['y'] = cluster_center_y
        elif n_points <= 20:
            # Small cluster - arrange in circle
            for i, (idx, point) in enumerate(cluster_points):
                point_angle = 2 * math.pi * i / n_points
                x = cluster_center_x + cluster_inner_radius * math.cos(point_angle)
                y = cluster_center_y + cluster_inner_radius * math.sin(point_angle)
                repositioned_points[idx]['x'] = x
                repositioned_points[idx]['y'] = y
        else:
            # Large cluster - arrange in spiral
            for i, (idx, point) in enumerate(cluster_points):
                # Spiral parameters
                spiral_radius = cluster_inner_radius * math.sqrt(i / n_points) * 2
                spiral_angle = 0.5 * math.sqrt(i) * 2 * math.pi
                
                x = cluster_center_x + spiral_radius * math.cos(spiral_angle)
                y = cluster_center_y + spiral_radius * math.sin(spiral_angle)
                repositioned_points[idx]['x'] = x
                repositioned_points[idx]['y'] = y
    
    # Position noise points in the center or randomly
    if noise_points:
        for i, (idx, point) in enumerate(noise_points):
            # Random position near center
            angle = 2 * math.pi * i / len(noise_points)
            radius = 5 + np.random.uniform(0, 10)  # Small radius near center
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            repositioned_points[idx]['x'] = x
            repositioned_points[idx]['y'] = y
    
    # Save repositioned points
    with open('out_full/points_with_clusters.json', 'w', encoding='utf-8') as f:
        json.dump(repositioned_points, f, indent=2)
    
    print(f"✅ Repositioned all points based on {len(clusters)} clusters")
    print(f"📊 Cluster layout: {len(sorted_clusters)} clusters in circular arrangement")
    print(f"🎯 Largest cluster: {len(sorted_clusters[0][1])} points")
    print(f"🔍 Noise points: {len(noise_points)} positioned near center")

if __name__ == "__main__":
    reposition_by_clusters()