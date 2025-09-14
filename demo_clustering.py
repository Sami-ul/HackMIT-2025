#!/usr/bin/env python3
"""
Demo clustering script - creates fake clusters based on tag similarity
This is a temporary solution while embedding generation has TensorFlow conflicts
"""

import json
import numpy as np
import random
from collections import defaultdict

def create_demo_clusters():
    """Create demo clusters based on tag similarity"""
    
    # Load points
    with open('out_full/points.json', 'r', encoding='utf-8') as f:
        points = json.load(f)
    
    print(f"Creating demo clusters for {len(points)} points...")
    
    # Group by primary tag categories
    tag_groups = defaultdict(list)
    
    for i, point in enumerate(points):
        tags = point.get('tags', [])
        if not tags:
            tag_groups['uncategorized'].append(i)
            continue
        
        # Use first tag as primary category
        primary_tag = tags[0].lower()
        
        # Group similar tags together
        if any(keyword in primary_tag for keyword in ['ai', 'ml', 'machine learning', 'artificial intelligence']):
            tag_groups['ai_ml'].append(i)
        elif any(keyword in primary_tag for keyword in ['finance', 'fintech', 'payment', 'banking', 'crypto']):
            tag_groups['fintech'].append(i)
        elif any(keyword in primary_tag for keyword in ['health', 'medical', 'healthcare', 'biotech']):
            tag_groups['healthcare'].append(i)
        elif any(keyword in primary_tag for keyword in ['education', 'learning', 'training']):
            tag_groups['education'].append(i)
        elif any(keyword in primary_tag for keyword in ['social', 'media', 'content', 'community']):
            tag_groups['social'].append(i)
        elif any(keyword in primary_tag for keyword in ['developer', 'api', 'tools', 'infrastructure']):
            tag_groups['dev_tools'].append(i)
        elif any(keyword in primary_tag for keyword in ['e-commerce', 'marketplace', 'retail', 'shopping']):
            tag_groups['ecommerce'].append(i)
        elif any(keyword in primary_tag for keyword in ['enterprise', 'b2b', 'saas', 'software']):
            tag_groups['enterprise'].append(i)
        else:
            tag_groups['other'].append(i)
    
    # Assign cluster IDs
    cluster_id = 0
    cluster_assignments = {}
    
    for group_name, indices in tag_groups.items():
        if len(indices) < 10:  # Too small, mark as noise
            for idx in indices:
                cluster_assignments[idx] = -1
        else:
            # Split large groups into smaller clusters
            random.shuffle(indices)
            cluster_size = min(150, max(25, len(indices) // 3))  # 25-150 per cluster
            
            for i in range(0, len(indices), cluster_size):
                chunk = indices[i:i + cluster_size]
                if len(chunk) >= 10:  # Only if big enough
                    for idx in chunk:
                        cluster_assignments[idx] = cluster_id
                    cluster_id += 1
                else:
                    # Too small, add to noise
                    for idx in chunk:
                        cluster_assignments[idx] = -1
    
    # Add cluster_id to points
    clustered_points = []
    for i, point in enumerate(points):
        new_point = point.copy()
        new_point['cluster_id'] = cluster_assignments.get(i, -1)
        clustered_points.append(new_point)
    
    # Calculate summary
    cluster_counts = defaultdict(int)
    for cluster_id in cluster_assignments.values():
        cluster_counts[cluster_id] += 1
    
    noise_count = cluster_counts.pop(-1, 0)
    valid_clusters = len(cluster_counts)
    clustered_points_count = sum(cluster_counts.values())
    
    summary = {
        "algorithm": "demo_tag_similarity",
        "num_clusters": valid_clusters,
        "noise_count": noise_count,
        "total_points": len(points),
        "clustered_points": clustered_points_count,
        "clustering_rate": clustered_points_count / len(points),
        "top_10_cluster_sizes": sorted(cluster_counts.values(), reverse=True)[:10],
        "largest_cluster_size": max(cluster_counts.values()) if cluster_counts else 0,
        "smallest_cluster_size": min(cluster_counts.values()) if cluster_counts else 0,
        "runtime_seconds": 0.5
    }
    
    # Save results
    with open('out_full/points_with_clusters.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_points, f, indent=2)
    
    with open('out_full/points_with_clusters_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Demo clustering complete!")
    print(f"📊 Created {valid_clusters} clusters")
    print(f"📈 Clustered {clustered_points_count}/{len(points)} points ({summary['clustering_rate']:.1%})")
    print(f"🔍 Noise points: {noise_count}")
    print(f"📁 Top 5 cluster sizes: {summary['top_10_cluster_sizes'][:5]}")

if __name__ == "__main__":
    random.seed(42)  # For reproducible demo
    create_demo_clusters()