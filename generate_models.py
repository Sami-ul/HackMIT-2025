#!/usr/bin/env python3
"""
Generate ML model files for YC Startup Map
Creates the large files that are excluded from git.
"""

import json
import numpy as np
import os
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
import umap
import faiss
import argparse
from tqdm import tqdm

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path("out_full")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_startup_data():
    """Load YC startup data from JSON."""
    print("📊 Loading YC startup data...")
    with open("ycombinator_startups.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} startups")
    return data

def generate_embeddings(data, batch_size=100):
    """Generate text embeddings for startup descriptions."""
    output_dir = ensure_output_dir()
    embeddings_path = output_dir / "text_embeddings.npy"
    
    if embeddings_path.exists():
        print("📂 Loading existing embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"✅ Loaded embeddings: {embeddings.shape}")
        return embeddings
    
    print("🤖 Generating text embeddings (this may take 5-10 minutes)...")
    
    # Use a lightweight but good model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare text data
    texts = []
    for startup in data:
        # Combine name, one-liner, and tags for rich embeddings
        text_parts = [
            startup.get('name', ''),
            startup.get('one_liner', ''),
            ' '.join(startup.get('tags', []))
        ]
        text = ' '.join(filter(None, text_parts))
        texts.append(text)
    
    # Generate embeddings in batches to manage memory
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    print(f"✅ Generated embeddings: {embeddings.shape} -> {embeddings_path}")
    
    return embeddings

def create_umap_model(embeddings):
    """Create and train UMAP model for 2D visualization."""
    output_dir = ensure_output_dir()
    umap_path = output_dir / "umap_model.joblib"
    fused_path = output_dir / "fused.npy"
    
    if umap_path.exists() and fused_path.exists():
        print("📂 Loading existing UMAP model...")
        umap_model = joblib.load(umap_path)
        fused_embeddings = np.load(fused_path)
        print(f"✅ Loaded UMAP model and fused embeddings: {fused_embeddings.shape}")
        return umap_model, fused_embeddings
    
    print("🗺️  Creating UMAP model for 2D visualization...")
    
    # Configure UMAP for good visualization
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        spread=1.0,
        random_state=42,
        metric='cosine'
    )
    
    # Fit and transform
    print("🔄 Fitting UMAP (this may take 2-3 minutes)...")
    fused_embeddings = umap_model.fit_transform(embeddings)
    
    # Scale coordinates for better visualization
    fused_embeddings = fused_embeddings * 2  # Scale up for deck.gl
    
    # Save model and embeddings
    joblib.dump(umap_model, umap_path)
    np.save(fused_path, fused_embeddings)
    
    print(f"✅ Created UMAP model -> {umap_path}")
    print(f"✅ Generated 2D coordinates -> {fused_path}")
    
    return umap_model, fused_embeddings

def create_search_index(embeddings):
    """Create FAISS HNSW index for similarity search."""
    output_dir = ensure_output_dir()
    index_path = output_dir / "index_hnsw.bin"
    
    if index_path.exists():
        print("📂 Loading existing search index...")
        index = faiss.read_index(str(index_path))
        print(f"✅ Loaded search index with {index.ntotal} vectors")
        return index
    
    print("🔍 Creating FAISS search index...")
    
    # Create HNSW index for fast similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    
    # Add embeddings to index
    print("📊 Adding embeddings to index...")
    index.add(embeddings.astype('float32'))
    
    # Save index
    faiss.write_index(index, str(index_path))
    print(f"✅ Created search index -> {index_path}")
    
    return index

def generate_points_json(data, coordinates):
    """Generate the final points.json file with all data."""
    output_dir = ensure_output_dir()
    points_path = output_dir / "points.json"
    
    if points_path.exists():
        print("📂 points.json already exists, skipping...")
        return
    
    print("📄 Generating points.json...")
    
    points = []
    for i, startup in enumerate(data):
        point = {
            "id": startup.get('id', i),
            "name": startup.get('name', 'Unknown'),
            "x": float(coordinates[i, 0]),
            "y": float(coordinates[i, 1]),
            "tags": startup.get('tags', []),
            "status": startup.get('status', 'Unknown'),
            "stage": startup.get('stage', 'Unknown'),
            "one_liner": startup.get('one_liner', ''),
            "long_description": startup.get('long_description', ''),
            "batch": startup.get('batch', ''),
            "logo_url": startup.get('small_logo_thumb_url', '')
        }
        points.append(point)
    
    # Save points
    with open(points_path, 'w', encoding='utf-8') as f:
        json.dump(points, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated points.json with {len(points)} startups -> {points_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ML model files for YC Startup Map")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation")
    parser.add_argument("--force", action="store_true", help="Regenerate all files even if they exist")
    args = parser.parse_args()
    
    if args.force:
        print("🔄 Force regeneration enabled - will recreate all files")
        import shutil
        output_dir = Path("out_full")
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    print("🚀 Starting YC Startup Map model generation...\n")
    
    try:
        # Step 1: Load data
        data = load_startup_data()
        
        # Step 2: Generate embeddings
        embeddings = generate_embeddings(data, batch_size=args.batch_size)
        
        # Step 3: Create UMAP model
        umap_model, coordinates = create_umap_model(embeddings)
        
        # Step 4: Create search index
        search_index = create_search_index(embeddings)
        
        # Step 5: Generate final points.json
        generate_points_json(data, coordinates)
        
        print("\n🎉 Model generation complete!")
        print("\nGenerated files:")
        output_dir = Path("out_full")
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  📁 {file_path.name} ({size_mb:.1f} MB)")
        
        print("\n✅ Ready to start the application!")
        print("   Backend: python working_server.py")
        print("   Frontend: npm run dev")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("Try running with --batch-size 50 if you have memory issues")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())