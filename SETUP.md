# YC Startup Map Setup Guide

This project visualizes Y Combinator startups using machine learning embeddings and UMAP dimensionality reduction.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

2. **Generate ML Model Files**
   ```bash
   python generate_models.py
   ```

3. **Start the Application**
   ```bash
   # Terminal 1: Start backend
   python working_server.py
   
   # Terminal 2: Start frontend
   npm run dev
   ```

4. **Open Browser**
   Navigate to http://localhost:3000

## What the Setup Script Does

The `generate_models.py` script will create the following large files (excluded from git):

- **umap_model.joblib** (234MB) - UMAP model for 2D coordinate generation
- **fused.npy** (140MB) - Combined embeddings from company data
- **text_embeddings.npy** (63MB) - Text embeddings for company descriptions
- **index_hnsw.bin** (71MB) - HNSW index for similarity search
- **out_full/points.json** - Final processed data with coordinates

## System Requirements

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **8GB+ RAM** (for processing embeddings)
- **2GB+ free disk space** (for generated model files)

## Troubleshooting

### Memory Issues
If you get out-of-memory errors:
```bash
# Process in smaller batches
python generate_models.py --batch-size 1000
```

### Missing Dependencies
```bash
# Install ML dependencies
pip install sentence-transformers umap-learn faiss-cpu joblib numpy

# Install web dependencies  
npm install @deck.gl/react @deck.gl/layers @deck.gl/core
```

### API Timeout
If the backend takes too long to start:
- The first run processes 5,410 startups and may take 2-3 minutes
- Subsequent runs load from cache and start in seconds

## File Structure

```
HackMIT-2025/
├── working_server.py          # FastAPI backend
├── app/map/page.tsx          # React frontend
├── ycombinator_startups.json # Raw YC data
├── generate_models.py        # ML model generation
├── out_full/                 # Generated files (excluded from git)
│   ├── points.json           # Processed coordinates
│   ├── umap_model.joblib     # UMAP model
│   ├── fused.npy            # Combined embeddings
│   └── ...                  # Other generated files
└── requirements.txt         # Python dependencies
```

## Development

The visualization shows:
- **5,410 YC startups** as colored nodes
- **Company names** when zoomed in
- **Status colors**: Active (green), Acquired (blue), Inactive (gray)
- **Interactive tooltips** with company details
- **Search and filtering** by tags/names

## Data Sources

- YC startup data from official YC API
- Company embeddings using sentence-transformers
- 2D coordinates via UMAP dimensionality reduction
- Similarity search using FAISS HNSW index