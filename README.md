# 🚀 YC Startup Map - Interactive Visualization

Interactive visualization of **5,410 Y Combinator startups** using ML embeddings and UMAP dimensionality reduction.

![Features](https://img.shields.io/badge/Startups-5410-blue) ![Status](https://img.shields.io/badge/Status-Active-green) ![Tech](https://img.shields.io/badge/Tech-ML%20%2B%20React-orange)

## 🚀 Quick Start for New Users

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/Sami-ul/HackMIT-2025
cd HackMIT-2025

# Python dependencies (ML models)
pip install -r requirements.txt

# Node.js dependencies (frontend)
npm install
```

### 2. Generate ML Model Files
```bash
# This creates large files excluded from git (~500MB total)
python generate_models.py
```
*⏱️ First run: 5-10 minutes to process all startups*

### 3. Start the Application
```bash
# Terminal 1: Backend API
python working_server.py

# Terminal 2: Frontend
npm run dev
```

### 4. Open Browser
Navigate to **http://localhost:3000**

## 📁 Generated Files (Not in Git)

Your peers will generate these locally using `generate_models.py`:

| File | Size | Purpose |
|------|------|---------|
| `umap_model.joblib` | 234MB | UMAP model for 2D coordinates |
| `fused.npy` | 140MB | Combined startup embeddings |
| `text_embeddings.npy` | 63MB | Text embeddings for similarity |
| `index_hnsw.bin` | 71MB | FAISS index for fast search |

## ✨ Features

- **5,410 YC startups** as interactive nodes
- **Company names** when zoomed in
- **Status colors**: Active (green), Acquired (blue), Inactive (gray)  
- **Interactive tooltips** with details and logos
- **Search and filtering** by name/tags
- **ML-powered positioning** using semantic similarity

## 🎨 Frontend Setup

### 1. Install Dependencies
```powershell
npm install
```

### 2. Start Development Server
```powershell
npm run dev
```

### 3. View the Map
- Homepage: http://localhost:3000
- Interactive Map: http://localhost:3000/map

## 🗺️ Map Features

### Interactive Visualization
- **deck.gl ScatterplotLayer** with orthographic view
- **Hover tooltips** showing startup name, one-liner, and tags
- **Click to find neighbors** - highlights similar startups for 3 seconds
- **Full viewport canvas** for smooth performance

### Sidebar Filters
- **Text search** - filter by name or tags
- **Tag checkboxes** - filter by specific categories
- **Real-time filtering** - instant client-side updates

### Add Startup Form
- **Modal form** with name, one-liner, description, tags
- **POST to /locate** endpoint
- **Animated placement** - spring animation from center to position
- **Neighbor highlighting** - shows temporary edges to similar startups
- **Automatic integration** - merges into main visualization after animation

## 🔧 Technical Stack

**Backend:**
- FastAPI + HNSW + OpenAI embeddings
- Real-time similarity search
- 3403-dimensional fused embeddings (text + tags)

**Frontend:**
- Next.js 14 with App Router
- deck.gl for WebGL visualization
- Framer Motion for animations
- Tailwind CSS for styling
- TypeScript for type safety

## 🎯 Usage Examples

### Test API Manually
```powershell
# Get all points
Invoke-WebRequest "http://127.0.0.1:8000/points"

# Find neighbors of startup #0
Invoke-WebRequest "http://127.0.0.1:8000/neighbors?id=0&k=10"

# Add new startup
$body = @{
    name = "My Robotics Co"
    one_liner = "Robotic pick-and-place"
    long_description = "We build advanced robotic systems for manufacturing automation"
    tags = @("Robotics", "Automation", "Manufacturing")
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://127.0.0.1:8000/locate" -Method POST -Body $body -ContentType "application/json"
```

### Frontend Features
1. **Explore the map** - pan, zoom, hover over startups
2. **Find similar companies** - click any point to see neighbors
3. **Filter by category** - use sidebar to filter by tags
4. **Search by name** - type in search box for instant results
5. **Add your startup** - click green button to add your company

## 🎨 Customization

### Colors & Styling
- Edit `app/map/page.tsx` for visualization settings
- Modify Tailwind classes for UI appearance
- Adjust `ScatterplotLayer` properties for point styles

### Animation Settings
- Modify Framer Motion springs in the component
- Adjust timing for neighbor highlighting
- Customize edge drawing for new startups

## 🚀 Ready to Launch!

Your complete startup similarity map is ready:
1. ✅ 5410 YC startups embedded and indexed
2. ✅ Fast similarity search with HNSW
3. ✅ Interactive visualization with deck.gl  
4. ✅ Real-time filtering and search
5. ✅ Add new startups with animations

**Start exploring:** http://localhost:3000/map