# 🚀 Startup Map - Complete Setup Guide

## ✅ Backend Server (Already Working!)

Your FastAPI server is running at `http://127.0.0.1:8000` with:
- ✅ OpenAI embeddings (`text-embedding-3-large`)
- ✅ 5410 startup embeddings loaded
- ✅ HNSW index for fast similarity search
- ✅ CORS enabled for frontend

**API Endpoints:**
- `GET /points` - All startup points with coordinates
- `GET /neighbors?id=0&k=10` - Find similar startups
- `POST /locate` - Add new startup and get position

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