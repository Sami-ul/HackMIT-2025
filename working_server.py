#!/usr/bin/env python3
# working_server.py - Reliable FastAPI server for serving points data
import json
import os
import random
import numpy as np
import asyncio
import time
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import threading
import queue

# Pydantic models
class StartupForm(BaseModel):
    name: str
    one_liner: str
    long_description: str
    tags: List[str]

class Neighbor(BaseModel):
    id: int
    distance: float

class AnalysisRequest(BaseModel):
    startup_id: int
    startup_data: Dict[str, Any]

class CompetitorInfo(BaseModel):
    name: str
    description: str
    status: str
    why_failed: Optional[str] = None
    success_factors: Optional[str] = None

app = FastAPI(title="YC Startup Map API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache points data
_points_cache = None
_cluster_summary = None

# Analysis progress tracking
analysis_progress = {}
analysis_results = {}

# Embedding cache to avoid redundant API calls
@lru_cache(maxsize=1000)
def get_cached_embedding(text_hash: str, text: str, use_openai: bool = False):
    """Get embedding with caching to reduce API costs"""
    if not use_openai:
        # Try free local embedding first (if available)
        try:
            # Set environment variables to avoid TensorFlow conflicts
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
            
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Import with error handling for TensorFlow conflicts
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as import_err:
                if "tensorflow" in str(import_err).lower():
                    # TensorFlow conflict - skip local embedding
                    raise Exception("TensorFlow conflict detected, using fallback")
                raise import_err
            
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Force CPU
            embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
            result = embedding / np.linalg.norm(embedding)  # Normalize
            return result
        except Exception as e:
            # Silently fall back without verbose logging
            pass
    
    # OpenAI fallback (only when necessary)
    try:
        from openai import OpenAI
        # Load API key
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip()
        except:
            pass
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not available")
        
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model="text-embedding-3-large", input=text)
        embedding = np.array(response.data[0].embedding)
        return embedding / np.linalg.norm(embedding)  # Normalize
    except Exception as e:
        print(f"OpenAI embedding failed: {e}")
        # If all else fails, raise exception to trigger tag-only fallback
        raise Exception(f"All embedding methods failed: {e}")

def get_startup_embedding(text: str, prefer_free: bool = True):
    """Get embedding with smart caching and cost optimization"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return get_cached_embedding(text_hash, text, use_openai=not prefer_free)

def load_points_data():
    """Load points data from JSON file, preferring clustered version"""
    global _points_cache, _cluster_summary
    if _points_cache is None:
        # Try to load clustered points first, fall back to regular points
        clustered_path = "out_full/points_with_clusters.json"
        regular_path = "out_full/points.json"
        summary_path = "out_full/points_with_clusters_summary.json"
        
        points_path = clustered_path if os.path.exists(clustered_path) else regular_path
        
        try:
            with open(points_path, "r", encoding="utf-8") as f:
                _points_cache = json.load(f)
            
            # Load cluster summary if available
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        _cluster_summary = json.load(f)
                    print(f"✅ Loaded cluster summary: {_cluster_summary.get('num_clusters', 0)} clusters")
                except Exception as e:
                    print(f"⚠️ Failed to load cluster summary: {e}")
                    _cluster_summary = None
            
            # Check if points have cluster_id
            has_clusters = any('cluster_id' in point for point in _points_cache[:10])
            cluster_info = " (with clusters)" if has_clusters else " (no clusters)"
            
            print(f"✅ Loaded {len(_points_cache)} points successfully{cluster_info}")
            
        except Exception as e:
            print(f"❌ Error loading points: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load points: {e}")
    return _points_cache

@app.on_event("startup")
def startup_event():
    """Load data on startup"""
    load_points_data()
    print("🚀 Server started successfully!")

@app.get("/")
def root():
    return {"message": "YC Startup Map API", "status": "running"}

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        data = load_points_data()
        has_clusters = any('cluster_id' in point for point in data[:10]) if data else False
        return {
            "status": "healthy",
            "points_count": len(data),
            "sample_point_keys": list(data[0].keys()) if data else [],
            "points_with_logos": sum(1 for p in data if p.get('logo_url')),
            "has_clusters": has_clusters,
            "cluster_summary": _cluster_summary
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/points")
def get_points():
    """Get all startup points data (includes cluster_id if available)"""
    try:
        data = load_points_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clusters/summary")
def get_cluster_summary():
    """Get clustering summary statistics"""
    try:
        # Ensure data is loaded (which loads cluster summary)
        load_points_data()
        
        if _cluster_summary is None:
            return {
                "status": "no_clusters",
                "message": "No clustering data available. Run cluster_embeddings.py first.",
                "num_clusters": 0,
                "clustered_points": 0,
                "clustering_rate": 0.0
            }
        
        return {
            "status": "available",
            **_cluster_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neighbors")
def get_neighbors(id: int, k: int = 8):
    """Get k nearest neighbors for a given startup ID"""
    try:
        data = load_points_data()
        
        # Find the target point
        target_point = None
        for point in data:
            if point.get('id') == id:
                target_point = point
                break
        
        if not target_point:
            raise HTTPException(status_code=404, detail=f"Startup with id {id} not found")
        
        # Calculate distances to all other points
        neighbors = []
        target_x, target_y = target_point.get('x', 0), target_point.get('y', 0)
        
        for point in data:
            if point.get('id') == id:
                continue  # Skip the target point itself
            
            point_x, point_y = point.get('x', 0), point.get('y', 0)
            distance = np.sqrt((target_x - point_x)**2 + (target_y - point_y)**2)
            neighbors.append({
                'id': point.get('id'),
                'distance': float(distance)
            })
        
        # Sort by distance and return top k
        neighbors.sort(key=lambda x: x['distance'])
        return neighbors[:k]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/locate")
def locate_startup(startup: StartupForm):
    """Locate where a new startup should be positioned using SMART embeddings (COST OPTIMIZED - 1 API call instead of 50+)"""
    try:
        data = load_points_data()
        
        if not data:
            raise HTTPException(status_code=500, detail="No existing data to use for positioning")
        
        # Build text for the new startup (same format as the embedding pipeline)
        startup_text = f"{startup.name}\n{startup.one_liner}\n{startup.long_description}\nTags: {', '.join(startup.tags)}"
        
        # Get embedding for the NEW startup only (1 API call instead of 50+)
        try:
            new_embedding = get_startup_embedding(startup_text, prefer_free=True)
            print(f"✅ Successfully embedded startup: {startup.name}")
        except Exception as e:
            print(f"⚠️ Embedding failed for {startup.name}, using tag-based fallback: {e}")
            # Fallback to tag-only approach if embedding fails
            return locate_startup_tags_only(startup, data)
        
        startup_tags = set(startup.tags)
        
        # First pass: Filter candidates by tag overlap (this is cheap and effective)
        tag_candidates = []
        for point in data:
            point_tags = set(point.get('tags', []))
            if point_tags and startup_tags:
                # Calculate tag similarity (Jaccard similarity)
                intersection = len(startup_tags.intersection(point_tags))
                union = len(startup_tags.union(point_tags))
                tag_similarity = intersection / union if union > 0 else 0
                
                if tag_similarity > 0:  # Any tag overlap
                    tag_candidates.append({
                        'point': point,
                        'tag_similarity': tag_similarity,
                        'shared_tags': list(startup_tags.intersection(point_tags))
                    })
        
        # Sort by tag similarity and take top candidates for embedding comparison
        tag_candidates.sort(key=lambda x: x['tag_similarity'], reverse=True)
        top_candidates = tag_candidates[:20]  # Only compare against top 20 instead of all 5000+
        
        # If no tag overlap, use broader similarity search
        if not top_candidates:
            # Look for startups with ANY related keywords in their descriptions
            startup_keywords = set(startup_text.lower().split())
            for point in data:
                point_text = f"{point.get('name', '')} {point.get('one_liner', '')} {point.get('long_description', '')}"
                point_keywords = set(point_text.lower().split())
                
                # Find common keywords (simple but effective)
                common_keywords = startup_keywords.intersection(point_keywords)
                if len(common_keywords) >= 2:  # At least 2 common words
                    keyword_similarity = len(common_keywords) / len(startup_keywords.union(point_keywords))
                    top_candidates.append({
                        'point': point,
                        'tag_similarity': 0,
                        'keyword_similarity': keyword_similarity,
                        'shared_tags': []
                    })
            
            # Take top keyword matches
            top_candidates.sort(key=lambda x: x.get('keyword_similarity', 0), reverse=True)
            top_candidates = top_candidates[:20]
        
        # Now do semantic similarity ONLY on the filtered candidates (much cheaper!)
        best_matches = []
        for candidate in top_candidates:
            point = candidate['point']
            
            # Use SAME text template as embedding pipeline (critical for consistency)
            point_text = f"{point.get('name', '')}\n{point.get('one_liner', '')}\n{point.get('long_description', '')}\nTags: {', '.join(point.get('tags', []))}"
            
            try:
                # Get cached embedding for this point (or compute if not cached)
                point_embedding = get_startup_embedding(point_text, prefer_free=True)
                
                # Calculate cosine similarity
                semantic_similarity = float(np.dot(new_embedding, point_embedding))
                
                # Combine semantic similarity with tag similarity
                combined_similarity = 0.6 * semantic_similarity + 0.4 * candidate['tag_similarity']
                
                best_matches.append({
                    'point': point,
                    'similarity': combined_similarity,
                    'semantic_similarity': semantic_similarity,
                    'tag_similarity': candidate['tag_similarity'],
                    'shared_tags': candidate['shared_tags']
                })
            except Exception as e:
                # If individual embedding fails, just use tag similarity
                best_matches.append({
                    'point': point,
                    'similarity': candidate['tag_similarity'],
                    'semantic_similarity': 0,
                    'tag_similarity': candidate['tag_similarity'],
                    'shared_tags': candidate['shared_tags']
                })
        
        # Sort by combined similarity and take top matches
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = best_matches[:30] if best_matches else []
        
        # Final fallback if no good matches
        if not top_matches:
            print("No semantic matches found, using random sampling")
            random_sample = random.sample(data, min(30, len(data)))
            top_matches = [{'point': p, 'similarity': 0.1, 'shared_tags': []} for p in random_sample]
        
        # Calculate weighted average position based on similarity
        weights = [max(match['similarity'], 0.01) for match in top_matches]
        total_weight = sum(weights)
        
        weighted_x = sum(match['point'].get('x', 0) * weights[i] for i, match in enumerate(top_matches)) / total_weight
        weighted_y = sum(match['point'].get('y', 0) * weights[i] for i, match in enumerate(top_matches)) / total_weight
        
        # Add some randomness to avoid exact overlap
        x = weighted_x + random.uniform(-1.5, 1.5)
        y = weighted_y + random.uniform(-1.5, 1.5)
        
        # Find nearest neighbors for the new position
        neighbors = []
        for point in data:
            point_x, point_y = point.get('x', 0), point.get('y', 0)
            distance = np.sqrt((x - point_x)**2 + (y - point_y)**2)
            neighbors.append({
                'row': point.get('id'),
                'score': max(0, 1.0 - distance / 100),  # Normalize score
                'name': point.get('name', '')
            })
        
        # Sort by distance and return top neighbors
        neighbors.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'x': float(x),
            'y': float(y),  
            'neighbors': neighbors[:8]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def locate_startup_tags_only(startup: StartupForm, data: List[Dict]) -> Dict:
    """Fallback positioning using only tag similarity (FREE backup method)"""
    try:
        startup_tags = set(startup.tags)
        
        # Find best tag matches
        best_matches = []
        for point in data:
            try:
                point_tags = set(point.get('tags', []))
                if point_tags and startup_tags:
                    jaccard = len(startup_tags & point_tags) / len(startup_tags | point_tags)
                    if jaccard > 0.05:  # Lower threshold
                        best_matches.append({'point': point, 'similarity': jaccard})
            except Exception:
                continue  # Skip problematic points
        
        # Sort and take top matches
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Ensure we have matches
        if not best_matches:
            # If no tag matches, use keyword similarity
            try:
                startup_keywords = set(f"{startup.name} {startup.one_liner} {startup.long_description}".lower().split())
                for point in data:
                    try:
                        point_keywords = set(f"{point.get('name', '')} {point.get('one_liner', '')} {point.get('long_description', '')}".lower().split())
                        common = startup_keywords & point_keywords
                        if len(common) >= 1:
                            similarity = len(common) / len(startup_keywords | point_keywords)
                            best_matches.append({'point': point, 'similarity': similarity})
                    except Exception:
                        continue
                
                best_matches.sort(key=lambda x: x['similarity'], reverse=True)
            except Exception:
                pass  # Continue to random fallback
        
        # Final fallback - random selection
        if not best_matches:
            try:
                random_points = random.sample(data, min(30, len(data)))
                best_matches = [{'point': p, 'similarity': 0.1} for p in random_points]
            except Exception:
                # Absolute fallback - use first 30 points
                best_matches = [{'point': p, 'similarity': 0.1} for p in data[:30]]
        
        top_matches = best_matches[:30]
        
        # Calculate position with error handling
        try:
            weights = [max(match['similarity'], 0.01) for match in top_matches]
            total_weight = sum(weights)
            
            if total_weight > 0:
                weighted_x = sum(match['point'].get('x', 0) * weights[i] for i, match in enumerate(top_matches)) / total_weight
                weighted_y = sum(match['point'].get('y', 0) * weights[i] for i, match in enumerate(top_matches)) / total_weight
            else:
                # Use center position if no weights
                weighted_x = sum(match['point'].get('x', 0) for match in top_matches) / len(top_matches)
                weighted_y = sum(match['point'].get('y', 0) for match in top_matches) / len(top_matches)
            
            x = weighted_x + random.uniform(-2.0, 2.0)
            y = weighted_y + random.uniform(-2.0, 2.0)
        except Exception:
            # Emergency fallback position
            x, y = 0.0, 0.0
        
        # Find neighbors with error handling
        neighbors = []
        try:
            for point in data:
                try:
                    point_x, point_y = point.get('x', 0), point.get('y', 0)
                    distance = np.sqrt((x - point_x)**2 + (y - point_y)**2)
                    neighbors.append({
                        'row': point.get('id'),
                        'score': max(0, 1.0 - distance / 100),
                        'name': point.get('name', '')
                    })
                except Exception:
                    continue
            
            neighbors.sort(key=lambda x: x['score'], reverse=True)
        except Exception:
            neighbors = []  # Empty neighbors if all fails
        
        return {
            'x': float(x),
            'y': float(y),
            'neighbors': neighbors[:8]
        }
        
    except Exception as e:
        # Absolute emergency fallback
        return {
            'x': 0.0,
            'y': 0.0,
            'neighbors': []
        }

def format_analysis_html(text: str) -> str:
    """Format the analysis text as HTML with proper styling"""
    lines = text.split('\n')
    formatted_lines = []
    in_ordered_list = False
    in_unordered_list = False
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            # Close any open lists for empty lines
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            formatted_lines.append('<br>')
            continue
        
        # Handle headers
        if stripped.startswith('### '):
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            header_text = stripped[4:]  # Remove ### 
            formatted_lines.append(f'<h3 class="text-lg font-semibold text-gray-700 mb-2 mt-4">{header_text}</h3>')
        elif stripped.startswith('## '):
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            header_text = stripped[3:]  # Remove ## 
            formatted_lines.append(f'<h2 class="text-xl font-semibold text-blue-600 mb-3 mt-5">{header_text}</h2>')
        elif stripped.startswith('# '):
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            header_text = stripped[2:]  # Remove # 
            formatted_lines.append(f'<h1 class="text-2xl font-bold text-gray-800 mb-4 mt-6">{header_text}</h1>')
        
        # Handle numbered lists
        elif stripped and len(stripped) > 2 and stripped[0].isdigit() and '. ' in stripped:
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            if not in_ordered_list:
                formatted_lines.append('<ol class="list-decimal list-inside space-y-2 mb-4 ml-4">')
                in_ordered_list = True
            list_content = stripped[stripped.find('. ') + 2:]
            formatted_lines.append(f'<li class="mb-1">{list_content}</li>')
        
        # Handle bullet points
        elif stripped.startswith('- '):
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if not in_unordered_list:
                formatted_lines.append('<ul class="list-disc list-inside space-y-1 mb-4 ml-4">')
                in_unordered_list = True
            list_content = stripped[2:]
            formatted_lines.append(f'<li class="mb-1">{list_content}</li>')
        
        # Handle regular paragraphs
        else:
            if in_ordered_list:
                formatted_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                formatted_lines.append('</ul>')
                in_unordered_list = False
            formatted_lines.append(f'<p class="mb-3 text-gray-700 leading-relaxed">{stripped}</p>')
    
    # Close any remaining open lists
    if in_ordered_list:
        formatted_lines.append('</ol>')
    if in_unordered_list:
        formatted_lines.append('</ul>')
    
    html = '\n'.join(formatted_lines)
    
    # Format bold text (handle ** pairs properly)
    import re
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong class="font-semibold text-gray-800">\1</strong>', html)
    
    return html

async def web_search_agent(company_name: str, description: str) -> Dict[str, Any]:
    """Simulate web search for company information"""
    # In a real implementation, this would use a web search API like Serper, Tavily, or Bing
    await asyncio.sleep(2)  # Simulate search time
    
    # Mock search results - in reality this would be actual web search
    return {
        "name": company_name,
        "current_status": "active",  # This would be determined from search results
        "recent_news": f"Recent developments for {company_name}...",
        "funding_info": "Series B, $25M raised",
        "market_position": "Strong competitor in the space"
    }

async def claude_analysis_agent(startup_data: Dict[str, Any], competitors_info: List[Dict[str, Any]]) -> str:
    """Use Claude to analyze competitors and generate insights"""
    try:
        # Try to load from .env file if available
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('ANTHROPIC_API_KEY='):
                        os.environ['ANTHROPIC_API_KEY'] = line.split('=', 1)[1].strip()
        except:
            pass
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "Analysis unavailable: ANTHROPIC_API_KEY not set"
        
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare the analysis prompt with tag context
        competitors_text = "\n\n".join([
            f"**{comp['name']}**: {comp.get('recent_news', 'No recent news')}\n"
            f"Status: {comp.get('current_status', 'Unknown')}\n"
            f"Market Position: {comp.get('market_position', 'Unknown')}\n"
            f"Shared Tags: {', '.join(comp.get('shared_tags', []))}\n"
            f"Tag Similarity: {comp.get('tag_similarity', 0):.2f}"
            for comp in competitors_info
        ])
        
        prompt = f"""
You are an expert startup advisor and venture capitalist analyzing the competitive landscape. Here's the startup seeking advice:

## Target Startup
**Name**: {startup_data['name']}
**One-liner**: {startup_data['one_liner']}
**Description**: {startup_data['long_description']}
**Market Tags**: {', '.join(startup_data['tags'])}

## Top 5 Similar Competitors
{competitors_text}

## Analysis Request
Provide a comprehensive strategic analysis with the following sections:

### 1. Market Landscape Assessment
- Market size and growth potential
- Key competitive dynamics
- Barriers to entry and moats
- Timing and market readiness

### 2. Competitor Success Analysis
- What the successful competitors did right
- Their key differentiators and value propositions
- Revenue models and unit economics
- Go-to-market strategies that worked

### 3. Failure Analysis & Risk Mitigation
- Why competitors failed or struggled
- Common mistakes and pitfalls in this space
- Red flags and warning signs to monitor
- How to avoid these failure modes

### 4. Strategic Positioning Recommendations
- Unique positioning opportunities
- Target customer segments to focus on
- Key value propositions to emphasize
- Competitive advantages to build

### 5. 90-Day Action Plan
- Week 1-2: Immediate priorities
- Month 1: Core product/market validation
- Month 2: Go-to-market foundation
- Month 3: Scale preparation

### 6. Success Metrics & Milestones
- Key metrics to track weekly/monthly
- Milestone targets for next 6 months
- Signs you're on the right track
- When to pivot or adjust strategy

Be specific, actionable, and data-driven. Focus on practical steps the founder can implement immediately.
"""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Analysis error: {str(e)}"

async def run_competitor_analysis(startup_id: int, startup_data: Dict[str, Any]):
    """Main analysis workflow"""
    try:
        # Update progress
        analysis_progress[startup_id] = "Finding similar competitors based on tags and content..."
        
        # Get all existing startups
        data = load_points_data()
        
        # Find competitors using tag similarity and content matching
        startup_tags = set(startup_data.get('tags', []))
        startup_text = f"{startup_data.get('name', '')} {startup_data.get('one_liner', '')} {startup_data.get('long_description', '')}"
        
        # Calculate similarity with all existing startups
        competitor_matches = []
        for point in data:
            point_tags = set(point.get('tags', []))
            
            # Calculate tag similarity
            if point_tags and startup_tags:
                intersection = len(startup_tags.intersection(point_tags))
                union = len(startup_tags.union(point_tags))
                tag_similarity = intersection / union if union > 0 else 0
            else:
                tag_similarity = 0
            
            # Calculate content similarity (simple keyword matching)
            point_text = f"{point.get('name', '')} {point.get('one_liner', '')} {point.get('long_description', '')}"
            common_words = set(startup_text.lower().split()) & set(point_text.lower().split())
            content_similarity = len(common_words) / max(len(startup_text.split()), len(point_text.split())) if point_text else 0
            
            # Combined similarity score
            combined_similarity = 0.7 * tag_similarity + 0.3 * content_similarity
            
            if combined_similarity > 0.1:  # Include any meaningful similarity
                competitor_matches.append({
                    'point': point,
                    'similarity': combined_similarity,
                    'tag_similarity': tag_similarity,
                    'shared_tags': list(startup_tags.intersection(point_tags))
                })
        
        # Sort by similarity and take top 5
        competitor_matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_5_competitors = competitor_matches[:5]
        
        # If no similar competitors found, take random ones for general market analysis
        if not top_5_competitors:
            analysis_progress[startup_id] = "No direct competitors found, analyzing general market..."
            random_competitors = random.sample(data, min(5, len(data)))
            top_5_competitors = [{
                'point': comp,
                'similarity': 0.1,
                'tag_similarity': 0,
                'shared_tags': []
            } for comp in random_competitors]
        
        # Research each competitor with tag context
        competitors_info = []
        for i, competitor_match in enumerate(top_5_competitors):
            competitor = competitor_match['point']
            analysis_progress[startup_id] = f"Researching competitor {i+1}/5: {competitor.get('name', 'Unknown')}..."
            
            search_result = await web_search_agent(
                competitor.get('name', ''), 
                competitor.get('one_liner', '')
            )
            
            # Add the pre-calculated tag information to the search result
            search_result['shared_tags'] = competitor_match['shared_tags']
            search_result['tag_similarity'] = competitor_match['tag_similarity']
            search_result['overall_similarity'] = competitor_match['similarity']
            
            competitors_info.append(search_result)
        
        # Generate analysis
        analysis_progress[startup_id] = "Generating strategic analysis..."
        
        analysis_report = await claude_analysis_agent(startup_data, competitors_info)
        
        # Format as HTML for better display with improved styling
        html_report = format_analysis_html(analysis_report)
        
        # Store results
        analysis_results[startup_id] = html_report
        analysis_progress[startup_id] = "Analysis complete!"
        
    except Exception as e:
        analysis_progress[startup_id] = f"Error: {str(e)}"

@app.post("/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start competitor analysis for a startup"""
    startup_id = request.startup_id
    
    # Initialize progress tracking
    analysis_progress[startup_id] = "Starting analysis..."
    analysis_results[startup_id] = None
    
    # Start analysis in background
    background_tasks.add_task(run_competitor_analysis, startup_id, request.startup_data)
    
    return {"message": "Analysis started", "startup_id": startup_id}

@app.get("/analysis-progress/{startup_id}")
async def get_analysis_progress(startup_id: int):
    """Stream analysis progress using Server-Sent Events"""
    async def generate():
        while True:
            progress = analysis_progress.get(startup_id, "Not started")
            results = analysis_results.get(startup_id)
            
            if results:
                # Analysis is complete
                yield f"data: {json.dumps({'progress': progress, 'completed': True, 'results': results})}\n\n"
                break
            elif "Error:" in progress or "complete!" in progress:
                # Analysis failed or completed without results
                yield f"data: {json.dumps({'progress': progress, 'completed': True, 'results': progress})}\n\n"
                break
            else:
                # Analysis in progress
                yield f"data: {json.dumps({'progress': progress, 'completed': False})}\n\n"
                await asyncio.sleep(2)  # Update every 2 seconds
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=False)