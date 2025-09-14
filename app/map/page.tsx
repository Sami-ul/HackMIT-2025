'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, LineLayer, TextLayer, IconLayer } from '@deck.gl/layers';
import { OrthographicView } from '@deck.gl/core';
import { motion, useSpring, useTransform } from 'framer-motion';

interface Point {
  id: number;
  name: string;
  x: number;
  y: number;
  tags: string[];
  one_liner?: string;
  long_description?: string;
  status?: string;
  batch?: string;
  stage?: string;
  logo_url?: string;
}

interface Neighbor {
  id: number;
  distance: number;
}

interface StartupForm {
  name: string;
  one_liner: string;
  long_description: string;
  tags: string[];
}

export default function MapPage() {
  const [points, setPoints] = useState<Point[]>([]);
  const [filteredPoints, setFilteredPoints] = useState<Point[]>([]);
  const [highlightedPoints, setHighlightedPoints] = useState<Set<number>>(new Set());
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);
  const [textFilter, setTextFilter] = useState('');
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [allTags, setAllTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Performance optimizations
  const [maxPoints, setMaxPoints] = useState(1500); // Limit initial render
  const [zoom, setZoom] = useState(1);
  const [viewState, setViewState] = useState({ target: [0, 0, 0], zoom: 1 });
  
  // Visual enhancements
  const [showTagConnections, setShowTagConnections] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [showClusterLabels, setShowClusterLabels] = useState(true);
  
  // Form state
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState<StartupForm>({
    name: '',
    one_liner: '',
    long_description: '',
    tags: []
  });
  const [newTagInput, setNewTagInput] = useState('');
  const [animatingPoint, setAnimatingPoint] = useState<Point | null>(null);
  const [tempEdges, setTempEdges] = useState<{from: Point, to: Point}[]>([]);

  // Animation springs
  const animX = useSpring(0);
  const animY = useSpring(0);

  // Color schemes based on status
  const getStatusColor = (point: Point): [number, number, number, number] => {
    const status = point.status?.toLowerCase() || 'unknown';
    // Debug first few points to check status values
    if (Math.random() < 0.001) console.log('Point status:', point.name, point.status, status);
    switch (status) {
      case 'active':
        return [34, 197, 94, 255]; // Green
      case 'inactive':
        return [156, 163, 175, 200]; // Gray
      case 'acquired':
        return [168, 85, 247, 255]; // Purple
      case 'public':
        return [59, 130, 246, 255]; // Blue
      case 'dead':
        return [239, 68, 68, 200]; // Red
      default:
        return [251, 191, 36, 255]; // Yellow/Orange for unknown
    }
  };



  // Fetch points on mount
  useEffect(() => {
    fetchPoints();
  }, []);

  const fetchPoints = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8002/points');
      if (!response.ok) throw new Error('Failed to fetch points');
      
      const rawData: Point[] = await response.json();
      
      // Scale up coordinates to spread nodes out more (2x spread instead of 3x)
      const data = rawData.map(point => ({
        ...point,
        x: point.x * 2,
        y: point.y * 2
      }));
      
      setPoints(data);
      setFilteredPoints(data);
      
      // Extract unique tags
      const tags = new Set<string>();
      data.forEach(point => point.tags?.forEach(tag => tags.add(tag)));
      setAllTags(Array.from(tags).sort());
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  // Optimized filtering with performance limits
  const displayPoints = useMemo(() => {
    let filtered = points;
    console.log(`displayPoints calc: points=${points.length}, maxPoints=${maxPoints}, zoom=${zoom}`);
    
    // Text filter
    if (textFilter) {
      const query = textFilter.toLowerCase();
      filtered = filtered.filter(point => 
        point.name.toLowerCase().includes(query) ||
        point.tags?.some((tag: string) => tag.toLowerCase().includes(query))
      );
    }
    
    // Tag filter  
    if (selectedTags.size > 0) {
      filtered = filtered.filter(point =>
        point.tags?.some((tag: string) => selectedTags.has(tag))
      );
    }
    
    // Performance limit: show max points based on zoom level
    const limit = zoom > 3 ? maxPoints * 2 : maxPoints;
    const result = filtered.slice(0, limit);
    console.log(`displayPoints result: ${result.length} out of ${filtered.length} filtered, limit=${limit}`);
    return result;
  }, [points, textFilter, selectedTags, maxPoints, zoom]);
  
  // Update filtered points for stats
  useEffect(() => {
    setFilteredPoints(displayPoints);
  }, [displayPoints]);

  // Generate tag connections
  const tagConnections = useMemo(() => {
    if (!showTagConnections || displayPoints.length < 2) return [];
    
    const connections: {source: Point, target: Point, sharedTags: string[]}[] = [];
    const maxConnections = 100; // Limit for performance
    
    for (let i = 0; i < displayPoints.length && connections.length < maxConnections; i++) {
      const point1 = displayPoints[i];
      for (let j = i + 1; j < displayPoints.length && connections.length < maxConnections; j++) {
        const point2 = displayPoints[j];
        
        // Find shared tags
        const sharedTags = point1.tags?.filter(tag => point2.tags?.includes(tag)) || [];
        
        // Only connect if they share 2+ tags and are close enough
        if (sharedTags.length >= 2) {
          const distance = Math.sqrt(
            Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2)
          );
          if (distance < 30) { // Only connect nearby points
            connections.push({source: point1, target: point2, sharedTags});
          }
        }
      }
    }
    return connections;
  }, [displayPoints, showTagConnections]);

  // Cluster naming algorithm
  const clusterLabels = useMemo(() => {
    if (displayPoints.length < 10) return [];
    
    // Simple grid-based clustering
    const gridSize = 8; // Divide space into 8x8 grid
    const minX = Math.min(...displayPoints.map(p => p.x));
    const maxX = Math.max(...displayPoints.map(p => p.x));
    const minY = Math.min(...displayPoints.map(p => p.y));
    const maxY = Math.max(...displayPoints.map(p => p.y));
    
    const cellWidth = (maxX - minX) / gridSize;
    const cellHeight = (maxY - minY) / gridSize;
    
    // Group points by grid cell
    const clusters: {[key: string]: Point[]} = {};
    
    displayPoints.forEach(point => {
      const cellX = Math.floor((point.x - minX) / cellWidth);
      const cellY = Math.floor((point.y - minY) / cellHeight);
      const key = `${cellX}-${cellY}`;
      
      if (!clusters[key]) clusters[key] = [];
      clusters[key].push(point);
    });
    
    // Generate labels for clusters with enough points
    const labels: {x: number, y: number, text: string}[] = [];
    
    Object.entries(clusters).forEach(([key, clusterPoints]) => {
      if (clusterPoints.length < 5) return; // Skip small clusters
      
      // Count tag frequencies
      const tagCounts: {[tag: string]: number} = {};
      clusterPoints.forEach(point => {
        point.tags?.forEach(tag => {
          tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        });
      });
      
      // Get top 2 most common tags
      const topTags = Object.entries(tagCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 2)
        .map(([tag]) => tag);
      
      if (topTags.length > 0) {
        // Calculate cluster center
        const centerX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
        const centerY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
        
        const label = topTags.length === 1 
          ? `${topTags[0]} Hub` 
          : `${topTags[0]} & ${topTags[1]}`;
        
        labels.push({
          x: centerX,
          y: centerY,
          text: label
        });
      }
    });
    
    return labels;
  }, [displayPoints]);

  const handlePointClick = useCallback(async (info: any) => {
    if (!info.object) return;
    
    const pointId = info.object.id;
    try {
      const response = await fetch(`http://localhost:8002/neighbors?id=${pointId}&k=8`);
      if (!response.ok) throw new Error('Failed to fetch neighbors');
      
      const neighbors: Neighbor[] = await response.json();
      const neighborIds = new Set(neighbors.map(n => n.id));
      neighborIds.add(pointId); // Include the clicked point
      
      setHighlightedPoints(neighborIds);
      
      // Clear highlights after 3 seconds
      setTimeout(() => {
        setHighlightedPoints(new Set());
      }, 3000);
    } catch (err) {
      console.error('Error fetching neighbors:', err);
    }
  }, []);

  const handleTagToggle = (tag: string) => {
    const newSelected = new Set(selectedTags);
    if (newSelected.has(tag)) {
      newSelected.delete(tag);
    } else {
      newSelected.add(tag);
    }
    setSelectedTags(newSelected);
  };

  const handleAddTag = () => {
    if (newTagInput.trim() && !formData.tags.includes(newTagInput.trim())) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, newTagInput.trim()]
      }));
      setNewTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleSubmitStartup = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetch('http://localhost:8002/locate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) throw new Error('Failed to locate startup');
      
      const result = await response.json();
      const newPoint: Point = {
        id: result.id || Date.now(),
        name: formData.name,
        x: result.x,
        y: result.y,
        tags: formData.tags,
        one_liner: formData.one_liner,
        long_description: formData.long_description
      };
      
      // Start animation from screen center
      const viewState = {
        target: [0, 0, 0],
        zoom: 1
      };
      
      animX.set(0);
      animY.set(0);
      setAnimatingPoint(newPoint);
      
      // Animate to final position
      animX.set(result.x);
      animY.set(result.y);
      
      // Show temporary edges to neighbors if provided
      if (result.neighbors) {
        const edges = result.neighbors.map((neighborId: number) => {
          const neighbor = points.find(p => p.id === neighborId);
          return neighbor ? { from: newPoint, to: neighbor } : null;
        }).filter(Boolean);
        setTempEdges(edges);
      }
      
      // After animation, add to main points
      setTimeout(() => {
        setPoints(prev => [...prev, newPoint]);
        setAnimatingPoint(null);
        setTempEdges([]);
        setFormData({ name: '', one_liner: '', long_description: '', tags: [] });
        setShowForm(false);
      }, 2000);
      
    } catch (err) {
      console.error('Error submitting startup:', err);
      alert('Failed to add startup. Please try again.');
    }
  };

  // Enhanced layers with colors, labels, and connections
  const layers: any[] = [];

  // Tag connections layer (if enabled)
  if (showTagConnections && tagConnections.length > 0) {
    layers.push(
      new LineLayer({
        id: 'tag-connections',
        data: tagConnections,
        getSourcePosition: (d: any) => [d.source.x, d.source.y, -1],
        getTargetPosition: (d: any) => [d.target.x, d.target.y, -1],
        getColor: [100, 100, 100, 60], // Light gray, transparent
        getWidth: 1,
        pickable: false
      })
    );
  }

  // Main points layer - all points as colored circles
  layers.push(
    new ScatterplotLayer({
      id: 'main-points',
      data: displayPoints,
      getPosition: (d: Point) => [d.x, d.y, 0] as [number, number, number],
      getRadius: (d: Point) => {
        // Scale with zoom level - smaller when zoomed out, bigger when zoomed in
        const baseSize = Math.max(1.5, Math.min(6, zoom * 2));
        return highlightedPoints.has(d.id) ? baseSize * 1.3 : baseSize;
      },
      getFillColor: (d: Point) => {
        if (highlightedPoints.has(d.id)) {
          return [255, 165, 0, 255]; // Orange for highlighted
        }
        return getStatusColor(d);
      },
      pickable: true,
      onClick: handlePointClick as any,
      onHover: (info: any) => setHoveredPoint(info.object || null),
      radiusScale: 1,
      radiusMinPixels: 1, // Smaller minimum for zoomed out view
      radiusMaxPixels: 12,   // Larger maximum for zoomed in view
      stroked: highlightedPoints.size > 0,
      getLineColor: [255, 255, 255, 255],
      lineWidthMinPixels: 1,
      filled: true,
      updateTriggers: {
        getRadius: [zoom, highlightedPoints.size], // Include zoom dependency
        getFillColor: [highlightedPoints.size],
        stroked: [highlightedPoints.size]
      }
    })
  );

  // Skip TextLayer - it's not working in this setup. We'll use HTML overlay with proper coordinates.

  // Company names layer - only when data is loaded and zoomed in
  if (!loading && displayPoints.length > 0 && zoom > 0.5) {
    // Always show at least 20 labels, more when zoomed in
    const maxLabels = Math.max(20, Math.min(100, Math.floor(zoom * 50)));
    const limitedPoints = displayPoints.slice(0, maxLabels);
    
    console.log(`Rendering ${limitedPoints.length} labels at zoom ${zoom}`);
    
    layers.push(
      new TextLayer({
        id: 'startup-labels',
        data: limitedPoints,
        getPosition: (d: Point) => [d.x, d.y, 3] as [number, number, number],
        getText: (d: Point) => d.name || 'Unknown',
        getSize: Math.max(8, Math.min(24, zoom * 4)), // Scale with zoom
        getAngle: 0,
        getTextAnchor: 'middle' as any,
        getAlignmentBaseline: 'center' as any,
        getColor: [255, 255, 255, 255], // Full opacity white
        pickable: true,
        onClick: handlePointClick as any,
        onHover: (info: any) => setHoveredPoint(info.object || null),
        fontFamily: 'Arial, sans-serif',
        fontWeight: 'bold',
        outlineColor: [0, 0, 0, 255], // Full opacity black outline
        outlineWidth: 2,
        updateTriggers: {
          data: [displayPoints.length, zoom] // Force update when zoom changes
        }
      })
    );
  }

  // Cluster labels layer (show when enabled and zoomed out)
  if (showClusterLabels && clusterLabels.length > 0 && zoom < 2) {
    layers.push(
      new TextLayer({
        id: 'cluster-labels',
        data: clusterLabels,
        getPosition: (d: any) => [d.x, d.y, 2] as [number, number, number],
        getText: (d: any) => d.text,
        getSize: 16,
        getAngle: 0,
        getTextAnchor: 'middle' as any,
        getAlignmentBaseline: 'center' as any,
        getColor: [255, 255, 0, 180], // Yellow cluster labels
        pickable: false,
        fontFamily: 'Arial, sans-serif',
        fontWeight: 'bold',
        outlineColor: [0, 0, 0, 200],
        outlineWidth: 2
      })
    );
  }

  // Animated point
  if (animatingPoint) {
    layers.push(
      new ScatterplotLayer({
        id: 'animating-point',
        data: [{ ...animatingPoint, x: animX.get(), y: animY.get() }],
        getPosition: (d: any) => [d.x, d.y, 2] as [number, number, number],
        getRadius: 10,
        getFillColor: [255, 50, 50, 255],
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 2,
        stroked: true,
        filled: true,
        pickable: false
      })
    );
  }

  const initialViewState = {
    target: [0, 0, 0] as [number, number, number],
    zoom: 1, // Start further out since coordinates are 3x larger
    minZoom: 0,
    maxZoom: 12
  };

  // Debounced zoom handler for better performance
  const handleViewStateChange = useCallback(({viewState: newViewState}: any) => {
    // Round zoom to reduce update frequency
    const roundedZoom = Math.round(newViewState.zoom * 4) / 4; // Quarter-step rounding
    console.log(`Zoom: ${roundedZoom}, displayPoints: ${displayPoints.length}`);
    setZoom(roundedZoom);
    setViewState(newViewState);
    return newViewState;
    
    // More aggressive point reduction for better performance
    if (roundedZoom < 1) {
      setMaxPoints(600); // Even fewer points when zoomed out
    } else if (roundedZoom < 2) {
      setMaxPoints(1200);
    } else if (roundedZoom < 4) {
      setMaxPoints(2000);
    } else if (roundedZoom < 6) {
      setMaxPoints(3000);
    } else {
      setMaxPoints(4000); // More points when very zoomed in
    }
    return viewState;
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl">Loading startup map...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl text-red-600">Error: {error}</div>
        <button 
          onClick={fetchPoints}
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* Main Map */}
      <div className="flex-1 relative">
        <DeckGL
          views={new OrthographicView()}
          initialViewState={initialViewState}
          controller={{
            doubleClickZoom: false, // Disable for performance
            touchRotate: false // Disable for performance
          }}
          layers={layers}
          onViewStateChange={handleViewStateChange}
          // Enhanced tooltip with status info
          getTooltip={({ object }) => 
            object && zoom > 1 && {
              html: `
                <div class="bg-black text-white p-3 rounded shadow-lg max-w-sm text-xs">
                  <div class="flex items-center gap-2 mb-2">
                    ${object.logo_url ? `<img src="${object.logo_url}" alt="${object.name} logo" class="w-8 h-8 rounded object-cover" onerror="this.style.display='none'">` : ''}
                    <div>
                      <div class="font-bold text-sm">${object.name}</div>
                      <div class="text-gray-300">${object.one_liner || ''}</div>
                    </div>
                  </div>
                  ${object.long_description ? `<div class="text-gray-200 mb-2 text-xs leading-relaxed">${object.long_description.substring(0, 200)}${object.long_description.length > 200 ? '...' : ''}</div>` : ''}
                  <div class="flex gap-2 text-xs mb-2">
                    ${object.status ? `<span class="text-gray-400">Status:</span> <span class="text-white">${object.status}</span>` : ''}
                    ${object.batch ? `<span class="text-gray-400">Batch:</span> <span class="text-white">${object.batch}</span>` : ''}
                  </div>
                  <div class="mt-2">
                    ${object.tags?.slice(0, 4).map((tag: string) => 
                      `<span class="inline-block bg-blue-600 px-2 py-1 rounded mr-1 mt-1">${tag}</span>`
                    ).join('') || ''}
                  </div>
                </div>
              `
            }
          }
          // Enhanced performance optimizations
          useDevicePixels={false} // Better performance on high-DPI displays
          debug={false} // Disable debug mode for better performance
          getCursor={({isDragging}) => isDragging ? 'grabbing' : 'grab'}
          style={{ position: 'absolute', width: '100%', height: '100%', cursor: 'grab' }}
        />

        {/* HTML overlay for company names - positioned using viewport transformation */}
        {zoom > 1.5 && !loading && displayPoints.length > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            {displayPoints.slice(0, 30).map((point) => {
              // Simple coordinate transformation based on viewport
              // This is a rough approximation - adjust the scaling factors as needed
              const scale = Math.pow(2, viewState.zoom - 1);
              const screenX = (point.x - viewState.target[0]) * scale * 50 + window.innerWidth / 2;
              const screenY = (point.y - viewState.target[1]) * scale * 50 + window.innerHeight / 2;
              
              // Only show labels that are on screen
              if (screenX < -100 || screenX > window.innerWidth + 100 || 
                  screenY < -100 || screenY > window.innerHeight + 100) {
                return null;
              }
              
              return (
                <div
                  key={point.id}
                  className="absolute text-white font-bold pointer-events-none"
                  style={{
                    left: `${screenX}px`,
                    top: `${screenY}px`,
                    transform: 'translate(-50%, -50%)',
                    fontSize: `${Math.max(10, Math.min(16, zoom * 2))}px`,
                    textShadow: '2px 2px 4px rgba(0,0,0,0.8)',
                    zIndex: 1000
                  }}
                >
                  {point.name}
                </div>
              );
            })}
          </div>
        )}



        {/* Performance Info */}
        <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded text-sm z-10">
          Showing {displayPoints.length} of {points.length} startups
          {zoom < 3 && <div className="text-xs text-gray-300">Zoom in to see labels</div>}
        </div>

        {/* Visual Controls */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-75 text-white px-4 py-2 rounded text-sm z-10 flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showTagConnections}
              onChange={(e) => setShowTagConnections(e.target.checked)}
              className="rounded"
            />
            <span>Tag Connections</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              className="rounded"
            />
            <span>Labels</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showClusterLabels}
              onChange={(e) => setShowClusterLabels(e.target.checked)}
              className="rounded"
            />
            <span>Cluster Names</span>
          </label>
        </div>

        {/* Add Startup Button */}
        <button
          onClick={() => setShowForm(!showForm)}
          className="absolute top-4 right-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 shadow-lg z-10"
        >
          + Add My Startup
        </button>
      </div>

      {/* Sidebar */}
      <div className="w-80 bg-gray-50 p-4 overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">Filter Startups</h2>
        
        {/* Text Filter */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Search</label>
          <input
            type="text"
            value={textFilter}
            onChange={(e) => setTextFilter(e.target.value)}
            placeholder="Search names or tags..."
            className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Tag Filters */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Tags</label>
          <div className="max-h-64 overflow-y-auto">
            {allTags.map(tag => (
              <label key={tag} className="flex items-center mb-2">
                <input
                  type="checkbox"
                  checked={selectedTags.has(tag)}
                  onChange={() => handleTagToggle(tag)}
                  className="mr-2"
                />
                <span className="text-sm">{tag}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Color Legend */}
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-2">Status Colors</h3>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Active</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-400"></div>
              <span>Inactive</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500"></div>
              <span>Acquired</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span>Public</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Dead</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <span>Unknown</span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="text-sm text-gray-600">
          Showing {filteredPoints.length} of {points.length} startups
        </div>
      </div>

      {/* Add Startup Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg max-w-md w-full max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-bold mb-4">Add Your Startup</h3>
            
            <form onSubmit={handleSubmitStartup}>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Name *</label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">One-liner *</label>
                <input
                  type="text"
                  required
                  value={formData.one_liner}
                  onChange={(e) => setFormData(prev => ({ ...prev, one_liner: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Description *</label>
                <textarea
                  required
                  rows={3}
                  value={formData.long_description}
                  onChange={(e) => setFormData(prev => ({ ...prev, long_description: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Tags</label>
                <div className="flex mb-2">
                  <input
                    type="text"
                    value={newTagInput}
                    onChange={(e) => setNewTagInput(e.target.value)}
                    placeholder="Add a tag..."
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-l focus:ring-2 focus:ring-blue-500"
                    onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddTag())}
                  />
                  <button
                    type="button"
                    onClick={handleAddTag}
                    className="px-4 py-2 bg-blue-500 text-white rounded-r hover:bg-blue-600"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {formData.tags.map(tag => (
                    <span key={tag} className="inline-flex items-center bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                      {tag}
                      <button
                        type="button"
                        onClick={() => handleRemoveTag(tag)}
                        className="ml-1 text-blue-600 hover:text-blue-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Add Startup
                </button>
                <button
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}