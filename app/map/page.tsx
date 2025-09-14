'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, LineLayer, TextLayer, IconLayer } from '@deck.gl/layers';
import { OrthographicView, COORDINATE_SYSTEM } from '@deck.gl/core';


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
  team_size?: number;
  cluster_id?: number;
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
  
  // Competitor analysis state
  const [selectedStartup, setSelectedStartup] = useState<Point | null>(null);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState('');
  const [analysisResults, setAnalysisResults] = useState<string | null>(null);
  
  // Performance optimizations
  const [zoom, setZoom] = useState(1);
  const [controlledViewState, setControlledViewState] = useState<any>(null);
  
  // Visual enhancements
  const [showTagConnections, setShowTagConnections] = useState(false);
  const [colorMode, setColorMode] = useState<'stage' | 'cluster' | 'density'>('stage');  
  const [clusterSummary, setClusterSummary] = useState<any>(null);
  
  // Form state
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState<StartupForm>({
    name: '',
    one_liner: '',
    long_description: '',
    tags: []
  });
  const [newTagInput, setNewTagInput] = useState('');
  const [isSubmittingStartup, setIsSubmittingStartup] = useState(false);

  // Color palettes
  const clusterPalette: [number, number, number, number][] = [
    [59, 130, 246, 255],   // Blue
    [34, 197, 94, 255],    // Green  
    [168, 85, 247, 255],   // Purple
    [239, 68, 68, 255],    // Red
    [251, 191, 36, 255],   // Yellow
    [14, 165, 233, 255],   // Sky blue
    [139, 69, 19, 255],    // Brown
    [255, 20, 147, 255],   // Deep pink
    [0, 191, 255, 255],    // Deep sky blue
    [50, 205, 50, 255],    // Lime green
    [255, 140, 0, 255],    // Dark orange
    [147, 112, 219, 255],  // Medium purple
    [220, 20, 60, 255],    // Crimson
    [0, 206, 209, 255],    // Dark turquoise
    [255, 215, 0, 255],    // Gold
    [128, 0, 128, 255],    // Purple
    [255, 69, 0, 255],     // Orange red
    [46, 139, 87, 255],    // Sea green
    [219, 112, 147, 255],  // Pale violet red
    [30, 144, 255, 255]    // Dodger blue
  ];

  // Color schemes based on status
  const getStatusColor = (point: Point): [number, number, number, number] => {
    const status = point.status?.toLowerCase() || 'unknown';
    switch (status) {
      case 'new':
        return [255, 255, 255, 255]; // White fill for new startups
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

  const getClusterColor = (point: Point): [number, number, number, number] => {
    if (point.cluster_id === undefined || point.cluster_id === -1) {
      return [156, 163, 175, 200]; // Gray for noise/unclustered
    }
    return clusterPalette[point.cluster_id % clusterPalette.length];
  };

  const getDensityColor = (point: Point): [number, number, number, number] => {
    // Simple density based on local neighborhood - this could be enhanced
    const localDensity = filteredPoints.filter(p => 
      Math.sqrt((p.x - point.x) ** 2 + (p.y - point.y) ** 2) < 5
    ).length;
    
    const intensity = Math.min(255, localDensity * 30);
    return [intensity, 255 - intensity, 100, 255];
  };

  const getPointColor = (point: Point): [number, number, number, number] => {
    switch (colorMode) {
      case 'cluster':
        return getClusterColor(point);
      case 'density': 
        return getDensityColor(point);
      case 'stage':
      default:
        return getStatusColor(point);
    }
  };



  // Fetch points and cluster summary on mount
  useEffect(() => {
    fetchPoints();
    fetchClusterSummary();
  }, []);

  const fetchPoints = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8003/points');
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

  const fetchClusterSummary = async () => {
    try {
      const response = await fetch('http://localhost:8003/clusters/summary');
      if (response.ok) {
        const summary = await response.json();
        setClusterSummary(summary);
      }
    } catch (err) {
      console.warn('Failed to fetch cluster summary:', err);
      setClusterSummary(null);
    }
  };

  // Optimized filtering with performance limits
  const displayPoints = useMemo(() => {
    let filtered = points;
    
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
    
    // Return all filtered points - no artificial limits
    return filtered;
  }, [points, textFilter, selectedTags, zoom]);
  
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



  const handlePointClick = useCallback(async (info: any) => {
    if (!info.object) return;
    
    const point = info.object as Point;
    const pointId = point.id;
    
    // If it's a new startup (status 'new'), show the analysis option
    if (point.status === 'new') {
      setSelectedStartup(point);
      setShowAnalysisModal(true);
      return;
    }
    
    try {
      const response = await fetch(`http://localhost:8003/neighbors?id=${pointId}&k=8`);
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

  const handleCompetitorAnalysis = async () => {
    if (!selectedStartup) return;
    
    setAnalysisLoading(true);
    setAnalysisProgress('Starting competitor analysis...');
    
    try {
      const response = await fetch('http://localhost:8003/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          startup_id: selectedStartup.id,
          startup_data: {
            name: selectedStartup.name,
            one_liner: selectedStartup.one_liner,
            long_description: selectedStartup.long_description,
            tags: selectedStartup.tags
          }
        }),
      });
      
      if (!response.ok) throw new Error('Failed to start analysis');
      
      // Use Server-Sent Events to track progress
      const eventSource = new EventSource(`http://localhost:8003/analysis-progress/${selectedStartup.id}`);
      
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setAnalysisProgress(data.progress);
        
        if (data.completed) {
          setAnalysisResults(data.results);
          setAnalysisLoading(false);
          eventSource.close();
        }
      };
      
      eventSource.onerror = () => {
        setAnalysisProgress('Error during analysis');
        setAnalysisLoading(false);
        eventSource.close();
      };
      
    } catch (err) {
      console.error('Error starting analysis:', err);
      setAnalysisProgress('Failed to start analysis');
      setAnalysisLoading(false);
    }
  };

  const handleSubmitStartup = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmittingStartup(true);
    
    try {
      const response = await fetch('http://localhost:8003/locate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) throw new Error('Failed to locate startup');
      
      const result = await response.json();
      
      // Scale coordinates to match the display (2x scaling like in fetchPoints)
      const scaledX = result.x * 2;
      const scaledY = result.y * 2;
      
      // Validate coordinates to prevent rendering issues
      if (!isFinite(scaledX) || !isFinite(scaledY)) {
        throw new Error('Invalid coordinates returned from server');
      }
      
      const newPoint: Point = {
        id: Date.now(),
        name: formData.name,
        x: scaledX,
        y: scaledY,
        tags: formData.tags,
        one_liner: formData.one_liner,
        long_description: formData.long_description,
        status: 'new' // Special status for newly added startups
      };
      
      console.log('Adding new point:', newPoint); // Debug log
      
      // Add the new point immediately to the points array
      setPoints(prev => [...prev, newPoint]);
      
      // Highlight the new point immediately
      setHighlightedPoints(new Set([newPoint.id]));
      
      // Zoom to fixed level 8.5 to show the new startup clearly
      setTimeout(() => {
        setControlledViewState({
          target: [scaledX, scaledY, 0] as [number, number, number],
          zoom: 8.5, // Fixed zoom level
          minZoom: 0,
          maxZoom: 12
        });
      }, 300);
      
      // Clear form and close modal
      setFormData({ name: '', one_liner: '', long_description: '', tags: [] });
      setShowForm(false);
      
      // Clear highlight after a few seconds
      setTimeout(() => {
        setHighlightedPoints(new Set());
        // Clear controlled view state to return control to user
        setControlledViewState(null);
      }, 5000);
      
    } catch (err) {
      console.error('Error submitting startup:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to add startup. Please try again.';
      alert(errorMessage);
    } finally {
      setIsSubmittingStartup(false);
      
      // Reset controlled view state on error
      setControlledViewState(null);
    }
  };

  // Enhanced layers with colors, labels, and connections (memoized for performance)
  const layers = useMemo(() => {
    const layersArray: any[] = [];

    // Tag connections layer (if enabled)
    if (showTagConnections && tagConnections.length > 0) {
      layersArray.push(
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
    layersArray.push(
    new ScatterplotLayer({
      id: `main-points-${colorMode}`,
      data: displayPoints,
      getPosition: (d: Point) => [d.x, d.y, 0],

      // --- KEY: size in pixels but scale slightly with zoom for visibility ---
      radiusUnits: 'pixels',
      getRadius: (d: Point) => {
        // Base size in pixels that scales gently with zoom level
        const baseSize = Math.max(3, Math.min(8, 2 + zoom * 0.8)); // Scale from 3px to 8px based on zoom
        return highlightedPoints.has(d.id) ? baseSize * 1.2 : baseSize;
      },
      lineWidthUnits: 'pixels',
      getLineWidth: (d: Point) => {
        // Scale stroke width slightly with zoom for visibility
        const baseStroke = Math.max(1, Math.min(2.5, 0.5 + zoom * 0.2));
        if (d.status === 'new') return baseStroke;                 // outline for new
        if (highlightedPoints.has(d.id)) return Math.max(1, baseStroke * 0.8);           // thinner outline when highlighted
        return 0;                                            // no outline otherwise
      },
      lineWidthMinPixels: 1, // ensure visible when very thin

      // Fill & stroke colors
      getFillColor: (d: Point) => {
        // Priority: highlighted > color mode (which handles 'new' status internally)
        if (highlightedPoints.has(d.id)) return [255, 165, 0, 255]; // orange when highlighted
        return getPointColor(d);
      },
      getLineColor: (d: Point) => {
        if (d.status === 'new') return [255, 255, 0, 255];   // yellow ring for new
        if (highlightedPoints.has(d.id)) return [255, 255, 255, 255];
        return [0, 0, 0, 60]; // subtle dark outline
      },

      // Interactions
      pickable: true,
      onClick: handlePointClick as any,
      onHover: (info: any) => setHoveredPoint(info.object || null),

      stroked: true,
      filled: true,

      updateTriggers: {
        getRadius: [zoom, highlightedPoints.size, points.length],
        getFillColor: [highlightedPoints.size, points.length, colorMode, clusterSummary],
        getLineColor: [highlightedPoints.size, points.length, colorMode],
        getLineWidth: [zoom, highlightedPoints.size, points.length]
      }
    })
  );

    return layersArray;
  }, [displayPoints, highlightedPoints, hoveredPoint, zoom, showTagConnections, tagConnections, colorMode, clusterSummary]);

  // Initial view state
  const initialViewState = {
    target: [0, 0, 0] as [number, number, number],
    zoom: 1,
    minZoom: 0,
    maxZoom: 12
  };

  // Debounced zoom handler for better performance
  const handleViewStateChange = useCallback(({viewState: newViewState}: any) => {
    // Use setTimeout to avoid setState during render
    setTimeout(() => {
      // Round zoom to reduce update frequency
      const roundedZoom = Math.round(newViewState.zoom * 4) / 4; // Quarter-step rounding
      setZoom(roundedZoom);
      
      // Clear controlled view state once user starts interacting
      if (controlledViewState) {
        setControlledViewState(null);
      }
    }, 0);
    
    return newViewState;
  }, [controlledViewState]);

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
          viewState={controlledViewState || undefined}
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
                    <div>
                      <div class="font-bold text-sm">${object.name}</div>
                      <div class="text-gray-300">${object.one_liner || ''}</div>
                    </div>
                  </div>
                  ${object.long_description ? `<div class="text-gray-200 mb-2 text-xs leading-relaxed">${object.long_description.substring(0, 200)}${object.long_description.length > 200 ? '...' : ''}</div>` : ''}
                  <div class="flex gap-2 text-xs mb-2">
                    ${object.status ? `<span class="text-gray-400">Status:</span> <span class="text-white">${object.status}</span>` : ''}
                    ${object.batch ? `<span class="text-gray-400">Batch:</span> <span class="text-white">${object.batch}</span>` : ''}
                    ${object.cluster_id !== undefined && object.cluster_id !== -1 ? `<span class="text-gray-400">Cluster:</span> <span class="text-white">C${object.cluster_id}</span>` : ''}
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





        {/* Performance Info */}
        <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded text-sm z-10">
          Showing {displayPoints.length} of {points.length} startups
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

        {/* Color Mode Toggle */}
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-3">Color Mode</h3>
          <div className="flex gap-2 mb-4">
            {(['stage', 'cluster', 'density'] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setColorMode(mode)}
                className={`px-3 py-1 text-xs rounded-full transition-colors ${
                  colorMode === mode
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>

          {/* Dynamic Legend */}
          {colorMode === 'stage' && (
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-white border-2 border-yellow-400"></div>
                <span>New</span>
              </div>
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
            </div>
          )}

          {colorMode === 'cluster' && clusterSummary && (
            <div className="space-y-1 text-xs">
              <div className="text-gray-600 mb-2">
                {clusterSummary.num_clusters} clusters • {Math.round(clusterSummary.clustering_rate * 100)}% clustered
              </div>
              {clusterPalette.slice(0, Math.min(10, clusterSummary.num_clusters)).map((color, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`}}
                  ></div>
                  <span>C{i}</span>
                </div>
              ))}
              {clusterSummary.noise_count > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                  <span>Noise ({clusterSummary.noise_count})</span>
                </div>
              )}
            </div>
          )}

          {colorMode === 'cluster' && !clusterSummary && (
            <div className="text-xs text-gray-500">
              No clustering data. Run <code>python cluster_embeddings.py</code> first.
            </div>
          )}

          {colorMode === 'density' && (
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span>High density</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <span>Medium density</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span>Low density</span>
              </div>
            </div>
          )}
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
                  disabled={isSubmittingStartup}
                  className={`flex-1 px-4 py-2 text-white rounded flex items-center justify-center ${
                    isSubmittingStartup 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-green-500 hover:bg-green-600'
                  }`}
                >
                  {isSubmittingStartup ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Adding...
                    </>
                  ) : (
                    'Add Startup'
                  )}
                </button>
                <button
                  type="button"
                  disabled={isSubmittingStartup}
                  onClick={() => setShowForm(false)}
                  className={`px-4 py-2 text-white rounded ${
                    isSubmittingStartup 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-gray-500 hover:bg-gray-600'
                  }`}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Competitor Analysis Modal */}
      {showAnalysisModal && selectedStartup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold">Competitor Analysis for {selectedStartup.name}</h3>
              <button
                onClick={() => {
                  setShowAnalysisModal(false);
                  setSelectedStartup(null);
                  setAnalysisResults(null);
                  setAnalysisProgress('');
                }}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            
            {!analysisResults && !analysisLoading && (
              <div className="text-center py-8">
                <div className="mb-4">
                  <h4 className="text-lg font-semibold mb-2">Ready to analyze your startup!</h4>
                  <p className="text-gray-600 mb-4">
                    We'll research your top 5 competitors, analyze their strategies, 
                    and provide insights on how to succeed in your market.
                  </p>
                </div>
                <button
                  onClick={handleCompetitorAnalysis}
                  className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-semibold"
                >
                  Start Deep Analysis
                </button>
              </div>
            )}
            
            {analysisLoading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p className="text-gray-600">{analysisProgress}</p>
              </div>
            )}
            
            {analysisResults && (
              <div className="analysis-results max-h-96 overflow-y-auto">
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg mb-4">
                  <h4 className="text-lg font-semibold text-gray-800 mb-2">🎯 Strategic Analysis Complete</h4>
                  <p className="text-sm text-gray-600">Based on analysis of your top 5 competitors and market research</p>
                </div>
                <div 
                  className="analysis-content"
                  style={{
                    fontSize: '15px',
                    lineHeight: '1.7',
                    color: '#374151'
                  }}
                  dangerouslySetInnerHTML={{ __html: analysisResults }}
                />
                <div className="mt-6 p-4 bg-green-50 rounded-lg border-l-4 border-green-400">
                  <p className="text-sm text-green-700">
                    💡 <strong>Next Steps:</strong> Use this analysis to refine your strategy and avoid common pitfalls. 
                    Focus on the 90-day action plan to get started immediately.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}