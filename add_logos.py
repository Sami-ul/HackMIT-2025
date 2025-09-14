#!/usr/bin/env python3
import json
import random

# Read the current points data
with open('out_full/points.json', 'r') as f:
    points = json.load(f)

# Read the original YC data to get logo URLs
with open('ycombinator_startups.json', 'r', encoding='utf-8') as f:
    yc_data = json.load(f)

# Create a mapping of id to logo URL
logo_map = {}
for startup in yc_data:
    if 'id' in startup and 'small_logo_thumb_url' in startup:
        logo_map[startup['id']] = startup['small_logo_thumb_url']

# Add logo URLs to points
updated_points = []
for point in points:
    point_id = point.get('id')
    if point_id and point_id in logo_map:
        point['logo_url'] = logo_map[point_id]
    else:
        point['logo_url'] = ''
    updated_points.append(point)

# Write updated points back
with open('out_full/points.json', 'w') as f:
    json.dump(updated_points, f)

print(f"Updated {len(updated_points)} points with logo URLs")
print(f"Found {len([p for p in updated_points if p.get('logo_url')])} points with valid logo URLs")