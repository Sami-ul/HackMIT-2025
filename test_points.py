#!/usr/bin/env python3
# test_points.py - Simple test script to verify points loading

import json
import os

def main():
    points_path = os.path.join("out_full", "points.json")
    print(f"Loading from: {points_path}")
    
    with open(points_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} points")
    print(f"First point: {data[0] if data else 'None'}")
    print(f"Sample point keys: {list(data[0].keys()) if data else 'None'}")
    
    # Check if logo_url is present
    has_logo = sum(1 for p in data if p.get('logo_url'))
    print(f"Points with logos: {has_logo}")

if __name__ == "__main__":
    main()