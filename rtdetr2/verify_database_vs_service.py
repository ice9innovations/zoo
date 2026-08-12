#!/usr/bin/env python3
"""
Verify that database stored results match live service output
"""

import requests
import psycopg2
import json
import os
from dotenv import load_dotenv

def get_live_service_result(image_url):
    """Get result from live RT-DETR service"""
    try:
        response = requests.post(
            "http://localhost:8080/v3/analyze",
            json={"url": image_url},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling service: {e}")
        return None

def get_database_result(image_url, db_connection):
    """Get stored result from database"""
    cursor = db_connection.cursor()

    # Extract filename from URL for database lookup
    filename = image_url.split('/')[-1]

    cursor.execute("""
        SELECT response_data
        FROM consensus_results
        WHERE image_path LIKE %s
        AND service_name = 'rtdetr'
        ORDER BY created_at DESC
        LIMIT 1
    """, (f'%{filename}%',))

    result = cursor.fetchone()
    if result:
        return json.loads(result[0])
    return None

def compare_detections(live_result, db_result):
    """Compare live service result with database result"""
    if not live_result or not db_result:
        return False, "Missing results"

    # Extract bounding boxes from both results
    live_boxes = live_result.get('bounding_boxes', [])
    db_boxes = db_result.get('bounding_boxes', [])

    if len(live_boxes) != len(db_boxes):
        return False, f"Different number of detections: live={len(live_boxes)}, db={len(db_boxes)}"

    # Compare each detection
    for i, (live_box, db_box) in enumerate(zip(live_boxes, db_boxes)):
        # Compare coordinates (allow small floating point differences)
        for coord in ['x1', 'y1', 'x2', 'y2']:
            live_val = live_box.get(coord, 0)
            db_val = db_box.get(coord, 0)
            if abs(live_val - db_val) > 0.001:
                return False, f"Detection {i} coord {coord} differs: live={live_val}, db={db_val}"

        # Compare confidence
        live_conf = live_box.get('confidence', 0)
        db_conf = db_box.get('confidence', 0)
        if abs(live_conf - db_conf) > 0.001:
            return False, f"Detection {i} confidence differs: live={live_conf}, db={db_conf}"

        # Compare label
        if live_box.get('label') != db_box.get('label'):
            return False, f"Detection {i} label differs: live={live_box.get('label')}, db={db_box.get('label')}"

    return True, "Results match"

def main():
    # Load environment
    load_dotenv()

    # Test images (using JPEG format)
    test_images = [
        "http://k1.local/val2017/000000262487.jpg",  # Baseball image
        "http://k1.local/val2017/000000000139.jpg",
        "http://k1.local/val2017/000000000285.jpg",
    ]

    # Connect to database using same env vars as benchmark script
    db_connection = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    print("Comparing live service vs database results:")
    print("=" * 60)

    all_match = True

    for image_url in test_images:
        print(f"\nTesting: {image_url}")

        # Get results from both sources
        live_result = get_live_service_result(image_url)
        db_result = get_database_result(image_url, db_connection)

        # Compare results
        match, message = compare_detections(live_result, db_result)

        if match:
            print(f"✓ MATCH: {message}")
        else:
            print(f"✗ MISMATCH: {message}")
            all_match = False

            # Print details for debugging
            if live_result:
                print(f"  Live detections: {len(live_result.get('bounding_boxes', []))}")
            if db_result:
                print(f"  DB detections: {len(db_result.get('bounding_boxes', []))}")

    db_connection.close()

    print("\n" + "=" * 60)
    if all_match:
        print("✓ All results match - database accurately reflects service output")
    else:
        print("✗ Mismatches found - database may not accurately reflect current service")

if __name__ == "__main__":
    main()