#!/usr/bin/env python3
# inspect_database.py

import os
import sys
import argparse
import sqlite3

def dict_factory(cursor, row):
    """Convert SQLite row to dictionary for easier access."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def connect_db(db_path):
    """Connect to the SQLite database."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory
    return conn

def inspect_database(run_dir):
    """Debug function to inspect the products in the database."""
    db_path = os.path.join(run_dir, "analysis_registry.db")
    db_conn = connect_db(db_path)
    
    if not db_conn:
        return
    
    # Check products table
    cursor = db_conn.cursor()
    
    print("\n----- Analysis Modules -----")
    cursor.execute("SELECT module_id, module_name, status FROM analysis_modules")
    modules = cursor.fetchall()
    for module in modules:
        print(f"ID: {module['module_id']}, Name: {module['module_name']}, Status: {module['status']}")
    
    print("\n----- Plot Products -----")
    cursor.execute("""
        SELECT p.product_id, m.module_name, p.product_type, p.category, p.subcategory, p.relative_path
        FROM analysis_products p
        JOIN analysis_modules m ON p.module_id = m.module_id
        WHERE p.product_type = 'png'
    """)
    products = cursor.fetchall()
    for product in products:
        print(f"Module: {product['module_name']}, " 
              f"Category: {product['category']}, " 
              f"Subcategory: {product['subcategory']}, " 
              f"Path: {product['relative_path']}")
    
    print("\n----- HTML Report Plot Queries -----")
    plot_queries = [
        # G-G distance plots
        ("g_g_distances", "png", "plot", "filtered_distances", "core_analysis_visualization_g_g"),
        ("g_g_distances", "png", "plot", "filtered_distances", "core_analysis_filtering"),
        # COM plots
        ("com_distances", "png", "plot", "filtered_distances", "core_analysis_visualization_com"),
        ("com_distances", "png", "plot", "filtered_distances", "core_analysis_filtering"),
        # KDE analysis
        ("com_kde", "png", "plot", "kde_analysis", "core_analysis_visualization_com"),
    ]
    
    for plot_key, product_type, category, subcategory, module_name in plot_queries:
        print(f"\nTrying to find: key={plot_key}, type={product_type}, category={category}, "
              f"subcategory={subcategory}, module={module_name}")
        
        cursor.execute("""
            SELECT p.relative_path 
            FROM analysis_products p
            JOIN analysis_modules m ON p.module_id = m.module_id
            WHERE p.product_type = ? AND p.category = ? AND p.subcategory = ? AND m.module_name = ?
        """, (product_type, category, subcategory, module_name))
        
        result = cursor.fetchone()
        if result:
            print(f"FOUND: {result['relative_path']}")
        else:
            print("NOT FOUND")
    
    # Additional query to check using different combinations
    print("\n----- Looser Plot Queries (by path pattern) -----")
    plot_patterns = [
        ("G_G_Distance_Plot.png", "G-G filtered plot"),
        ("COM_Distance_Plot.png", "COM filtered plot"),
        ("COM_Stability_KDE_Analysis.png", "COM KDE plot")
    ]
    
    for pattern, description in plot_patterns:
        cursor.execute("""
            SELECT p.product_id, m.module_name, p.category, p.subcategory, p.relative_path
            FROM analysis_products p
            JOIN analysis_modules m ON p.module_id = m.module_id
            WHERE p.relative_path LIKE ?
        """, (f"%{pattern}%",))
        
        results = cursor.fetchall()
        print(f"\nSearching for {description} with pattern '{pattern}':")
        if results:
            for result in results:
                print(f"FOUND: Module={result['module_name']}, "
                      f"Category={result['category']}, "
                      f"Subcategory={result['subcategory']}, "
                      f"Path={result['relative_path']}")
        else:
            print("NOT FOUND")
    
    # Show metrics for reference
    print("\n----- Metrics -----")
    cursor.execute("""
        SELECT m.module_name, met.metric_name, met.value, met.units
        FROM metrics met
        JOIN analysis_modules m ON met.module_id = m.module_id
        ORDER BY m.module_name, met.metric_name
    """)
    metrics = cursor.fetchall()
    for metric in metrics:
        print(f"Module: {metric['module_name']}, "
              f"Metric: {metric['metric_name']}, "
              f"Value: {metric['value']}, "
              f"Units: {metric['units']}")
    
    db_conn.close()

def main():
    parser = argparse.ArgumentParser(description='Inspect pore analysis database')
    parser.add_argument('run_dir', help='Path to the run directory containing analysis_registry.db')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    db_path = os.path.join(args.run_dir, "analysis_registry.db")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return 1
    
    print(f"Inspecting database at: {db_path}")
    inspect_database(args.run_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
