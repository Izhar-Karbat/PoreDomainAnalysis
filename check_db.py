import sqlite3

# Connect to the database
conn = sqlite3.connect('/home/labs/bmeitan/karbati/rCs1/enhanced_report_results/enhanced_cross_analysis.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check systems and their groups
cursor.execute('SELECT run_name, group_type FROM systems')
print('Systems by group:')
toxin_bound = []
toxin_free = []
unknown = []

for row in cursor.fetchall():
    print(f"  {row['run_name']}: {row['group_type']}")
    if row['group_type'] == 'toxin-bound':
        toxin_bound.append(row['run_name'])
    elif row['group_type'] == 'toxin-free':
        toxin_free.append(row['run_name'])
    else:
        unknown.append(row['run_name'])

print(f"\nToxin-bound systems ({len(toxin_bound)}): {', '.join(toxin_bound)}")
print(f"Toxin-free systems ({len(toxin_free)}): {', '.join(toxin_free)}")
print(f"Unknown systems ({len(unknown)}): {', '.join(unknown)}")

# Count metrics
cursor.execute('SELECT COUNT(*) FROM aggregated_metrics')
print(f"\nTotal metrics: {cursor.fetchone()[0]}")

# Check metrics by group
cursor.execute('''
    SELECT s.group_type, COUNT(m.metric_id) as metric_count
    FROM aggregated_metrics m
    JOIN systems s ON m.system_id = s.system_id
    GROUP BY s.group_type
''')
print("\nMetrics by group:")
for row in cursor.fetchall():
    print(f"  {row['group_type']}: {row['metric_count']} metrics")

# Close connection
conn.close()