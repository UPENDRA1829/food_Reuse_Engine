import sqlite3

db_path = r"D:\Food_Redistribute_Project\backend\food_waste.db"

connection = sqlite3.connect(db_path)
cursor = connection.cursor()

print("TABLES:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for row in cursor.fetchall():
    print(row[0])

for table in ["sales", "inventory"]:
    print(f"\n{table.upper()} COLUMNS:")
    cursor.execute(f"PRAGMA table_info({table})")
    for column in cursor.fetchall():
        print(column)

connection.close()
print("\nRead-only inspection completed.")
