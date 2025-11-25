#!/usr/bin/env python3
import sqlite3
import os

def check_database():
    # Connect to SQLite database
    db_path = 'attendance.db'
    if not os.path.exists(db_path):
        print("Database file not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check evidence table
    cursor.execute("SELECT id, file_path FROM evidence")
    evidences = cursor.fetchall()

    print("Evidence records in database:")
    for evidence_id, file_path in evidences:
        print(f'ID: {evidence_id}, Path: {repr(file_path)}')

    conn.close()

    # Check upload folder
    upload_folder = 'uploads'
    print(f"\nUpload folder: {upload_folder}")
    print("Files in upload folder:")
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            print(f"  {filename}")
    else:
        print("  Upload folder does not exist!")

if __name__ == '__main__':
    check_database()