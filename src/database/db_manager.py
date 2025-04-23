#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database management module for traffic surveillance.
"""

import os
import sqlite3
import logging
import json
import cv2
import numpy as np
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Class for managing the database for traffic surveillance.
    """
    
    def __init__(self, db_type="sqlite", db_path="data/output/surveillance.db"):
        """
        Initialize the database manager.
        
        Args:
            db_type (str): Type of database (sqlite, postgresql, vector_db)
            db_path (str): Path to the database file
        """
        self.db_type = db_type
        self.db_path = db_path
        self.conn = None
        self.vector_db = None
        
        logger.info(f"Initializing database manager with {db_type} database")
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._connect_db()
        self._create_tables()
    
    def _connect_db(self):
        """
        Connect to the database.
        """
        try:
            if self.db_type == "sqlite":
                self.conn = sqlite3.connect(self.db_path)
                logger.info(f"Connected to SQLite database at {self.db_path}")
            elif self.db_type == "postgresql":
                # Placeholder for PostgreSQL connection
                logger.error("PostgreSQL connection not implemented")
                raise NotImplementedError("PostgreSQL connection not implemented")
            elif self.db_type == "vector_db":
                # Connect to SQLite for regular data
                self.conn = sqlite3.connect(self.db_path)
                
                # Initialize vector database
                self._init_vector_db()
                
                logger.info(f"Connected to SQLite + Vector database at {self.db_path}")
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                raise ValueError(f"Unsupported database type: {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            # Fallback to in-memory SQLite database
            self.conn = sqlite3.connect(":memory:")
            logger.info("Falling back to in-memory SQLite database")
    
    def _init_vector_db(self):
        """
        Initialize the vector database.
        """
        try:
            import faiss
            
            # Create a simple index with 512 dimensions (typical for image embeddings)
            self.vector_db = faiss.IndexFlatL2(512)
            logger.info("Initialized FAISS vector database")
        
        except ImportError:
            logger.error("FAISS not available, vector database functionality disabled")
            self.vector_db = None
    
    def _create_tables(self):
        """
        Create database tables.
        """
        try:
            cursor = self.conn.cursor()
            
            # Create violations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                license_plate TEXT,
                confidence REAL,
                speed REAL,
                image BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create tracks table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                positions TEXT,
                license_plate TEXT,
                helmet_status TEXT,
                max_speed REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create vector embeddings table if using vector database
            if self.db_type == "vector_db" and self.vector_db is not None:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reference_id INTEGER NOT NULL,
                    reference_type TEXT NOT NULL,
                    embedding_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def store_violation(self, violation_type, timestamp, frame_number, track_id,
                        license_plate="", confidence=0.0, speed=0.0, image=None):
        """
        Store a violation in the database.
        
        Args:
            violation_type (str): Type of violation (speeding, no_helmet, etc.)
            timestamp (float): Timestamp of the violation
            frame_number (int): Frame number of the violation
            track_id (int): Track ID of the violator
            license_plate (str): License plate text (if available)
            confidence (float): Confidence of the license plate recognition
            speed (float): Speed of the vehicle (for speeding violations)
            image (numpy.ndarray): Image of the violation
            
        Returns:
            int: ID of the inserted violation
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert image to binary
            image_binary = None
            if image is not None:
                _, image_binary = cv2.imencode('.jpg', image)
                image_binary = image_binary.tobytes()
            
            # Insert violation
            cursor.execute('''
            INSERT INTO violations (
                violation_type, timestamp, frame_number, track_id,
                license_plate, confidence, speed, image
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                violation_type, timestamp, frame_number, track_id,
                license_plate, confidence, speed, image_binary
            ))
            
            self.conn.commit()
            violation_id = cursor.lastrowid
            
            # Store image embedding if vector database is available
            if self.db_type == "vector_db" and self.vector_db is not None and image is not None:
                self._store_embedding(image, violation_id, "violation")
            
            logger.info(f"Stored {violation_type} violation for track {track_id}")
            return violation_id
        
        except Exception as e:
            logger.error(f"Error storing violation: {e}")
            return None
    
    def store_track(self, track):
        """
        Store a track in the database.
        
        Args:
            track: Track object
            
        Returns:
            int: ID of the inserted track
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert positions to JSON
            positions_json = json.dumps(track.positions)
            
            # Get license plate text
            license_plate = ""
            if track.license_plate and "text" in track.license_plate:
                license_plate = track.license_plate["text"]
            
            # Insert track
            cursor.execute('''
            INSERT INTO tracks (
                track_id, class_name, first_seen, last_seen,
                positions, license_plate, helmet_status, max_speed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track.track_id, track.class_name, track.timestamps[0], track.timestamps[-1],
                positions_json, license_plate, track.helmet_status, track.speed
            ))
            
            self.conn.commit()
            track_id = cursor.lastrowid
            
            logger.info(f"Stored track {track.track_id} of class {track.class_name}")
            return track_id
        
        except Exception as e:
            logger.error(f"Error storing track: {e}")
            return None
    
    def _store_embedding(self, image, reference_id, reference_type):
        """
        Store an image embedding in the vector database.
        
        Args:
            image (numpy.ndarray): Image to embed
            reference_id (int): ID of the reference object
            reference_type (str): Type of reference (violation, track)
            
        Returns:
            int: ID of the embedding
        """
        try:
            # Generate embedding (placeholder - in a real system, use a proper embedding model)
            # For demonstration, we'll use a simple resizing and flattening
            resized = cv2.resize(image, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            embedding = gray.flatten().astype('float32')
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Pad or truncate to 512 dimensions
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            else:
                embedding = embedding[:512]
            
            # Add to FAISS index
            embedding = np.array([embedding], dtype='float32')
            embedding_id = self.vector_db.ntotal
            self.vector_db.add(embedding)
            
            # Store reference in SQLite
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO embeddings (reference_id, reference_type, embedding_id)
            VALUES (?, ?, ?)
            ''', (reference_id, reference_type, embedding_id))
            
            self.conn.commit()
            
            logger.info(f"Stored embedding for {reference_type} {reference_id}")
            return embedding_id
        
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return None
    
    def search_similar_images(self, image, limit=5):
        """
        Search for similar images in the vector database.
        
        Args:
            image (numpy.ndarray): Query image
            limit (int): Maximum number of results
            
        Returns:
            list: List of similar image references
        """
        if self.db_type != "vector_db" or self.vector_db is None:
            logger.error("Vector database not available")
            return []
        
        try:
            # Generate embedding for the query image
            resized = cv2.resize(image, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            embedding = gray.flatten().astype('float32')
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Pad or truncate to 512 dimensions
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            else:
                embedding = embedding[:512]
            
            # Search in FAISS index
            embedding = np.array([embedding], dtype='float32')
            distances, indices = self.vector_db.search(embedding, limit)
            
            # Get references from SQLite
            cursor = self.conn.cursor()
            results = []
            
            for idx in indices[0]:
                if idx != -1:  # Valid index
                    cursor.execute('''
                    SELECT reference_id, reference_type FROM embeddings
                    WHERE embedding_id = ?
                    ''', (int(idx),))
                    
                    row = cursor.fetchone()
                    if row:
                        reference_id, reference_type = row
                        
                        # Get additional information based on reference type
                        if reference_type == "violation":
                            cursor.execute('''
                            SELECT violation_type, timestamp, license_plate, speed
                            FROM violations WHERE id = ?
                            ''', (reference_id,))
                            
                            violation_info = cursor.fetchone()
                            if violation_info:
                                results.append({
                                    "id": reference_id,
                                    "type": reference_type,
                                    "violation_type": violation_info[0],
                                    "timestamp": violation_info[1],
                                    "license_plate": violation_info[2],
                                    "speed": violation_info[3]
                                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching similar images: {e}")
            return []
    
    def get_violations(self, limit=100, offset=0, violation_type=None):
        """
        Get violations from the database.
        
        Args:
            limit (int): Maximum number of results
            offset (int): Offset for pagination
            violation_type (str): Type of violation to filter by
            
        Returns:
            list: List of violations
        """
        try:
            cursor = self.conn.cursor()
            
            query = '''
            SELECT id, violation_type, timestamp, frame_number, track_id,
                   license_plate, confidence, speed, image
            FROM violations
            '''
            
            params = []
            
            if violation_type:
                query += " WHERE violation_type = ?"
                params.append(violation_type)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            violations = []
            for row in cursor.fetchall():
                violation = {
                    "id": row[0],
                    "violation_type": row[1],
                    "timestamp": row[2],
                    "frame_number": row[3],
                    "track_id": row[4],
                    "license_plate": row[5],
                    "confidence": row[6],
                    "speed": row[7]
                }
                
                # Convert binary image to base64 for web display
                if row[8]:
                    image_base64 = base64.b64encode(row[8]).decode('utf-8')
                    violation["image"] = f"data:image/jpeg;base64,{image_base64}"
                
                violations.append(violation)
            
            return violations
        
        except Exception as e:
            logger.error(f"Error getting violations: {e}")
            return []
    
    def get_violation_stats(self):
        """
        Get violation statistics.
        
        Returns:
            dict: Violation statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Get total violations by type
            cursor.execute('''
            SELECT violation_type, COUNT(*) FROM violations
            GROUP BY violation_type
            ''')
            
            violation_counts = {}
            for row in cursor.fetchall():
                violation_counts[row[0]] = row[1]
            
            # Get violations by hour
            cursor.execute('''
            SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                   COUNT(*) FROM violations
            GROUP BY hour
            ORDER BY hour
            ''')
            
            violations_by_hour = {}
            for row in cursor.fetchall():
                violations_by_hour[row[0]] = row[1]
            
            return {
                "total": sum(violation_counts.values()),
                "by_type": violation_counts,
                "by_hour": violations_by_hour
            }
        
        except Exception as e:
            logger.error(f"Error getting violation stats: {e}")
            return {"total": 0, "by_type": {}, "by_hour": {}}
    
    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
