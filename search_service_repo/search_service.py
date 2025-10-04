#!/usr/bin/env python3
"""
CCTV Photo Search Service

This service provides intelligent search capabilities for CCTV photos
with face recognition and RAG (Retrieval-Augmented Generation).
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import base64
import json

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure

from .config import get_config
from .embeddings import VisualEmbeddingEngine, FaceDatabaseManager
from .rag_search import RAGSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="CCTV Photo Search API",
    description="Search API for CCTV photos with face recognition and RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components
visual_engine = None
rag_engine = None
face_manager = None
face_collection = None

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    time_range: Optional[Dict[str, float]] = None
    camera_location: Optional[str] = None

class AddFaceRequest(BaseModel):
    name: str
    image_data: str  # Base64 encoded image

class PhotoResult(BaseModel):
    filename: str
    timestamp: float
    camera_location: str
    relevance_score: float
    explanation: str
    faces_detected: Dict[str, Any]
    s3_url: str
    bson_time: str
    face_count: int

class SearchResponse(BaseModel):
    results: List[PhotoResult]
    total_found: int
    query: str

class StatsResponse(BaseModel):
    total_photos: int
    recent_photos_24h: int
    photos_with_faces: int
    known_faces_detected: int
    unknown_faces_detected: int
    known_faces_in_db: int

def initialize_services():
    """Initialize all services and load known faces."""
    global visual_engine, rag_engine, face_manager, face_collection
    
    try:
        # Initialize components
        visual_engine = VisualEmbeddingEngine()
        rag_engine = RAGSearchEngine(visual_engine)
        face_manager = FaceDatabaseManager(visual_engine)
        
        # Connect to MongoDB
        client = MongoClient(config["mongo_host"], config["mongo_port"])
        db = client[config["mongo_db"]]
        face_collection = db[config["face_collection"]]
        
        # Load known faces from database
        load_known_faces()
        
        logger.info("Search service initialized successfully")
        
    except Exception as error:
        logger.error(f"Failed to initialize search service: {error}")
        raise

def load_known_faces():
    """Load known faces from the faces collection."""
    try:
        # Look for photos with identified faces
        known_faces_docs = face_collection.find({
            "embeddings.face_0_person": {"$exists": True}
        }).limit(100)
        
        loaded_count = 0
        for doc in known_faces_docs:
            embeddings = doc.get('embeddings', {})
            for key, value in embeddings.items():
                if key.endswith('_person'):
                    face_id = key.replace('_person', '')
                    person_name = value.decode('utf-8') if isinstance(value, bytes) else value
                    if person_name != 'unknown' and face_id in embeddings:
                        face_manager.known_faces[person_name] = embeddings[face_id]
                        loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} known faces from database")
        
    except Exception as error:
        logger.error(f"Failed to load known faces: {error}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    initialize_services()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CCTV Photo Search API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "search": "/search",
            "add_face": "/add_face",
            "stats": "/stats",
            "family_photos": "/family_photos",
            "stranger_photos": "/stranger_photos"
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_photos(request: SearchRequest):
    """Search photos using natural language queries."""
    try:
        # Get photos from faces collection
        photos = list(face_collection.find().sort("date", -1).limit(config["search_limit"]))
        
        # Prepare filters
        filters = {}
        if request.time_range:
            filters["time_range"] = request.time_range
        if request.camera_location:
            filters["camera_location"] = request.camera_location
        
        # Perform RAG search
        results = rag_engine.search_photos(
            query=request.query,
            photos_data=photos,
            top_k=request.limit,
            filters=filters
        )
        
        # Format results
        photo_results = []
        for result in results:
            photo = result['photo']
            photo_results.append(PhotoResult(
                filename=photo.get('filename', ''),
                timestamp=photo.get('date', 0),
                camera_location=photo.get('camera_location', ''),
                relevance_score=result['relevance_score'],
                explanation=result['search_explanation'],
                faces_detected=result['photo'].get('faces_detected', {}),
                s3_url=photo.get('s3_file_url', ''),
                bson_time=photo.get('bsonTime', ''),
                face_count=photo.get('face_count', 0)
            ))
        
        return SearchResponse(
            results=photo_results,
            total_found=len(photo_results),
            query=request.query
        )
        
    except Exception as error:
        logger.error(f"Error in photo search: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/add_face")
async def add_known_face(request: AddFaceRequest):
    """Add a person to the known faces database."""
    try:
        # Decode base64 image data
        image_data = base64.b64decode(request.image_data)
        
        # Add to face database
        success = face_manager.add_person(request.name, image_data)
        
        if success:
            return {"message": f"Successfully added {request.name} to known faces database"}
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to add {request.name}. Make sure the image contains a clear face."
            )
            
    except Exception as error:
        logger.error(f"Error adding face: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/stats", response_model=StatsResponse)
async def get_photo_stats():
    """Get statistics about the photo database."""
    try:
        total_photos = face_collection.count_documents({})
        
        # Get recent photos (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_photos = face_collection.count_documents({
            "date": {"$gte": yesterday.timestamp()}
        })
        
        # Count photos with faces
        photos_with_faces = face_collection.count_documents({
            "has_faces": True
        })
        
        # Count known vs unknown faces
        known_faces_count = face_collection.count_documents({
            "embeddings.face_0_person": {"$ne": b"unknown"}
        })
        
        return StatsResponse(
            total_photos=total_photos,
            recent_photos_24h=recent_photos,
            photos_with_faces=photos_with_faces,
            known_faces_detected=known_faces_count,
            unknown_faces_detected=photos_with_faces - known_faces_count,
            known_faces_in_db=len(face_manager.known_faces)
        )
        
    except Exception as error:
        logger.error(f"Error getting stats: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/family_photos", response_model=SearchResponse)
async def find_family_photos(limit: int = Query(10, ge=1, le=50)):
    """Find photos containing family members."""
    try:
        photos = list(face_collection.find().sort("date", -1).limit(config["search_limit"]))
        
        results = rag_engine.search_photos(
            query="photos with me and my son",
            photos_data=photos,
            top_k=limit
        )
        
        photo_results = []
        for result in results:
            photo = result['photo']
            photo_results.append(PhotoResult(
                filename=photo.get('filename', ''),
                timestamp=photo.get('date', 0),
                camera_location=photo.get('camera_location', ''),
                relevance_score=result['relevance_score'],
                explanation=result['search_explanation'],
                faces_detected=result['photo'].get('faces_detected', {}),
                s3_url=photo.get('s3_file_url', ''),
                bson_time=photo.get('bsonTime', ''),
                face_count=photo.get('face_count', 0)
            ))
        
        return SearchResponse(
            results=photo_results,
            total_found=len(photo_results),
            query="family photos"
        )
        
    except Exception as error:
        logger.error(f"Error finding family photos: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/stranger_photos", response_model=SearchResponse)
async def find_stranger_photos(limit: int = Query(10, ge=1, le=50)):
    """Find photos containing unknown/stranger faces."""
    try:
        photos = list(face_collection.find().sort("date", -1).limit(config["search_limit"]))
        
        results = rag_engine.search_photos(
            query="photos with strangers or unknown people",
            photos_data=photos,
            top_k=limit
        )
        
        photo_results = []
        for result in results:
            photo = result['photo']
            photo_results.append(PhotoResult(
                filename=photo.get('filename', ''),
                timestamp=photo.get('date', 0),
                camera_location=photo.get('camera_location', ''),
                relevance_score=result['relevance_score'],
                explanation=result['search_explanation'],
                faces_detected=result['photo'].get('faces_detected', {}),
                s3_url=photo.get('s3_file_url', ''),
                bson_time=photo.get('bsonTime', ''),
                face_count=photo.get('face_count', 0)
            ))
        
        return SearchResponse(
            results=photo_results,
            total_found=len(photo_results),
            query="stranger photos"
        )
        
    except Exception as error:
        logger.error(f"Error finding stranger photos: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/known_faces")
async def get_known_faces():
    """Get list of known faces in the database."""
    try:
        known_faces = list(face_manager.known_faces.keys())
        return {"known_faces": known_faces}
    except Exception as error:
        logger.error(f"Error getting known faces: {error}")
        raise HTTPException(status_code=500, detail=str(error))

@app.delete("/known_faces/{name}")
async def remove_known_face(name: str):
    """Remove a person from the known faces database."""
    try:
        success = face_manager.remove_person(name)
        if success:
            return {"message": f"Successfully removed {name} from known faces database"}
        else:
            raise HTTPException(status_code=404, detail=f"Person {name} not found in database")
    except Exception as error:
        logger.error(f"Error removing face: {error}")
        raise HTTPException(status_code=500, detail=str(error))

def main():
    """Main function to run the search service."""
    import uvicorn
    uvicorn.run(app, host=config["api_host"], port=config["api_port"])

if __name__ == "__main__":
    main()
