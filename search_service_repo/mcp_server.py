#!/usr/bin/env python3
"""
MCP (Model Context Protocol) server for CCTV photo search.

This server provides AI models with access to your CCTV photo search capabilities
through the MCP protocol.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, CallToolResult, ListResourcesRequest, ListResourcesResult,
    ReadResourceRequest, ReadResourceResult, ListToolsRequest, ListToolsResult
)

from .config import get_config
from .embeddings import VisualEmbeddingEngine, FaceDatabaseManager
from .rag_search import RAGSearchEngine
from pymongo import MongoClient, DESCENDING

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

class CCTVMCPHandler:
    """MCP handler for CCTV photo search capabilities."""
    
    def __init__(self):
        """Initialize the MCP handler."""
        self.visual_engine = VisualEmbeddingEngine()
        self.rag_engine = RAGSearchEngine(self.visual_engine)
        self.face_manager = FaceDatabaseManager(self.visual_engine)
        
        # Connect to MongoDB
        self.client = MongoClient(config["mongo_host"], config["mongo_port"])
        self.db = self.client[config["mongo_db"]]
        self.face_collection = self.db[config["face_collection"]]
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known faces from database."""
        try:
            # Look for photos with identified faces
            known_faces_docs = self.face_collection.find({
                "embeddings.face_0_person": {"$exists": True}
            }).limit(100)
            
            for doc in known_faces_docs:
                embeddings = doc.get('embeddings', {})
                for key, value in embeddings.items():
                    if key.endswith('_person'):
                        face_id = key.replace('_person', '')
                        person_name = value.decode('utf-8') if isinstance(value, bytes) else value
                        if person_name != 'unknown' and face_id in embeddings:
                            self.face_manager.known_faces[person_name] = embeddings[face_id]
            
            logger.info(f"Loaded {len(self.face_manager.known_faces)} known faces")
        except Exception as error:
            logger.error(f"Failed to load known faces: {error}")
    
    async def search_photos(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search photos using natural language query."""
        try:
            # Get photos from database
            photos = list(self.face_collection.find().sort("date", -1).limit(1000))
            
            # Perform RAG search
            results = self.rag_engine.search_photos(
                query=query,
                photos_data=photos,
                top_k=limit,
                filters=filters or {}
            )
            
            # Format results for MCP
            formatted_results = []
            for result in results:
                photo = result['photo']
                formatted_results.append({
                    'filename': photo.get('filename', ''),
                    'timestamp': photo.get('date', 0),
                    'camera_location': photo.get('camera_location', ''),
                    'relevance_score': result['relevance_score'],
                    'explanation': result['search_explanation'],
                    'faces_detected': result['photo'].get('faces_detected', {}),
                    's3_url': photo.get('s3_file_url', ''),
                    'bson_time': photo.get('bsonTime', '')
                })
            
            return formatted_results
            
        except Exception as error:
            logger.error(f"Error in photo search: {error}")
            return []
    
    async def add_known_face(self, name: str, image_data: bytes) -> bool:
        """Add a person to the known faces database."""
        return self.face_manager.add_person(name, image_data)
    
    async def get_photo_stats(self) -> Dict[str, Any]:
        """Get statistics about the photo database."""
        try:
            total_photos = self.face_collection.count_documents({})
            
            # Get recent photos (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_photos = self.face_collection.count_documents({
                "date": {"$gte": yesterday.timestamp()}
            })
            
            # Count photos with faces
            photos_with_faces = self.face_collection.count_documents({
                "has_faces": True
            })
            
            # Count known vs unknown faces
            known_faces_count = self.face_collection.count_documents({
                "embeddings.face_0_person": {"$ne": b"unknown"}
            })
            
            return {
                "total_photos": total_photos,
                "recent_photos_24h": recent_photos,
                "photos_with_faces": photos_with_faces,
                "known_faces_detected": known_faces_count,
                "unknown_faces_detected": photos_with_faces - known_faces_count,
                "known_faces_in_db": len(self.face_manager.known_faces)
            }
            
        except Exception as error:
            logger.error(f"Error getting photo stats: {error}")
            return {}

# Initialize MCP handler
handler = CCTVMCPHandler()

# Create MCP server
server = Server("cctv-photo-search")

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="cctv://photos",
            name="CCTV Photos",
            description="Database of CCTV photos with face detection and embeddings",
            mimeType="application/json"
        ),
        Resource(
            uri="cctv://faces",
            name="Known Faces",
            description="Database of known faces for recognition",
            mimeType="application/json"
        ),
        Resource(
            uri="cctv://stats",
            name="Photo Statistics",
            description="Statistics about the photo database",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "cctv://photos":
        # Return sample of recent photos
        photos = await handler.search_photos("recent photos", limit=5)
        return json.dumps(photos, indent=2)
    
    elif uri == "cctv://faces":
        # Return known faces
        known_faces = list(handler.face_manager.known_faces.keys())
        return json.dumps({"known_faces": known_faces}, indent=2)
    
    elif uri == "cctv://stats":
        # Return statistics
        stats = await handler.get_photo_stats()
        return json.dumps(stats, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_photos",
            description="Search CCTV photos using natural language queries. Can find photos with specific people, times, locations, or activities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'photos of me and my son', 'strangers at front door', 'morning activity')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "time_range": {
                        "type": "object",
                        "description": "Optional time range filter",
                        "properties": {
                            "start": {"type": "number", "description": "Start timestamp"},
                            "end": {"type": "number", "description": "End timestamp"}
                        }
                    },
                    "camera_location": {
                        "type": "string",
                        "description": "Optional camera location filter"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="add_known_face",
            description="Add a person to the known faces database for recognition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Person's name (e.g., 'me', 'son', 'dad')"
                    },
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data containing the person's face"
                    }
                },
                "required": ["name", "image_data"]
            }
        ),
        Tool(
            name="get_photo_stats",
            description="Get statistics about the photo database including face detection counts.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="find_family_photos",
            description="Find photos containing family members (you and your son).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="find_stranger_photos",
            description="Find photos containing unknown/stranger faces.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "search_photos":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            filters = {}
            
            if "time_range" in arguments:
                filters["time_range"] = arguments["time_range"]
            if "camera_location" in arguments:
                filters["camera_location"] = arguments["camera_location"]
            
            results = await handler.search_photos(query, limit, filters)
            
            if not results:
                return [TextContent(type="text", text="No photos found matching your query.")]
            
            # Format results
            response = f"Found {len(results)} photos matching '{query}':\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['filename']}\n"
                response += f"   Time: {datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                response += f"   Location: {result['camera_location']}\n"
                response += f"   Relevance: {result['relevance_score']:.3f}\n"
                response += f"   Explanation: {result['explanation']}\n"
                if result['faces_detected']:
                    faces = result['faces_detected']
                    if faces.get('known_faces'):
                        response += f"   Known faces: {', '.join(faces['known_faces'])}\n"
                    if faces.get('unknown_faces', 0) > 0:
                        response += f"   Unknown faces: {faces['unknown_faces']}\n"
                response += "\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "add_known_face":
            name = arguments["name"]
            image_data_b64 = arguments["image_data"]
            
            # Decode base64 image data
            import base64
            image_data = base64.b64decode(image_data_b64)
            
            success = await handler.add_known_face(name, image_data)
            
            if success:
                return [TextContent(type="text", text=f"Successfully added {name} to known faces database.")]
            else:
                return [TextContent(type="text", text=f"Failed to add {name} to known faces database. Make sure the image contains a clear face.")]
        
        elif name == "get_photo_stats":
            stats = await handler.get_photo_stats()
            
            response = "CCTV Photo Database Statistics:\n\n"
            response += f"Total photos: {stats.get('total_photos', 0)}\n"
            response += f"Recent photos (24h): {stats.get('recent_photos_24h', 0)}\n"
            response += f"Photos with faces: {stats.get('photos_with_faces', 0)}\n"
            response += f"Known faces detected: {stats.get('known_faces_detected', 0)}\n"
            response += f"Unknown faces detected: {stats.get('unknown_faces_detected', 0)}\n"
            response += f"Known faces in database: {stats.get('known_faces_in_db', 0)}\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "find_family_photos":
            limit = arguments.get("limit", 10)
            results = await handler.search_photos("photos with me and my son", limit)
            
            if not results:
                return [TextContent(type="text", text="No family photos found. Make sure you've added yourself and your son to the known faces database.")]
            
            response = f"Found {len(results)} family photos:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['filename']}\n"
                response += f"   Time: {datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                response += f"   Location: {result['camera_location']}\n"
                if result['faces_detected'].get('known_faces'):
                    response += f"   Family members: {', '.join(result['faces_detected']['known_faces'])}\n"
                response += "\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "find_stranger_photos":
            limit = arguments.get("limit", 10)
            results = await handler.search_photos("photos with strangers or unknown people", limit)
            
            if not results:
                return [TextContent(type="text", text="No photos with strangers found.")]
            
            response = f"Found {len(results)} photos with strangers:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['filename']}\n"
                response += f"   Time: {datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                response += f"   Location: {result['camera_location']}\n"
                if result['faces_detected'].get('unknown_faces', 0) > 0:
                    response += f"   Unknown faces: {result['faces_detected']['unknown_faces']}\n"
                response += "\n"
            
            return [TextContent(type="text", text=response)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as error:
        logger.error(f"Error handling tool call {name}: {error}")
        return [TextContent(type="text", text=f"Error: {str(error)}")]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cctv-photo-search",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
