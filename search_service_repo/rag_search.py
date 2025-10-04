"""
RAG (Retrieval-Augmented Generation) search system for CCTV photos.

This module provides intelligent search capabilities using embeddings
and natural language queries for finding specific photos.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from .embeddings import VisualEmbeddingEngine, FaceDatabaseManager

logger = logging.getLogger(__name__)


class RAGSearchEngine:
    """RAG-powered search engine for CCTV photos with natural language queries."""
    
    def __init__(self, visual_engine: VisualEmbeddingEngine):
        """Initialize the RAG search engine."""
        self.visual_engine = visual_engine
        
    def search_photos(self, 
                     query: str,
                     photos_data: List[Dict[str, Any]],
                     top_k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search photos using natural language queries.
        
        Args:
            query: Natural language search query
            photos_data: List of photo documents from MongoDB
            top_k: Number of results to return
            filters: Optional filters (time_range, camera_location, etc.)
            
        Returns:
            List of matching photo documents with relevance scores
        """
        try:
            # Parse the query to extract search criteria
            search_criteria = self._parse_query(query)
            
            # Apply filters
            filtered_photos = self._apply_filters(photos_data, filters or {})
            
            # Extract embeddings for search
            candidate_embeddings = []
            for i, photo in enumerate(filtered_photos):
                if 'embeddings' in photo:
                    candidate_embeddings.append({
                        'index': i,
                        'embeddings': photo['embeddings'],
                        'metadata': {
                            'filename': photo.get('filename', ''),
                            'timestamp': photo.get('date', 0),
                            'camera_location': photo.get('camera_location', ''),
                            'faces_detected': self._extract_face_info(photo)
                        }
                    })
            
            # Perform multi-modal search
            results = self._perform_search(search_criteria, candidate_embeddings, top_k)
            
            # Format results
            formatted_results = []
            for idx, score in results:
                if idx < len(filtered_photos):
                    photo = filtered_photos[idx]
                    formatted_results.append({
                        'photo': photo,
                        'relevance_score': score,
                        'search_explanation': self._generate_explanation(search_criteria, photo)
                    })
            
            return formatted_results
            
        except Exception as error:
            logger.error(f"Error in photo search: {error}")
            return []
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract search criteria."""
        query_lower = query.lower()
        criteria = {
            'text_query': query,
            'visual_similarity': None,
            'time_filters': {},
            'scene_filters': {},
            'activity_filters': {},
            'face_filters': {}
        }
        
        # Time-based queries
        if any(word in query_lower for word in ['morning', 'am', '9am', '8am', '7am']):
            criteria['time_filters']['hour'] = 9
        elif any(word in query_lower for word in ['afternoon', 'pm', '2pm', '3pm', '4pm']):
            criteria['time_filters']['hour'] = 15
        elif any(word in query_lower for word in ['evening', '6pm', '7pm', '8pm']):
            criteria['time_filters']['hour'] = 19
        elif any(word in query_lower for word in ['night', 'midnight', '11pm', '12am']):
            criteria['time_filters']['hour'] = 23
            
        if any(word in query_lower for word in ['weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            criteria['time_filters']['weekday'] = 1
        elif any(word in query_lower for word in ['weekend', 'saturday', 'sunday']):
            criteria['time_filters']['weekday'] = 6
        
        # Location-based queries
        if any(word in query_lower for word in ['front door', 'entrance', 'doorway']):
            criteria['scene_filters']['location'] = 'front_door'
        elif any(word in query_lower for word in ['backyard', 'garden', 'outdoor']):
            criteria['scene_filters']['location'] = 'backyard'
        elif any(word in query_lower for word in ['indoor', 'inside', 'living room']):
            criteria['scene_filters']['indoor_outdoor'] = False
        elif any(word in query_lower for word in ['outdoor', 'outside', 'garden']):
            criteria['scene_filters']['indoor_outdoor'] = True
        
        # Activity-based queries
        if any(word in query_lower for word in ['busy', 'active', 'movement', 'people']):
            criteria['activity_filters']['high_activity'] = True
        elif any(word in query_lower for word in ['quiet', 'empty', 'still', 'no one']):
            criteria['activity_filters']['high_activity'] = False
        
        # Face-based queries
        if any(word in query_lower for word in ['me', 'myself', 'dad', 'father']):
            criteria['face_filters']['known_person'] = 'me'
        elif any(word in query_lower for word in ['son', 'child', 'boy']):
            criteria['face_filters']['known_person'] = 'son'
        elif any(word in query_lower for word in ['stranger', 'unknown', 'visitor']):
            criteria['face_filters']['unknown_person'] = True
        elif any(word in query_lower for word in ['family', 'us', 'together']):
            criteria['face_filters']['family_member'] = True
        
        return criteria
    
    def _apply_filters(self, photos_data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply additional filters to photo data."""
        filtered = photos_data
        
        # Time range filter
        if 'time_range' in filters:
            start_time = filters['time_range'].get('start', 0)
            end_time = filters['time_range'].get('end', float('inf'))
            filtered = [p for p in filtered if start_time <= p.get('date', 0) <= end_time]
        
        # Camera location filter
        if 'camera_location' in filters:
            location = filters['camera_location']
            filtered = [p for p in filtered if p.get('camera_location', '') == location]
        
        return filtered
    
    def _extract_face_info(self, photo: Dict[str, Any]) -> Dict[str, Any]:
        """Extract face information from photo embeddings."""
        face_info = {
            'faces_detected': 0,
            'known_faces': [],
            'unknown_faces': 0
        }
        
        if 'embeddings' in photo:
            embeddings = photo['embeddings']
            
            # Count faces and identify known/unknown
            for key, value in embeddings.items():
                if key.startswith('face_') and not key.endswith('_bbox') and not key.endswith('_confidence'):
                    face_info['faces_detected'] += 1
                    
                    # Check if this face was identified
                    person_key = f'{key}_person'
                    if person_key in embeddings:
                        person_name = embeddings[person_key].decode('utf-8') if isinstance(value, bytes) else value
                        if person_name != 'unknown':
                            face_info['known_faces'].append(person_name)
                        else:
                            face_info['unknown_faces'] += 1
                    else:
                        face_info['unknown_faces'] += 1
        
        return face_info
    
    def _perform_search(self, 
                       criteria: Dict[str, Any], 
                       candidate_embeddings: List[Dict], 
                       top_k: int) -> List[Tuple[int, float]]:
        """Perform the actual search using multiple modalities."""
        
        # For now, implement a simple text-based search
        # In a full implementation, you would use the embeddings for similarity search
        results = []
        
        for i, candidate in enumerate(candidate_embeddings):
            score = 0.0
            
            # Simple keyword matching for now
            query_lower = criteria.get('text_query', '').lower()
            metadata = candidate['metadata']
            
            # Check filename
            if query_lower in metadata['filename'].lower():
                score += 0.5
            
            # Check camera location
            if query_lower in metadata['camera_location'].lower():
                score += 0.3
            
            # Check face information
            faces = metadata['faces_detected']
            if 'me' in query_lower and 'me' in faces.get('known_faces', []):
                score += 0.8
            if 'son' in query_lower and 'son' in faces.get('known_faces', []):
                score += 0.8
            if 'stranger' in query_lower and faces.get('unknown_faces', 0) > 0:
                score += 0.7
            
            if score > 0:
                results.append((i, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _filter_by_faces(self, 
                        results: List[Tuple[int, float]], 
                        candidate_embeddings: List[Dict],
                        face_filters: Dict[str, Any]) -> List[Tuple[int, float]]:
        """Filter results based on face criteria."""
        filtered_results = []
        
        for idx, score in results:
            if idx < len(candidate_embeddings):
                candidate = candidate_embeddings[idx]
                face_info = candidate['metadata']['faces_detected']
                
                # Check face criteria
                matches = True
                
                if 'known_person' in face_filters:
                    target_person = face_filters['known_person']
                    if target_person not in face_info['known_faces']:
                        matches = False
                
                if 'unknown_person' in face_filters and face_filters['unknown_person']:
                    if face_info['unknown_faces'] == 0:
                        matches = False
                
                if 'family_member' in face_filters and face_filters['family_member']:
                    if not face_info['known_faces']:
                        matches = False
                
                if matches:
                    filtered_results.append((idx, score))
        
        return filtered_results
    
    def _generate_explanation(self, criteria: Dict[str, Any], photo: Dict[str, Any]) -> str:
        """Generate explanation for why a photo matched the search."""
        explanations = []
        
        # Time-based explanation
        if criteria.get('time_filters'):
            timestamp = photo.get('date', 0)
            dt = datetime.fromtimestamp(timestamp)
            explanations.append(f"Taken at {dt.strftime('%H:%M on %A')}")
        
        # Location-based explanation
        if criteria.get('scene_filters', {}).get('location'):
            location = photo.get('camera_location', 'unknown location')
            explanations.append(f"From {location} camera")
        
        # Face-based explanation
        face_info = self._extract_face_info(photo)
        if face_info['faces_detected'] > 0:
            if face_info['known_faces']:
                explanations.append(f"Contains {', '.join(face_info['known_faces'])}")
            if face_info['unknown_faces'] > 0:
                explanations.append(f"Contains {face_info['unknown_faces']} unknown person(s)")
        
        return "; ".join(explanations) if explanations else "Matched search criteria"
