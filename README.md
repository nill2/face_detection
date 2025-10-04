# CCTV Face Detection & Efficient Search Embeddings

This application analyzes photos from your CCTV camera, detects faces, and generates efficient embeddings optimized for search queries.

## 🎯 Purpose

- **Face Detection**: Detect faces using YOLOv8n-face model
- **Vehicle Detection**: Detect cars and motorcycles using color/shape analysis
- **Efficient Embeddings**: Generate optimized embeddings for fast search
- **Search Optimization**: Prepare data for external search service

## 🚀 Features

### Face Detection
- **YOLOv8n-face**: Primary face detection model
- **OpenCV HaarCascade**: Fallback detection method
- **High Accuracy**: Optimized for CCTV camera conditions

### Vehicle Detection
- **Car Detection**: Color-based detection (white, black, gray, blue, red)
- **Motorcycle Detection**: Shape and color analysis
- **Vehicle Score**: Confidence score for vehicle presence

### Efficient Embeddings
- **Visual Content**: CLIP-based image understanding (512D)
- **Temporal Context**: Time-based features (9D)
- **Scene Analysis**: Lighting, indoor/outdoor, quality (7D)
- **Object Detection**: Face count, vehicle score, face regions
- **Search Text**: Optimized text descriptions for semantic search

## 🏗️ Architecture

```
Raw Photos → Face Detection → Efficient Embeddings → MongoDB Storage
     ↓              ↓              ↓                    ↓
  images        faces found    search_embeddings    faces collection
 collection                   + metadata
```

## 📊 Database Schema

### Input: `images` collection
```json
{
  "filename": "cctv_camera_1_1234567890.jpg",
  "data": "<binary_image_data>",
  "date": 1234567890,
  "camera_location": "front_door"
}
```

### Output: `faces` collection
```json
{
  "filename": "cctv_camera_1_1234567890.jpg",
  "data": "<binary_image_data>",
  "embedding": "<legacy_text_embedding>",
  "search_embeddings": {
    "visual": "<clip_embedding_512d>",
    "temporal": "<temporal_features_9d>",
    "scene": "<scene_features_7d>",
    "face_count": "<face_count_1d>",
    "face_0": "<face_region_4096d>",
    "face_0_bbox": "<bounding_box_4d>",
    "vehicle_score": "<vehicle_confidence_1d>",
    "search_text": "<text_embedding_384d>"
  },
  "camera_location": "front_door",
  "date": 1234567890,
  "has_faces": true,
  "face_count": 2,
  "vehicle_detected": 0.8,
  "processing_timestamp": 1234567890
}
```

## 🚀 Usage

### Run Face Processing
```bash
# Process photos once
python main.py

# Or run continuously
python -c "
from image_processor.processor import PhotoProcessor
import time
p = PhotoProcessor()
while True:
    p.process_photos()
    time.sleep(60)
"
```

### Configuration
```bash
# Environment variables
export MONGO_HOST=localhost
export MONGO_PORT=27017
export MONGO_DB=photos
export MONGO_COLLECTION=images
export FACE_COLLECTION=nill-home-faces
export FACE_DETECTION_MODEL=yolov8n-face.pt
export FACES_HISTORY_DAYS=30
```

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `opencv-python` - Face detection and image processing
- `ultralytics` - YOLO models
- `transformers` - CLIP embeddings
- `torch` - Neural networks
- `sentence-transformers` - Text embeddings
- `pymongo` - Database

## 📈 Performance

### Processing Speed
- **Face Detection**: ~50-100ms per image
- **Embedding Generation**: ~200-500ms per image
- **Total Processing**: ~250-600ms per image

### Memory Usage
- **CLIP Model**: ~1.5GB
- **Text Model**: ~200MB
- **Total Memory**: ~2GB

### Storage Efficiency
- **Visual Embedding**: 2KB (512D float32)
- **Temporal Embedding**: 36 bytes (9D float32)
- **Scene Embedding**: 28 bytes (7D float32)
- **Face Embeddings**: 4KB per face (64x64 grayscale)
- **Total per Image**: ~6-10KB (depending on face count)

## 🔍 Search Capabilities

The generated embeddings enable efficient search for:

### Face-based Search
- **"photos with faces"** - Find all photos with detected faces
- **"photos with 2 faces"** - Find photos with specific face count
- **"photos with me"** - Face recognition (requires external service)

### Vehicle Search
- **"photos with cars"** - Find photos with vehicle detection
- **"photos with motorcycles"** - Find motorcycle detections
- **"high vehicle activity"** - Find photos with high vehicle scores

### Time-based Search
- **"morning photos"** - Find photos from morning hours
- **"night time"** - Find photos from night hours
- **"weekend activity"** - Find photos from weekends
- **"photos from last week"** - Time range queries

### Scene-based Search
- **"outdoor photos"** - Find outdoor scenes
- **"indoor photos"** - Find indoor scenes
- **"bright photos"** - Find well-lit photos
- **"dark photos"** - Find low-light photos

### Location Search
- **"front door camera"** - Find photos from specific camera
- **"backyard photos"** - Location-based filtering

## 🎯 Embedding Types Explained

### 1. Visual Content (CLIP)
- **Purpose**: Understand what's in the image
- **Size**: 512 dimensions
- **Use**: "photos with people", "outdoor scenes", "vehicles"

### 2. Temporal Context
- **Purpose**: Time-based search
- **Size**: 9 dimensions
- **Features**: Hour, day, month, time periods, weekday/weekend
- **Use**: "morning photos", "night time", "weekend activity"

### 3. Scene Analysis
- **Purpose**: Environment and quality
- **Size**: 7 dimensions
- **Features**: Brightness, contrast, color, quality, indoor/outdoor
- **Use**: "bright photos", "outdoor scenes", "high quality"

### 4. Object Detection
- **Purpose**: Count and locate objects
- **Features**: Face count, face regions, vehicle score
- **Use**: "photos with faces", "photos with cars", "multiple people"

### 5. Search Text
- **Purpose**: Semantic search
- **Size**: 384 dimensions
- **Content**: "morning bright camera front_door weekday"
- **Use**: Natural language queries

## 🔗 Integration with Search Service

This service prepares data for an external search service:

1. **Processes photos** and generates embeddings
2. **Stores in MongoDB** with rich metadata
3. **Search service reads** from `faces` collection
4. **Enables fast queries** using pre-computed embeddings

## 📝 Example Queries

The search service can handle queries like:

- **"photos of me and my son"** - Family recognition
- **"strangers at front door"** - Unknown people detection
- **"morning activity"** - Time-based search
- **"photos with cars"** - Vehicle detection
- **"bright outdoor photos"** - Scene-based search
- **"photos from last week"** - Temporal filtering

## 🚀 Next Steps

1. **Run this service** to process your CCTV photos
2. **Create search service** (separate repo) to query the data
3. **Add face recognition** to identify family members
4. **Build web interface** for easy photo search

This service provides the **data processing foundation** for intelligent CCTV photo search!