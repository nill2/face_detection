# Image Processor Module

A Python application for processing images, detecting faces using OpenCV, and storing the results in MongoDB. This project is designed to handle images uploaded to an FTP server and process them efficiently in a scalable manner.

---

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in images.
- **MongoDB Integration**: Connects to MongoDB to fetch and store processed image data.
- **Logging**: Provides detailed logs for all operations, ensuring easy debugging and monitoring.
- **Extensibility**: Modular structure allows for easy addition of new features.

---

## Requirements

- Python 3.10+
- MongoDB
- Dependencies specified in `requirements.txt` or `environment.yml`

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/image-processor.git
   cd image-processor
   ```

2. **Set Up Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

   using `Conda`

   ```bash
   conda env create -f environment.yml
   conda activate image-processor
   ```

3. **Run the Application**

   ```bash
   python main.py
   ```

## File Structure

```plain text
.
├── main.py                     # Entry point for the application
├── image_processor/            # Core processing module
│   ├── processor.py            # PhotoProcessor class for face detection
│   ├── config.py               # Configuration variables
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment configuration
├── tests/                      # Unit and integration tests
│   ├── test_unit.py            # Unit tests for processor module
│   └── test_core.py            # E2E tests for the entire pipeline
└── README.md                   # Project documentation
```

## Usage

1. Upload images to the MongoDB collection (MONGO_COLLECTION) via the FTP server or other sources.
2. The PhotoProcessor will:
   Fetch images from MongoDB.
   Detect faces using OpenCV.
   Store images with detected faces in the FACE_COLLECTION.
3. Logs will be generated to track processing details.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this module.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
