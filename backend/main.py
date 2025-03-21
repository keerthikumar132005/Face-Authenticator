from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
import numpy as np
import datetime
from typing import List, Dict
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import cv2
from scipy.spatial.distance import cosine

load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Weaviate client
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
)

# Create schema if it doesn't exist
try:
    collections = client.collections.list_all()
    if "FaceEmbeddings" not in [c.name for c in collections]:
        client.collections.create(
            name="FaceEmbeddings",
            vectorizer_config=weaviate.config.Configure.Vectorizer.none(),
            properties=[
                weaviate.properties.Property(
                    name="user_id",
                    data_type=weaviate.properties.DataType.TEXT
                ),
                weaviate.properties.Property(
                    name="timestamp",
                    data_type=weaviate.properties.DataType.DATE
                ),
                weaviate.properties.Property(
                    name="vector",
                    data_type=weaviate.properties.DataType.NUMBER_ARRAY
                )
            ]
        )
        print("Created FaceEmbeddings collection")
except Exception as e:
    print(f"Error creating schema: {e}")

class FaceData(BaseModel):
    user_id: str
    landmarks: List[Dict[str, float]]

def validate_landmarks(landmarks: List[Dict[str, float]]) -> bool:
    """Validate that landmarks contain all required points."""
    if not landmarks or len(landmarks) < 68:  # We need at least 68 facial landmarks
        return False
    
    # Check that all landmarks have x and y coordinates
    for lm in landmarks:
        if 'x' not in lm or 'y' not in lm:
            return False
        if not isinstance(lm['x'], (int, float)) or not isinstance(lm['y'], (int, float)):
            return False
    
    return True

def normalize_landmarks(landmarks: List[Dict[str, float]], target_length: int = 1434) -> np.ndarray:
    """Normalize facial landmarks to be scale, translation and rotation invariant."""
    if not validate_landmarks(landmarks):
        raise ValueError("Invalid landmarks provided")
        
    # Convert landmarks to numpy array
    points = np.array([[lm['x'], lm['y']] for lm in landmarks])
    
    if points.size == 0:
        raise ValueError("Empty landmarks array")
    
    # Center the points by subtracting mean
    centered = points - np.mean(points, axis=0)
    
    # Scale to unit size
    scale = np.sqrt(np.sum(centered ** 2)) / len(centered)
    normalized = centered / (scale + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Calculate principal components for rotation invariance
    covariance = np.cov(normalized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Project points onto principal components
    rotated = np.dot(normalized, eigenvectors)
    
    # Flatten and pad/truncate to match target length
    flattened = rotated.flatten()
    current_length = len(flattened)
    
    if current_length < target_length:
        # Pad with zeros if shorter
        padded = np.pad(flattened, (0, target_length - current_length))
        return padded
    else:
        # Truncate if longer
        return flattened[:target_length]

def compute_geometric_features(landmarks: List[Dict[str, float]], target_length: int = 1434) -> np.ndarray:
    """Compute geometric features from facial landmarks."""
    if not validate_landmarks(landmarks):
        raise ValueError("Invalid landmarks provided")
        
    points = np.array([[lm['x'], lm['y']] for lm in landmarks])
    
    if len(points) < 68:
        raise ValueError(f"Insufficient landmarks. Expected 68, got {len(points)}")
    
    try:
        # Extract facial feature points for key regions
        left_eye = points[36:42]
        right_eye = points[42:48]
        nose = points[27:36]
        mouth = points[48:68]
        jaw = points[0:17]
        left_eyebrow = points[17:22]
        right_eyebrow = points[22:27]
        
        features = []
        
        # Function to compute angle between three points
        def compute_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return angle
        
        # Face width and height (base measurements)
        face_width = np.linalg.norm(jaw[0] - jaw[-1])
        face_height = np.linalg.norm(np.mean(jaw[0:2], axis=0) - points[8])
        
        if face_width == 0 or face_height == 0:
            raise ValueError("Invalid face measurements (zero width or height)")
        
        # 1. Eye Measurements
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        left_eye_height = np.mean([np.linalg.norm(left_eye[1] - left_eye[5]), 
                                 np.linalg.norm(left_eye[2] - left_eye[4])])
        right_eye_height = np.mean([np.linalg.norm(right_eye[1] - right_eye[5]),
                                  np.linalg.norm(right_eye[2] - right_eye[4])])
        eye_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        
        # 2. Eyebrow Measurements
        left_eyebrow_arch = np.mean([compute_angle(left_eyebrow[i], left_eyebrow[i+1], left_eyebrow[i+2]) 
                                   for i in range(len(left_eyebrow)-2)])
        right_eyebrow_arch = np.mean([compute_angle(right_eyebrow[i], right_eyebrow[i+1], right_eyebrow[i+2]) 
                                    for i in range(len(right_eyebrow)-2)])
        eyebrow_distance = np.linalg.norm(np.mean(left_eyebrow, axis=0) - np.mean(right_eyebrow, axis=0))
        
        # 3. Nose Measurements
        nose_length = np.linalg.norm(nose[0] - nose[-1])
        nose_width = np.linalg.norm(nose[4] - nose[8])
        nose_tip_angle = compute_angle(nose[1], nose[4], nose[7])
        nose_bridge_length = np.linalg.norm(nose[0] - nose[3])
        
        # 4. Mouth Measurements
        mouth_width = np.linalg.norm(mouth[0] - mouth[6])
        mouth_height = np.mean([
            np.linalg.norm(mouth[2] - mouth[10]),
            np.linalg.norm(mouth[3] - mouth[9]),
            np.linalg.norm(mouth[4] - mouth[8])
        ])
        mouth_area = mouth_width * mouth_height
        lip_thickness = np.mean([
            np.linalg.norm(mouth[2] - mouth[14]),
            np.linalg.norm(mouth[3] - mouth[13]),
            np.linalg.norm(mouth[4] - mouth[12])
        ])
        
        # 5. Jaw Measurements
        jaw_angles = [compute_angle(jaw[i], jaw[i+1], jaw[i+2]) for i in range(len(jaw)-2)]
        mean_jaw_angle = np.mean(jaw_angles)
        jaw_length = sum(np.linalg.norm(jaw[i] - jaw[i+1]) for i in range(len(jaw)-1))
        
        # 6. Face Shape Ratios
        face_ratio = face_width / face_height
        eye_spacing_ratio = eye_distance / face_width
        nose_ratio = nose_width / nose_length
        mouth_ratio = mouth_width / mouth_height
        
        # 7. Relative Positions
        eye_to_nose_distance = np.linalg.norm(np.mean([np.mean(left_eye, axis=0), 
                                                      np.mean(right_eye, axis=0)], axis=0) - nose[-1])
        nose_to_mouth_distance = np.linalg.norm(nose[-1] - np.mean(mouth[2:5], axis=0))
        eye_to_eyebrow_distance = np.mean([
            np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(left_eyebrow, axis=0)),
            np.linalg.norm(np.mean(right_eye, axis=0) - np.mean(right_eyebrow, axis=0))
        ])
        
        # Compile all features (normalized by face width or height as appropriate)
        features.extend([
            # Eye features
            left_eye_width / face_width,
            right_eye_width / face_width,
            left_eye_height / face_height,
            right_eye_height / face_height,
            eye_distance / face_width,
            
            # Eyebrow features
            left_eyebrow_arch,
            right_eyebrow_arch,
            eyebrow_distance / face_width,
            
            # Nose features
            nose_length / face_height,
            nose_width / face_width,
            nose_tip_angle,
            nose_bridge_length / face_height,
            
            # Mouth features
            mouth_width / face_width,
            mouth_height / face_height,
            mouth_area / (face_width * face_height),
            lip_thickness / face_height,
            
            # Jaw features
            mean_jaw_angle,
            jaw_length / face_width,
            
            # Face shape ratios
            face_ratio,
            eye_spacing_ratio,
            nose_ratio,
            mouth_ratio,
            
            # Relative positions
            eye_to_nose_distance / face_height,
            nose_to_mouth_distance / face_height,
            eye_to_eyebrow_distance / face_height
        ])
        
        # Add normalized landmark positions
        normalized_points = normalize_landmarks(landmarks)
        features.extend(normalized_points.flatten())
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Check for invalid values
        if not np.all(np.isfinite(features)):
            raise ValueError("Invalid feature values detected (inf or nan)")
        
        # Ensure consistent length
        if len(features) < target_length:
            features = np.pad(features, (0, target_length - len(features)))
        else:
            features = features[:target_length]
        
        return features
        
    except Exception as e:
        raise ValueError(f"Error computing geometric features: {str(e)}")

def extract_face_embedding(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """Extract face embedding using geometric features and normalized landmarks."""
    try:
        # Validate landmarks
        if not validate_landmarks(landmarks):
            raise ValueError("Invalid or insufficient landmarks provided")
        
        # Compute geometric features with target length matching existing vectors
        features = compute_geometric_features(landmarks, target_length=1434)
        
        # Check for invalid values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Invalid values in features (NaN or Inf)")
        
        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm < 1e-10:
            raise ValueError("Feature vector has zero norm")
            
        embedding = features / (norm + 1e-10)
        
        return embedding
        
    except Exception as e:
        print(f"Error in extract_face_embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing face: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    try:
        collections = client.collections.list_all()
        return {
            "status": "ok",
            "message": "Face Authentication API is running",
            "weaviate_connection": "ok",
            "collections": [c.name for c in collections]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Face Authentication API is running but Weaviate connection failed",
            "error": str(e)
        }

@app.post("/register")
async def register_face(data: FaceData):
    """Register a new face in the database."""
    try:
        # Validate input data
        if not data.landmarks:
            raise HTTPException(status_code=400, detail="No landmarks provided")
            
        if not validate_landmarks(data.landmarks):
            raise HTTPException(status_code=400, detail="Invalid or insufficient landmarks provided")
            
        # Extract face embedding
        embedding = extract_face_embedding(data.landmarks)
        
        # Validate embedding
        if len(embedding) != 1434:
            raise HTTPException(
                status_code=500, 
                detail=f"Invalid embedding length: expected 1434, got {len(embedding)}"
            )
        
        # Get the FaceEmbeddings collection
        collection = client.collections.get("FaceEmbeddings")
        
        try:
            # Format timestamp
            current_time = datetime.datetime.now(datetime.timezone.utc)
            rfc3339_timestamp = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Store embedding
            obj = collection.data.insert(
                properties={
                    "user_id": data.user_id,
                    "timestamp": rfc3339_timestamp,
                    "vector": embedding.tolist()
                },
                vector=embedding.tolist()
            )
            print(f"Successfully stored face embedding with ID: {obj}")
            return {"message": "Face registered successfully", "user_id": data.user_id}
            
        except Exception as e:
            print(f"Error storing in Weaviate: {e}")
            raise HTTPException(status_code=500, detail=f"Error storing face data: {str(e)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in register_face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/authenticate")
async def authenticate_face(data: FaceData):
    """Authenticate a face against stored embeddings using geometric features."""
    try:
        # Extract face embedding
        input_embedding = extract_face_embedding(data.landmarks)
        
        # Get the FaceEmbeddings collection
        collection = client.collections.get("FaceEmbeddings")
        
        try:
            # Query for the user's face embeddings
            response = collection.query.fetch_objects(
                filters=Filter.by_property("user_id").equal(data.user_id),
                return_properties=["user_id", "vector", "timestamp"],
                limit=5  # Get last 5 registrations for better comparison
            )

            if not response.objects:
                raise HTTPException(status_code=401, detail=f"No registered face found for user_id: {data.user_id}")

            # Compare with stored embeddings using cosine similarity
            similarities = []
            timestamps = []
            for stored_face in response.objects:
                stored_embedding = np.array(stored_face.properties["vector"])
                # Calculate cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(input_embedding, stored_embedding)
                similarities.append(similarity)
                timestamps.append(stored_face.properties["timestamp"])

            # Get the highest similarity score
            best_similarity = max(similarities)
            similarity_percentage = best_similarity * 100
            
            # Very strict threshold - 96% similarity required
            if best_similarity >= 0.96:  # Slightly reduced from 97% for better balance
                # Get the timestamp of the best matching face
                best_match_idx = similarities.index(best_similarity)
                match_time = timestamps[best_match_idx]
                
                return {
                    "message": "Authentication successful",
                    "user_id": data.user_id,
                    "confidence": similarity_percentage,
                    "match_time": match_time
                }
            
            # If similarity is close but not quite there, give a helpful message
            elif best_similarity >= 0.90:
                raise HTTPException(
                    status_code=401,
                    detail=f"Authentication failed - Face similarity ({similarity_percentage:.1f}%) is close but below required threshold (96%). Please ensure good lighting and face alignment."
                )
            else:
                raise HTTPException(
                    status_code=401, 
                    detail=f"Authentication failed - Face similarity ({similarity_percentage:.1f}%) significantly below required threshold (96%). This may not be the same person."
                )
            
        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Error querying Weaviate: {e}")
            raise HTTPException(status_code=500, detail=f"Error during authentication: {str(e)}")

    except Exception as e:
        print(f"Error in authenticate_face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
