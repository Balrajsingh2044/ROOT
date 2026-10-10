import torch
import torchvision
import cv2
import numpy as np
from typing import Tuple

class VideoIntegrityChecker:
    def __init__(self):
        """
        Initialize the pre-trained 3D CNN (ResNet 3D 18).
        We use torchvision's built-in model pre-trained on Kinetics-400.
        """
        print("Loading 3D CNN Model for Integrity Check...")
        # Load pre-trained weights
        self.model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        self.model.eval() # Set to evaluation mode
        
        # Move to GPU if available, else CPU (or MPS for Mac)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}!")

    def extract_frames(self, video_path: str, num_frames: int = 16) -> np.ndarray:
        """
        Extracts 'num_frames' evenly spaced frames from a video file.
        Returns a numpy array of shape (num_frames, height, width, channels)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("Could not read video file.")
            
        # Calculate indices to extract frames evenly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Resize to 112x112 (standard input size for R3D_18)
                frame = cv2.resize(frame, (112, 112))
                # Convert BGR (OpenCV) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
        cap.release()
        return np.array(frames)

    def check_looping_heuristic(self, frames: np.ndarray) -> float:
        """
        MVP Hack: Calculate frame-to-frame variance. 
        Deepfakes/Looped videos often have suspiciously low variance between frames.
        Returns a score from 0.0 (highly suspicious/looped) to 1.0 (natural motion).
        """
        if len(frames) < 2:
            return 1.0
            
        # Calculate mean absolute difference between consecutive frames
        diffs = np.abs(np.diff(frames, axis=0))
        mean_diff = np.mean(diffs)
        
        # Normalize: typical natural video has mean_diff > 10. 
        # We map this to a 0-1 score. (Tweak these thresholds based on your mock videos)
        score = min(1.0, mean_diff / 15.0)
        return float(score)

    def predict(self, video_path: str) -> dict:
        """
        Main inference function. Returns integrity score and verdict.
        """
        try:
            # 1. Extract frames
            frames = self.extract_frames(video_path, num_frames=16)
            
            # 2. Run Heuristic Check (Fast, catches looped fakes)
            heuristic_score = self.check_looping_heuristic(frames)
            
            # 3. Run 3D CNN (Optional for MVP: we can use it to boost confidence)
            # Preprocess for PyTorch: (Frames, Height, Width, Channels) -> (Batch, Channels, Frames, Height, Width)
            tensor_frames = torch.from_numpy(frames).float() / 255.0
            tensor_frames = tensor_frames.permute(3, 0, 1, 2).unsqueeze(0) # Shape: (1, 3, 16, 112, 112)
            tensor_frames = tensor_frames.to(self.device)
            
            with torch.no_grad():
                # Get features from the CNN
                features = self.model(tensor_frames)
                # For MVP, we combine CNN feature magnitude with heuristic score
                # A real deepfake detector would have a classification head, but this is a robust MVP proxy
                cnn_confidence = torch.sigmoid(features.mean()).item()
                
            # Final combined score (weight heuristic heavily for loop detection)
            final_score = (heuristic_score * 0.7) + (cnn_confidence * 0.3)
            
            is_authentic = final_score > 0.65 # Threshold for "Real"
            
            return {
                "is_authentic": is_authentic,
                "confidence_score": round(final_score, 4),
                "message": "Video passed integrity check." if is_authentic else "Warning: Video shows signs of looping or manipulation."
            }
            
        except Exception as e:
            return {
                "is_authentic": False,
                "confidence_score": 0.0,
                "message": f"Error processing video: {str(e)}"
            }

# Global instance to load model only once (saves memory!)
integrity_checker = VideoIntegrityChecker()