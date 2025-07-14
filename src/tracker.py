import numpy as np
import torch
from torchvision import models, transforms
import cv2

class ReIDTracker:
    def __init__(self, sim_threshold=0.8, memory_size=5):
        self.next_id = 0
        self.tracks = {}
        self.embeddings = {}  # Stores list of embeddings for each ID
        self.sim_threshold = sim_threshold
        self.memory_size = memory_size

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_embedding(self, crop):
        with torch.no_grad():
            img = self.transform(crop).unsqueeze(0)
            embedding = self.resnet(img).squeeze().numpy()
            return embedding / np.linalg.norm(embedding)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update(self, frame, detections):
        updated_tracks = {}
        used_ids = set()

        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = self.get_embedding(crop)

            best_match_id = None
            best_score = -1

            for track_id, emb_list in self.embeddings.items():
                if track_id in used_ids:
                    continue
                similarities = [self.cosine_similarity(embedding, emb) for emb in emb_list]
                avg_score = max(similarities) if similarities else 0
                if avg_score > self.sim_threshold and avg_score > best_score:
                    best_score = avg_score
                    best_match_id = track_id

            if best_match_id is not None:
                updated_tracks[best_match_id] = box
                self.embeddings[best_match_id].append(embedding)
                if len(self.embeddings[best_match_id]) > self.memory_size:
                    self.embeddings[best_match_id].pop(0)
                used_ids.add(best_match_id)
            else:
                updated_tracks[self.next_id] = box
                self.embeddings[self.next_id] = [embedding]
                used_ids.add(self.next_id)
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks
