# Report - Player Re-Identification

## Approach
- Used YOLOv11 model (`best.pt`) for player detection.
- Assigned tracking IDs using IoU and feature matching.
- Re-identified players after leaving frame using saved feature embeddings.

## Challenges
- Occlusion and reentry of players
- Varying appearance due to camera angle and lighting

## Future Work
- Improve feature extraction using CLIP or ReID-specific models
