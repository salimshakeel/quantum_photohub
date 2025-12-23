"""
Align Images - Registration before HDR

Responsibilities (later):
- Normalize inputs (resize, RGB, float [0,1])
- Implement alignment methods:
  - MTB (Median Threshold Bitmap) for bracketed exposure sets
  - Feature-based (e.g., ORB/SIFT) with homography for robust cases
- Output aligned images for reuse by all HDR techniques
"""


