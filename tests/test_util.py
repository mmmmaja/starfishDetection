import torch
from src.starfish.model import NMS

def test_NMS():
    boxes = torch.tensor([
    [10, 10, 50, 50],  # Box 1
    [12, 12, 48, 48],  # Box 2 (overlaps significantly with Box 1)
    [55, 55, 95, 95],  # Box 3 (no overlap with others)
    [60, 60, 100, 100],  # Box 4 (overlaps partially with Box 3)
    [200, 200, 250, 250],  # Box 5 (completely separate)
    ])

    scores = torch.tensor([0.9, 0.85, 0.6, 0.75, 0.5])
    iou_threshold = 0.5
    expected_boxes = torch.tensor([
        [10, 10, 50, 50],  # Keep Box 1 (highest score in overlap group)
        [60, 60, 100, 100],  # Keep Box 4 (highest score in overlap group)
        [200, 200, 250, 250],  # Keep Box 5 (no overlap)
    ])
    expected_scores = torch.tensor([0.9, 0.75, 0.5])

    kept_scores, kept_boxes = NMS(scores, boxes, iou_threshold)

    assert torch.allclose(kept_boxes, expected_boxes), f"Expected boxes: {expected_boxes}, Got: {kept_boxes}"
    assert torch.allclose(kept_scores, expected_scores), f"Expected scores: {expected_scores}, Got: {kept_scores}"
