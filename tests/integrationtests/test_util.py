import numpy
# import torch
# from src.starfish.model import NMS, get_AP

# def test_perfect_predictions():
#     # One-to-one perfect match
#     scores = torch.tensor([1.0, 1.0])
#     pred_boxes = torch.tensor([[10, 10, 20, 20], [50, 50, 60, 60]])
#     gt_boxes = torch.tensor([[10, 10, 20, 20], [50, 50, 60, 60]])
#     iou_threshold = 1.0

#     ap = get_AP(scores, pred_boxes, gt_boxes, iou_threshold)
#     assert 0.99 < ap


# def test_partial_matches():
#     # Some predictions match, others don't
#     scores = torch.tensor([0.9, 0.8, 0.7])
#     pred_boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]])
#     gt_boxes = torch.tensor([[10, 10, 20, 20], [50, 50, 60, 60]])
#     iou_threshold = 0.5

#     ap = get_AP(scores, pred_boxes, gt_boxes, iou_threshold)
#     assert 0.3332 < ap and ap < 0.3334


# def test_high_iou_threshold():
#     # High IoU threshold causing no matches
#     scores = torch.tensor([0.9, 0.8])
#     pred_boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
#     gt_boxes = torch.tensor([[10, 10, 21, 21], [30, 30, 41, 41]])
#     iou_threshold = 0.95

#     ap = get_AP(scores, pred_boxes, gt_boxes, iou_threshold)
#     assert ap < 1e-3  # Allows for a small margin of error


# def test_low_iou_threshold():
#     # Low IoU threshold causing more matches
#     scores = torch.tensor([0.9, 0.8])
#     pred_boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
#     gt_boxes = torch.tensor([[10, 10, 21, 21], [30, 30, 41, 41]])
#     iou_threshold = 0.1

#     ap = get_AP(scores, pred_boxes, gt_boxes, iou_threshold)
#     assert True


# def test_NMS():
#     boxes = torch.tensor([
#     [10, 10, 50, 50],  # Box 1
#     [12, 12, 48, 48],  # Box 2 (overlaps significantly with Box 1)
#     [55, 55, 95, 95],  # Box 3 (no overlap with others)
#     [60, 60, 100, 100],  # Box 4 (overlaps partially with Box 3)
#     [200, 200, 250, 250],  # Box 5 (completely separate)
#     ])

#     scores = torch.tensor([0.9, 0.85, 0.6, 0.75, 0.5])
#     iou_threshold = 0.5
#     expected_boxes = torch.tensor([
#         [10, 10, 50, 50],  # Keep Box 1 (highest score in overlap group)
#         [60, 60, 100, 100],  # Keep Box 4 (highest score in overlap group)
#         [200, 200, 250, 250],  # Keep Box 5 (no overlap)
#     ])
#     expected_scores = torch.tensor([0.9, 0.75, 0.5])

#     kept_scores, kept_boxes = NMS(scores, boxes, iou_threshold)

#     assert torch.allclose(kept_boxes, expected_boxes), f"Expected boxes: {expected_boxes}, Got: {kept_boxes}"
#     assert torch.allclose(kept_scores, expected_scores), f"Expected scores: {expected_scores}, Got: {kept_scores}"
