"""
Advanced Visual Search Tools for V2
Includes dynamic branching, spatial reasoning, and complexity assessment
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple, Dict
import math


class DynamicGridSplitTool:
    """
    Enhanced grid splitting with dynamic branching factor selection
    """

    @staticmethod
    def assess_complexity(image_pil: Image.Image) -> str:
        """
        Assess image complexity to determine optimal split factor

        Returns: 'low', 'medium', 'high'
        """
        # Convert to numpy for analysis
        img_array = np.array(image_pil.convert('RGB'))

        # 1. Calculate edge density (indicates detail level)
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        edge_density = (np.sum(edges_h > 20) + np.sum(edges_v > 20)) / gray.size

        # 2. Calculate color variance (indicates visual diversity)
        color_variance = np.std(img_array)

        # 3. Calculate spatial entropy
        hist, _ = np.histogram(gray.flatten(), bins=32, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        spatial_entropy = -np.sum(hist * np.log2(hist))

        # Decision logic
        complexity_score = (
            0.4 * (edge_density / 0.3) +  # Normalized edge density
            0.3 * (color_variance / 80) +  # Normalized color variance
            0.3 * (spatial_entropy / 5)    # Normalized entropy
        )

        if complexity_score < 0.4:
            return 'low'
        elif complexity_score < 0.7:
            return 'medium'
        else:
            return 'high'

    @staticmethod
    def get_split_factor(complexity: str, depth: int, max_depth: int) -> int:
        """
        Determine split factor based on complexity and depth

        Args:
            complexity: 'low', 'medium', 'high'
            depth: current search depth
            max_depth: maximum allowed depth

        Returns: split factor (2, 4, or 9)
        """
        # Early depths: prefer more branching for exploration
        # Later depths: reduce branching to avoid over-fragmentation

        if depth >= max_depth - 1:
            # Near max depth, use minimal splitting
            return 2

        if complexity == 'low':
            return 2 if depth > 0 else 4
        elif complexity == 'medium':
            return 4
        else:  # high complexity
            return 9 if depth < 2 else 4

    @staticmethod
    def split_box(bbox: Tuple[int, int, int, int],
                  split_factor: int,
                  overlap_ratio: float = 0.1) -> List[Tuple[int, int, int, int]]:
        """
        Split a bounding box into sub-regions with optional overlap

        Args:
            bbox: (x, y, w, h)
            split_factor: 2, 4, or 9
            overlap_ratio: overlap between adjacent regions (0.0 - 0.3)

        Returns: List of sub-boxes [(x, y, w, h), ...]
        """
        x, y, w, h = bbox

        # Determine grid layout
        if split_factor == 2:
            # Adaptive 2-split based on aspect ratio
            ratio = h / w if w > 0 else 1.0
            if ratio > 1.5:  # Tall image
                rows, cols = 2, 1
            elif ratio < 0.67:  # Wide image
                rows, cols = 1, 2
            else:  # Square-ish, prefer horizontal split
                rows, cols = 1, 2

        elif split_factor == 4:
            # Standard 2x2 grid
            rows, cols = 2, 2

        elif split_factor == 9:
            # Dense 3x3 grid for high complexity
            rows, cols = 3, 3

        else:
            raise ValueError(f"Unsupported split_factor: {split_factor}")

        # Calculate stride with overlap
        w_stride = w / cols
        h_stride = h / rows

        # Expand each cell by overlap_ratio
        w_overlap = w_stride * overlap_ratio
        h_overlap = h_stride * overlap_ratio

        sub_boxes = []
        for r in range(rows):
            for c in range(cols):
                # Base position
                cx = x + c * w_stride
                cy = y + r * h_stride

                # Expand with overlap
                cx = max(x, cx - w_overlap)
                cy = max(y, cy - h_overlap)

                # Calculate width and height
                cw = w_stride + 2 * w_overlap
                ch = h_stride + 2 * h_overlap

                # Clip to parent bounds
                if cx + cw > x + w:
                    cw = x + w - cx
                if cy + ch > y + h:
                    ch = y + h - cy

                sub_boxes.append((int(cx), int(cy), int(cw), int(ch)))

        return sub_boxes


class SpatialRelationshipTool:
    """
    Tools for reasoning about spatial relationships between objects
    """

    @staticmethod
    def get_box_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate center point of a box"""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)

    @staticmethod
    def check_left_right_relation(box1: Tuple[int, int, int, int],
                                   box2: Tuple[int, int, int, int],
                                   threshold: float = 0.3) -> str:
        """
        Check if box1 is to the left or right of box2

        Returns: 'left', 'right', or 'overlapping'
        """
        cx1, cy1 = SpatialRelationshipTool.get_box_center(box1)
        cx2, cy2 = SpatialRelationshipTool.get_box_center(box2)

        # Calculate relative position
        diff = cx1 - cx2
        avg_width = (box1[2] + box2[2]) / 2

        # Normalize by width
        normalized_diff = diff / avg_width if avg_width > 0 else 0

        if normalized_diff < -threshold:
            return 'left'
        elif normalized_diff > threshold:
            return 'right'
        else:
            return 'overlapping'

    @staticmethod
    def check_above_below_relation(box1: Tuple[int, int, int, int],
                                    box2: Tuple[int, int, int, int],
                                    threshold: float = 0.3) -> str:
        """
        Check if box1 is above or below box2

        Returns: 'above', 'below', or 'same_level'
        """
        cx1, cy1 = SpatialRelationshipTool.get_box_center(box1)
        cx2, cy2 = SpatialRelationshipTool.get_box_center(box2)

        diff = cy1 - cy2
        avg_height = (box1[3] + box2[3]) / 2

        normalized_diff = diff / avg_height if avg_height > 0 else 0

        if normalized_diff < -threshold:
            return 'above'
        elif normalized_diff > threshold:
            return 'below'
        else:
            return 'same_level'

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int],
                    boxB: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two boxes"""
        x1_A, y1_A, w_A, h_A = boxA
        x2_A, y2_A = x1_A + w_A, y1_A + h_A

        x1_B, y1_B, w_B, h_B = boxB
        x2_B, y2_B = x1_B + w_B, y1_B + h_B

        xA = max(x1_A, x1_B)
        yA = max(y1_A, y1_B)
        xB = min(x2_A, x2_B)
        yB = min(y2_A, y2_B)

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        if interArea == 0:
            return 0.0

        boxAArea = w_A * h_A
        boxBArea = w_B * h_B
        unionArea = boxAArea + boxBArea - interArea

        return interArea / float(unionArea) if unionArea > 0 else 0.0

    @staticmethod
    def verify_relation(box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int],
                       relation_type: str) -> bool:
        """
        Verify if two boxes satisfy a given spatial relation

        Args:
            box1, box2: bounding boxes
            relation_type: 'left-right', 'above-below', 'containment', etc.

        Returns: True if relation is satisfied
        """
        if relation_type == 'left-right':
            rel = SpatialRelationshipTool.check_left_right_relation(box1, box2)
            return rel in ['left', 'right']

        elif relation_type == 'above-below':
            rel = SpatialRelationshipTool.check_above_below_relation(box1, box2)
            return rel in ['above', 'below']

        elif relation_type == 'containment':
            iou = SpatialRelationshipTool.compute_iou(box1, box2)
            # box1 inside box2 if box1 is much smaller and high overlap
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            return iou > 0.5 and area1 < area2 * 0.8

        elif relation_type == 'front-back':
            # Approximate front-back by checking if one is below the other
            # (assumes camera perspective where lower = closer)
            rel = SpatialRelationshipTool.check_above_below_relation(box1, box2)
            return rel in ['above', 'below']

        else:
            # Unknown relation type, return True by default
            return True


class UncertaintyEstimator:
    """
    Estimate uncertainty and confidence in search decisions
    """

    @staticmethod
    def compute_entropy(probs: List[float]) -> float:
        """Calculate Shannon entropy of a probability distribution"""
        entropy = 0.0
        for p in probs:
            if p > 1e-9:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def compute_variance(values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return variance

    @staticmethod
    def compute_confidence_interval(scores: List[float],
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for a list of scores

        Returns: (lower_bound, upper_bound)
        """
        if not scores:
            return (0.0, 0.0)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)

        # Z-score for confidence level (assuming normal distribution)
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%

        margin = z_score * (std_score / math.sqrt(n))

        return (mean_score - margin, mean_score + margin)

    @staticmethod
    def estimate_uncertainty_score(prior: float,
                                   posterior: float,
                                   entropy: float,
                                   variance: float) -> float:
        """
        Compute overall uncertainty score combining multiple factors

        Returns: uncertainty in range [0, 1], where 0 = certain, 1 = uncertain
        """
        # Disagreement between prior and posterior
        disagreement = abs(prior - posterior)

        # Normalize entropy (max entropy for 4-way split is log2(4) = 2)
        normalized_entropy = min(entropy / 2.0, 1.0)

        # Normalize variance (assuming max variance around 0.25)
        normalized_variance = min(variance / 0.25, 1.0)

        # Weighted combination
        uncertainty = (
            0.4 * disagreement +
            0.4 * normalized_entropy +
            0.2 * normalized_variance
        )

        return min(uncertainty, 1.0)


class VisualizationTool:
    """
    Enhanced visualization for debugging and result presentation
    """

    @staticmethod
    def draw_search_tree(image_pil: Image.Image,
                        nodes: List,
                        output_path: str):
        """
        Visualize the beam search tree on the image

        Args:
            image_pil: original image
            nodes: list of search nodes
            output_path: where to save the visualization
        """
        img_draw = image_pil.copy()
        draw = ImageDraw.Draw(img_draw)

        # Color map based on depth
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

        for node in nodes:
            x, y, w, h = node.box_xywh
            depth = node.depth
            color = colors[depth % len(colors)]

            # Line width based on score
            width = max(2, int(node.final_score * 10))

            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=width)

            # Draw node ID
            draw.text((x+5, y+5), f"N{node.id[:4]}", fill=color)

        img_draw.save(output_path)

    @staticmethod
    def create_heatmap(image_pil: Image.Image,
                      boxes_with_scores: List[Tuple[Tuple, float]],
                      output_path: str):
        """
        Create a heatmap visualization of search attention

        Args:
            image_pil: original image
            boxes_with_scores: list of (box, score) tuples
            output_path: where to save
        """
        # Create a blank heatmap
        w, h = image_pil.size
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Accumulate scores in the heatmap
        for (x, y, bw, bh), score in boxes_with_scores:
            x, y, bw, bh = int(x), int(y), int(bw), int(bh)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            heatmap[y:y2, x:x2] += score

        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

        # Convert to RGB heatmap (using jet colormap approximation)
        heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = heatmap  # Red channel
        heatmap_rgb[:, :, 2] = 255 - heatmap  # Blue channel

        # Blend with original image
        heatmap_img = Image.fromarray(heatmap_rgb)
        blended = Image.blend(image_pil.convert('RGB'), heatmap_img, alpha=0.5)
        blended.save(output_path)


class CoarseGroundingHelper:
    """
    Helper for coarse-grained initial grounding
    """

    @staticmethod
    def divide_into_9_regions(image_size: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Divide image into 9 regions (3x3 grid)

        Returns: dict mapping region name to (x, y, w, h)
        """
        w, h = image_size
        w_third = w // 3
        h_third = h // 3

        regions = {
            'TL': (0, 0, w_third, h_third),
            'TC': (w_third, 0, w_third, h_third),
            'TR': (2*w_third, 0, w - 2*w_third, h_third),

            'ML': (0, h_third, w_third, h_third),
            'MC': (w_third, h_third, w_third, h_third),
            'MR': (2*w_third, h_third, w - 2*w_third, h_third),

            'BL': (0, 2*h_third, w_third, h - 2*h_third),
            'BC': (w_third, 2*h_third, w_third, h - 2*h_third),
            'BR': (2*w_third, 2*h_third, w - 2*w_third, h - 2*h_third)
        }

        return regions

    @staticmethod
    def filter_regions_by_priority(all_regions: Dict[str, Tuple],
                                   priority_list: List[str],
                                   top_k: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Filter regions based on priority ranking

        Args:
            all_regions: dict from divide_into_9_regions
            priority_list: ordered list of region names
            top_k: how many regions to return

        Returns: list of boxes
        """
        boxes = []
        for region_name in priority_list[:top_k]:
            if region_name in all_regions:
                boxes.append(all_regions[region_name])
        return boxes
