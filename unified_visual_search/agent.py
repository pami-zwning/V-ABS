"""
Enhanced Visual Search Agent V2 with Hierarchical Beam Search
Implements multi-target cooperative search with uncertainty-aware reasoning
"""

import os
import json
import uuid
import re
import sys
import math
import time
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from prompt import VisualSearchPrompts
from tool import (
    DynamicGridSplitTool,
    SpatialRelationshipTool,
    UncertaintyEstimator,
    VisualizationTool
)

try:
    from utils_llm import chat_vlm, vlm_tracker
except ImportError:
    print("❌ Error: 'utils_llm' module not found.")
    def chat_vlm(*args, **kwargs):
        return "{}", []
    vlm_tracker = None


@dataclass
class SearchNode:
    """
    Node in the beam search tree
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    box_xywh: Tuple[int, int, int, int] = (0, 0, 0, 0)
    depth: int = 0
    parent: Optional['SearchNode'] = None
    action_idx: Optional[int] = None
    target_name: str = ""

    # Scoring components
    prior_score: float = 0.0
    posterior_score: float = 0.0
    heuristic_score: float = 0.0
    uncertainty_score: float = 0.0
    final_score: float = 0.0

    # Additional metadata
    entropy: float = 0.0
    variance: float = 0.0
    split_factor: int = 4
    adaptive_weights: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def __lt__(self, other):
        """For heap operations (max-heap, so reverse comparison)"""
        return self.final_score > other.final_score


class VisualSearchAgent:
    """
    Enhanced Visual Search Agent with Hierarchical Beam Search
    """

    def __init__(self,
                 output_dir: str,
                 max_depth: int = 4,
                 beam_width: int = 3,
                 model_name: str = "gpt4o",
                 num_threads: int = 8,
                 verbose: bool = True,
                 token_set: str = "yes_no",
                 mu_threshold: float = 0.5,
                 no_heuristic: bool = False,
                 entropy_skip_threshold: float = -1.0):
        """
        Args:
            output_dir: directory for saving results
            max_depth: maximum search depth
            beam_width: number of candidates to keep per depth (K in beam search)
            model_name: VLM model to use
            num_threads: parallelism for VLM queries
            verbose: whether to print detailed logs
            token_set: logit token set - 'yes_no', 'true_false', 'correct_incorrect', or 'all' (E4)
            mu_threshold: entropy threshold for adaptive weighting (E5)
            no_heuristic: if True, set heuristic weight to 0 (E7)
            entropy_skip_threshold: skip observer when entropy < this value; -1 = disabled (E3)
        """
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.model_name = model_name
        self.num_threads = num_threads
        self.verbose = verbose

        # Rebuttal experiment parameters
        self.token_set = token_set
        self.mu_threshold = mu_threshold
        self.no_heuristic = no_heuristic
        self.entropy_skip_threshold = entropy_skip_threshold

        # Thresholds
        self.high_confidence_threshold = 0.85
        self.low_confidence_threshold = 0.35
        self.pruning_threshold = 0.20

        # Score distribution collector (E1, E2)
        self.score_distributions = []

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        self.log_dir = os.path.join(output_dir, "logs")
        self.image_save_dir = os.path.join(output_dir, "images")
        self.temp_dir = os.path.join(output_dir, "temp_images")

        for d in [self.log_dir, self.image_save_dir, self.temp_dir]:
            os.makedirs(d, exist_ok=True)

        # State
        self.root_image_pil = None
        self.search_history = []

    # ==================== Helper Methods ====================

    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, 'ℹ️')} {message}")

    def _chat_wrapper(self, prompt_text: str, images: List[Image.Image], logit: bool = False) -> Any:
        """
        Wrapper for VLM chat with image handling
        """
        temp_paths = []
        try:
            img_tag_str = ""
            for i, img in enumerate(images):
                temp_name = f"{uuid.uuid4().hex}_{i}.jpg"
                temp_path = os.path.join(self.temp_dir, temp_name)
                img.save(temp_path, format="JPEG")
                temp_paths.append(temp_path)
                img_tag_str += f"<img src='{temp_path}'> "

            full_prompt = img_tag_str + prompt_text
            ret = chat_vlm(full_prompt, self.model_name, logit=logit, temperature=0.7)

            if logit:
                return ret[2] if isinstance(ret, tuple) and len(ret) >= 3 else {}
            else:
                return ret[0] if isinstance(ret, tuple) else ret

        except Exception as e:
            self._log(f"VLM chat error: {e}", "ERROR")
            return {} if logit else ""
        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass

    def _extract_json_from_text(self, text: str) -> Dict:
        """Parse JSON from VLM response"""
        if not text:
            return {"status": False, "content": {}}

        # Remove markdown code blocks
        text_clean = re.sub(r"```json|```", "", text).strip()

        # Try to parse as JSON
        decoder = json.JSONDecoder()
        pos = 0

        while True:
            match_index = text_clean.find('{', pos)
            if match_index == -1:
                break
            try:
                obj, end_offset = decoder.raw_decode(text_clean[match_index:])
                if isinstance(obj, dict):
                    return {"status": True, "content": obj}
                pos = match_index + end_offset
            except json.JSONDecodeError:
                pos = match_index + 1

        # Fallback
        return {"status": False, "content": {}}

    def _softmax(self, logits: List[float]) -> List[float]:
        """Softmax with numerical stability"""
        if not logits:
            return []
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps] if sum_exps > 0 else [1.0 / len(logits)] * len(logits)

    def _extract_yes_logit(self, prob_map: Dict[str, float]) -> float:
        """Extract positive-direction probability from logit map (E4: configurable token set)"""
        if not prob_map:
            return -10.0

        TOKEN_SETS = {
            "yes_no": ['Yes', ' Yes', 'yes', ' yes', 'YES', 'Y'],
            "true_false": ['True', ' True', 'true', ' true', 'TRUE', 'T'],
            "correct_incorrect": ['Correct', ' Correct', 'correct', ' correct', 'CORRECT'],
            "all": ['Yes', ' Yes', 'yes', ' yes', 'YES', 'Y',
                     'True', ' True', 'true', ' true', 'TRUE', 'T',
                     'Correct', ' Correct', 'correct', ' correct', 'CORRECT'],
        }
        tokens = TOKEN_SETS.get(self.token_set, TOKEN_SETS["yes_no"])

        best_val = -10.0
        for k in tokens:
            if k in prob_map:
                best_val = max(best_val, prob_map[k])
        return best_val

    # ==================== Phase 1: Target Extraction ====================

    def _extract_targets_with_relations(self, question: str) -> Dict:
        """
        Extract targets and spatial relations from question
        Returns: {
            'targets': List[str],
            'relation': str,  # 'spatial', 'containment', 'none'
            'relation_pairs': List[Tuple[str, str]]
        }
        """
        prompt = VisualSearchPrompts.EXTRACT_TARGETS_WITH_RELATIONS.format(question=question)
        response = self._chat_wrapper(prompt, [self.root_image_pil])

        parsed = self._extract_json_from_text(response)

        if parsed['status']:
            content = parsed['content']
            targets = content.get('targets', [])
            relation = content.get('relation', 'none')
            relation_pairs = content.get('relation_pairs', [])
            relation_type = content.get('relation_type', '')

            # Filter invalid targets
            forbidden_words = {
                'left', 'right', 'top', 'bottom', 'center', 'middle',
                'object', 'image', 'picture', 'option', 'choice',
                'a', 'b', 'c', 'd'
            }

            valid_targets = []
            for t in targets:
                t_str = str(t).strip()
                if t_str.lower() not in forbidden_words and len(t_str) > 1:
                    valid_targets.append(t_str)

            return {
                'targets': valid_targets if valid_targets else ['object'],
                'relation': relation,
                'relation_type': relation_type,
                'relation_pairs': relation_pairs
            }

        else:
            # Fallback: simple regex extraction
            self._log("Failed to parse targets JSON, using fallback", "WARNING")
            return {
                'targets': ['object'],
                'relation': 'none',
                'relation_type': '',
                'relation_pairs': []
            }

    # ==================== Phase 2: Hierarchical Beam Search ====================

    def _expand_node(self, node: SearchNode, target: str, question: str) -> List[SearchNode]:
        """
        Expand a node by splitting and evaluating children
        """
        x, y, w, h = node.box_xywh
        parent_img = self.root_image_pil.crop((x, y, x+w, y+h))

        # 1. Determine split factor based on complexity
        split_factor = 4

        self._log(f"Expanding node {node.id[:6]} (depth={node.depth}, split={split_factor})")

        # 2. Split the region
        sub_boxes = DynamicGridSplitTool.split_box(node.box_xywh, split_factor, overlap_ratio=0.1)

        # 3. Compute prior scores (parallel)
        priors = self._batch_compute_priors(parent_img, sub_boxes, target, question, split_factor)

        # Check if prior is confident enough to skip observer (E3)
        prior_entropy = UncertaintyEstimator.compute_entropy(priors)
        skip_observer = (self.entropy_skip_threshold > 0 and prior_entropy < self.entropy_skip_threshold)

        if skip_observer:
            self._log(f"Prior entropy={prior_entropy:.3f} < threshold={self.entropy_skip_threshold}. Skipping observer.")

        # 4. Prepare children
        children = []
        child_images = []

        for i, box in enumerate(sub_boxes):
            child = SearchNode(
                box_xywh=box,
                depth=node.depth + 1,
                parent=node,
                action_idx=i,
                target_name=target
            )
            child.prior_score = priors[i]
            child.split_factor = split_factor
            children.append(child)

            # Crop child image
            bx, by, bw, bh = box
            child_img = self.root_image_pil.crop((bx, by, bx+bw, by+bh))
            child_images.append(child_img)

        # 5. Compute posterior scores (skip if prior is confident)
        if skip_observer:
            posteriors = priors  # Use prior as posterior when skipping
            entropy = prior_entropy
        else:
            posteriors, entropy = self._batch_compute_posteriors(child_images, target, question)

        # 6. Compute heuristic and final scores
        for i, child in enumerate(children):
            child.posterior_score = posteriors[i]
            child.entropy = entropy
            child.heuristic_score = self._compute_heuristic(child)

            # Compute variance (measure disagreement across children)
            all_posteriors = posteriors
            child.variance = UncertaintyEstimator.compute_variance(all_posteriors)

            # Uncertainty score
            child.uncertainty_score = UncertaintyEstimator.estimate_uncertainty_score(
                child.prior_score,
                child.posterior_score,
                child.entropy,
                child.variance
            )

            # Adaptive weighted fusion
            child.final_score, child.adaptive_weights = self._compute_final_score(child)

        return children

    def _batch_compute_priors(self,
                             parent_img: Image.Image,
                             sub_boxes: List[Tuple],
                             target: str,
                             question: str,
                             split_factor: int) -> List[float]:
        """
        Compute prior scores for all sub-regions in parallel
        """
        # Direction labels
        if split_factor == 2:
            directions = ["First half", "Second half"]
        elif split_factor == 4:
            directions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        else:  # 9
            directions = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]

        logits = [-10.0] * len(sub_boxes)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_idx = {}
            for i, direction in enumerate(directions[:len(sub_boxes)]):
                prompt = VisualSearchPrompts.PRIOR_SCORE_WITH_CONTEXT.format(
                    target=target,
                    question=question,
                    direction=direction
                )
                future = executor.submit(self._chat_wrapper, prompt, [parent_img], True)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    val = self._extract_yes_logit(res)
                    if val != -10.0:
                        logits[idx] = val
                except Exception as e:
                    self._log(f"Prior computation error for idx {idx}: {e}", "WARNING")

        probs = self._softmax(logits)
        return probs

    def _batch_compute_posteriors(self,
                                  child_images: List[Image.Image],
                                  target: str,
                                  question: str) -> Tuple[List[float], float]:
        """
        Compute posterior scores for child regions in parallel

        Returns: (probabilities, entropy)
        """
        logits = [-10.0] * len(child_images)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_idx = {}
            for i, img in enumerate(child_images):
                prompt = VisualSearchPrompts.POSTERIOR_SCORE_CONTRASTIVE.format(
                    question=question,
                    target=target
                )
                future = executor.submit(self._chat_wrapper, prompt, [img], True)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    val = self._extract_yes_logit(res)
                    if val != -10.0:
                        logits[idx] = val
                except Exception as e:
                    self._log(f"Posterior computation error for idx {idx}: {e}", "WARNING")

        probs = self._softmax(logits)
        entropy = UncertaintyEstimator.compute_entropy(probs)

        return probs, entropy

    def _compute_heuristic(self, node: SearchNode) -> float:
        """
        Compute heuristic score based on depth and box size
        """
        # Depth penalty: deeper nodes get slightly lower heuristic
        depth_factor = 1.0 / (1.0 + 0.1 * node.depth)

        # Size factor: prefer larger regions early, smaller regions later
        img_w, img_h = self.root_image_pil.size
        total_area = img_w * img_h
        node_area = node.box_xywh[2] * node.box_xywh[3]
        size_ratio = node_area / total_area if total_area > 0 else 0.5

        # Early depths: prefer larger regions
        # Later depths: size matters less
        if node.depth < 2:
            size_factor = size_ratio
        else:
            size_factor = 0.5

        heuristic = 0.7 * depth_factor + 0.3 * size_factor
        return heuristic

    def _compute_final_score(self, node: SearchNode) -> Tuple[float, Dict[str, float]]:
        """
        Adaptive weighted fusion using entropy-based sigmoid weighting.
        Implements: w_p = 1/(1+e^{beta*(H_t - mu)}), w_o = 1 - w_p
        From paper Eq.(7).

        Returns: (final_score, weights_dict)
        """
        beta = 2.0  # Sensitivity parameter

        # Entropy-based sigmoid weighting (paper Eq.7)
        H_t = node.entropy
        w_prior = 1.0 / (1.0 + math.exp(beta * (H_t - self.mu_threshold)))
        w_posterior = 1.0 - w_prior
        w_heuristic = 0.0 if self.no_heuristic else 0.15

        # Normalize so they sum to 1
        total = w_prior + w_posterior + w_heuristic
        if total > 0:
            w_prior /= total
            w_posterior /= total
            w_heuristic /= total

        # Compute final score
        final = (w_prior * node.prior_score +
                w_posterior * node.posterior_score +
                w_heuristic * node.heuristic_score)

        weights = {
            'prior': w_prior,
            'posterior': w_posterior,
            'heuristic': w_heuristic
        }

        # E1/E2: Record score distribution for analysis
        self.score_distributions.append({
            'depth': node.depth,
            'prior_score': node.prior_score,
            'posterior_score': node.posterior_score,
            'heuristic_score': node.heuristic_score,
            'entropy': node.entropy,
            'variance': node.variance,
            'uncertainty': node.uncertainty_score,
            'final_score': final,
            'weights': weights,
        })

        return final, weights

    def _beam_search_single_target(self, target: str, question: str, q_id: str) -> List[SearchNode]:
        """
        Perform beam search for a single target.
        Starts from the full image, splits into 4 sub-regions, and iterates.

        Returns: List of final candidate nodes (beam)
        """
        self._log(f"Starting beam search for target: '{target}'", "INFO")

        # Initialize: single root node covering the full image
        w, h = self.root_image_pil.size
        root_node = SearchNode(
            box_xywh=(0, 0, w, h),
            depth=0,
            target_name=target
        )
        root_node.prior_score = 1.0
        root_node.posterior_score = 0.5
        root_node.heuristic_score = 1.0
        root_node.final_score, root_node.adaptive_weights = self._compute_final_score(root_node)

        beam = [root_node]

        # Beam search loop
        for depth in range(self.max_depth):
            self._log(f"Depth {depth} -> {depth+1}: Beam size = {len(beam)}")

            # Check stopping criteria
            best_node = max(beam, key=lambda n: n.posterior_score)
            if best_node.posterior_score > self.high_confidence_threshold:
                self._log(f"High confidence reached: {best_node.posterior_score:.3f}. Stopping.", "SUCCESS")
                break

            # Expand all nodes in current beam
            all_children = []
            for node in beam:
                children = self._expand_node(node, target, question)
                all_children.extend(children)

            if not all_children:
                self._log("No children generated. Stopping.", "WARNING")
                break

            # Prune low-confidence nodes
            all_children = [c for c in all_children if c.final_score > self.pruning_threshold]

            if not all_children:
                self._log("All children pruned. Stopping.", "WARNING")
                break

            # Select top-K for next beam
            all_children.sort(key=lambda c: c.final_score, reverse=True)
            beam = all_children[:self.beam_width]

            # Log beam
            self._print_beam(beam)

        # Return final beam
        self._log(f"Beam search completed. Final beam size: {len(beam)}", "SUCCESS")
        return beam

    def _print_beam(self, beam: List[SearchNode]):
        """Print current beam state"""
        if not beam or not self.verbose:
            return

        print("\n" + "="*80)
        print(f"{'ID':<10} | {'Depth':<5} | {'Prior':<6} | {'Post':<6} | {'Heur':<6} | {'Uncert':<6} | {'Final':<6}")
        print("="*80)
        for node in beam:
            print(f"{node.id[:8]:<10} | {node.depth:<5} | {node.prior_score:.3f}  | {node.posterior_score:.3f}  | "
                  f"{node.heuristic_score:.3f}  | {node.uncertainty_score:.3f}  | {node.final_score:.3f}")
        print("="*80 + "\n")

    # ==================== Phase 3: Multi-Target Cooperation ====================

    def _cooperative_multi_target_search(self,
                                         targets: List[str],
                                         question: str,
                                         relation_info: Dict,
                                         q_id: str) -> Dict[str, SearchNode]:
        """
        Perform cooperative search for multiple targets considering their spatial relations

        Returns: dict mapping target name to best node
        """
        if len(targets) == 1:
            # Single target, no cooperation needed
            beam = self._beam_search_single_target(targets[0], question, q_id)
            return {targets[0]: max(beam, key=lambda n: n.final_score)}

        # Multi-target case
        self._log(f"Multi-target cooperative search for: {targets}", "INFO")

        # Step 1: Independent search for each target
        target_beams = {}
        for target in targets:
            beam = self._beam_search_single_target(target, question, q_id)
            target_beams[target] = beam

        # Step 2: Select best candidate considering spatial relations
        relation_type = relation_info.get('relation', 'none')
        relation_pairs = relation_info.get('relation_pairs', [])

        if relation_type != 'none' and relation_pairs:
            self._log(f"Verifying spatial relation: {relation_type}", "INFO")
            best_nodes = self._select_consistent_nodes(target_beams, relation_pairs, relation_type, question)
        else:
            # No relations, just pick highest scoring node for each target
            best_nodes = {}
            for target, beam in target_beams.items():
                best_nodes[target] = max(beam, key=lambda n: n.final_score)

        return best_nodes

    def _select_consistent_nodes(self,
                                target_beams: Dict[str, List[SearchNode]],
                                relation_pairs: List[List[str]],
                                relation_type: str,
                                question: str) -> Dict[str, SearchNode]:
        """
        Select nodes that are consistent with spatial relations
        """
        # For simplicity, consider first relation pair
        if not relation_pairs:
            return {t: max(beam, key=lambda n: n.final_score) for t, beam in target_beams.items()}

        target1, target2 = relation_pairs[0]

        if target1 not in target_beams or target2 not in target_beams:
            return {t: max(beam, key=lambda n: n.final_score) for t, beam in target_beams.items()}

        beam1 = target_beams[target1]
        beam2 = target_beams[target2]

        # Find best combination that satisfies spatial relation
        best_combo = None
        best_score = -1.0

        for node1 in beam1:
            for node2 in beam2:
                # Check if spatial relation is satisfied
                is_valid = SpatialRelationshipTool.verify_relation(
                    node1.box_xywh,
                    node2.box_xywh,
                    relation_type
                )

                # Combined score
                combo_score = (node1.final_score + node2.final_score) / 2

                # Bonus for valid relations
                if is_valid:
                    combo_score *= 1.2

                if combo_score > best_score:
                    best_score = combo_score
                    best_combo = (node1, node2)

        if best_combo:
            result = {target1: best_combo[0], target2: best_combo[1]}
        else:
            result = {target1: beam1[0], target2: beam2[0]}

        # Add other targets
        for target, beam in target_beams.items():
            if target not in result:
                result[target] = max(beam, key=lambda n: n.final_score)

        return result

    # ==================== Phase 4: Self-Reflection ====================

    def _self_reflection(self,
                        best_nodes: Dict[str, SearchNode],
                        question: str) -> Dict[str, Any]:
        """
        Self-reflection to check if search was thorough

        Returns: dict with reflection results
        """
        # Compute average confidence
        avg_confidence = np.mean([node.posterior_score for node in best_nodes.values()])

        if avg_confidence > 0.7:
            return {
                'should_backtrack': False,
                'confidence': avg_confidence,
                'reasoning': 'High confidence, no backtracking needed'
            }

        # Low confidence, ask model for reflection
        search_summary = "\n".join([
            f"- {target}: box={node.box_xywh}, confidence={node.posterior_score:.2f}"
            for target, node in best_nodes.items()
        ])

        prompt = VisualSearchPrompts.SELF_REFLECTION.format(
            question=question,
            search_summary=search_summary,
            confidence=avg_confidence
        )

        response = self._chat_wrapper(prompt, [self.root_image_pil])
        parsed = self._extract_json_from_text(response)

        if parsed['status']:
            return parsed['content']
        else:
            return {
                'should_backtrack': False,
                'confidence': avg_confidence,
                'reasoning': 'Reflection failed to parse'
            }

    # ==================== Phase 5: Answer Generation ====================

    def _generate_final_answer(self,
                              best_nodes: Dict[str, SearchNode],
                              question: str,
                              q_id: str) -> str:
        """
        Generate final answer based on located evidence
        """
        self._log("Generating final answer", "INFO")

        # Create annotated global view
        global_img = self.root_image_pil.copy()
        draw = ImageDraw.Draw(global_img)

        # Load font
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        except:
            try:
                font = ImageFont.load_default(size=40)
            except:
                font = ImageFont.load_default()

        # Draw boxes and collect crops
        evidence_imgs = [global_img]
        crop_descriptions = []

        for i, (target, node) in enumerate(best_nodes.items()):
            x, y, w, h = node.box_xywh
            draw.rectangle([x, y, x+w, y+h], outline="red", width=5)

            # Add label
            label_text = target
            draw.text((x+10, y+10), label_text, fill="yellow", font=font)

            # Crop
            crop = self.root_image_pil.crop((x, y, x+w, y+h))
            evidence_imgs.append(crop)
            crop_descriptions.append(f"- Image {i+2}: '{target}' at box {node.box_xywh}")

        # Save annotated image
        global_save_path = os.path.join(self.image_save_dir, f"{q_id}_annotated.jpg")
        global_img.save(global_save_path)

        # Prepare evidence description
        evidence_desc = "Image 1: Global view with all targets marked\n" + "\n".join(crop_descriptions)

        # Compute statistics
        num_targets = len(best_nodes)
        max_depth = max(node.depth for node in best_nodes.values())
        avg_confidence = np.mean([node.posterior_score for node in best_nodes.values()])

        # Generate answer
        prompt = VisualSearchPrompts.ANSWER_WITH_EVIDENCE_V2.format(
            question=question,
            evidence_descriptions=evidence_desc,
            num_targets=num_targets,
            max_depth=max_depth,
            avg_confidence=f"{avg_confidence:.2f}",
            relations_verified="Yes" if len(best_nodes) > 1 else "N/A"
        )

        response = self._chat_wrapper(prompt, evidence_imgs)

        # Try to parse JSON answer
        parsed = self._extract_json_from_text(response)
        if parsed['status'] and 'answer' in parsed['content']:
            answer = parsed['content']['answer']
        else:
            # Fallback: use raw response
            answer = response.strip()

        self._log(f"Final answer: {answer}", "SUCCESS")
        return answer

    # ==================== Main Entry Point ====================

    def run(self, img_path: str, question_text: str, q_id: str) -> str:
        """
        Main entry point for enhanced visual search

        Args:
            img_path: path to image
            question_text: question text
            q_id: question ID

        Returns: final answer string
        """
        self._log(f"="*80, "INFO")
        self._log(f"Visual Search V2 - Question ID: {q_id}", "INFO")
        self._log(f"Question: {question_text}", "INFO")
        self._log(f"="*80, "INFO")

        # Reset per-sample trackers
        if vlm_tracker:
            vlm_tracker.reset()
        self.score_distributions = []

        # Load image (tolerate truncated files)
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.root_image_pil = Image.open(img_path).convert('RGB')

        # Phase 1: Extract targets and relations
        target_info = self._extract_targets_with_relations(question_text)
        targets = target_info['targets']
        self._log(f"Extracted targets: {targets}", "INFO")
        self._log(f"Relation type: {target_info.get('relation', 'none')}", "INFO")

        # Phase 2 & 3: Cooperative multi-target search
        best_nodes = self._cooperative_multi_target_search(targets, question_text, target_info, q_id)

        # Phase 4: Self-reflection (optional)
        reflection = self._self_reflection(best_nodes, question_text)
        if reflection.get('should_backtrack', False):
            self._log(f"Reflection suggests backtracking: {reflection.get('reasoning', '')}", "WARNING")
            # TODO: Implement backtracking logic if needed

        # Phase 5: Generate final answer
        final_answer = self._generate_final_answer(best_nodes, question_text, q_id)

        # Save logs
        # Get VLM call stats (E3/E8)
        vlm_stats = vlm_tracker.get_stats() if vlm_tracker else {}

        log_data = {
            'q_id': q_id,
            'question': question_text,
            'targets': targets,
            'relation_info': target_info,
            'best_nodes': {
                target: {
                    'box': node.box_xywh,
                    'depth': node.depth,
                    'scores': {
                        'prior': node.prior_score,
                        'posterior': node.posterior_score,
                        'heuristic': node.heuristic_score,
                        'final': node.final_score
                    }
                }
                for target, node in best_nodes.items()
            },
            'reflection': reflection,
            'answer': final_answer,
            # Rebuttal analysis data
            'score_distributions': self.score_distributions,  # E1/E2
            'vlm_stats': vlm_stats,  # E3/E8
            'config': {  # Record experiment config
                'token_set': self.token_set,
                'mu_threshold': self.mu_threshold,
                'no_heuristic': self.no_heuristic,
                'entropy_skip_threshold': self.entropy_skip_threshold,
                'beam_width': self.beam_width,
                'max_depth': self.max_depth,
            }
        }

        log_file = os.path.join(self.log_dir, f"{q_id}_search_log.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        return final_answer

    def run_direct(self, img_path: str, question_text: str, q_id: str) -> Tuple[str, str]:
        """
        Direct VQA without search (baseline)

        Returns: (answer, reasoning)
        """
        self._log(f"Direct VQA mode for Q{q_id}", "INFO")
        self.root_image_pil = Image.open(img_path).convert('RGB')

        prompt = f"Question: {question_text}\n\nAnswer the question based on the image. If multiple-choice, output the option letter."
        response = self._chat_wrapper(prompt, [self.root_image_pil])

        return response.strip(), "Direct VQA (no search)"
