import os
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils_llm import chat_vlm
from utils import parse_json_from_response
from prompt import JigsawPrompts, DirectPrompts

class SearchNode:
    def __init__(self, state, img_path, parent=None, action=None, depth=0):
        self.state = tuple(state)
        self.img_path = img_path
        self.parent = parent
        self.action = action
        self.depth = depth

        self.s_policy = 0.0
        self.s_observer = 0.0
        self.s_heuristic = 0.0
        self.s_final = 0.0

    def __hash__(self): return hash(self.state)
    def __eq__(self, other): return self.state == other.state

class JigsawAgent:
    def __init__(self, env, model_name="gpt4o", verbose=False, entropy_skip_threshold=-1.0):
        self.env = env
        self.model_name = model_name
        self.verbose = verbose
        self.entropy_skip_threshold = entropy_skip_threshold
        self.cv_evaluator = env.evaluator
        self.grid_n = env.rows
        self.vlm_cache = {}
        self.initial_img_path = None

    # --- Direct Mode ---
    def run_direct(self):
        print(f"🚀 Running Direct Mode for {self.env.img_name}")
        prompt = DirectPrompts.DIRECT_SOLVE.format(
            grid_n=self.grid_n, img_path=self.env.current_img_path
        )
        response, _ = chat_vlm(prompt, self.model_name, [], temperature=0.0)
        parsed = parse_json_from_response(response)
        
        if parsed['status']:
            final_perm = parsed['content'].get('final_permutation', [])
            if len(final_perm) == self.grid_n ** 2:
                final_path = self.env.rearrange_tiles(final_perm, tag="DIRECT_RESULT")
                print(f"✅ Direct Result: {final_perm}")
                return final_perm
        print("❌ Direct Parse Failed")
        return list(range(1, self.grid_n**2 + 1))

    # --- V-BAS (Adaptive Beam Search) Mode ---
    def run_beam(self, beam_width=3, max_depth=5):
        print(f"Running V-BAS Mode | Beam: {beam_width} | Depth: {max_depth}")
        
        # 1. Init
        init_perm = list(range(1, self.grid_n**2 + 1))
        # Save the initial scrambled image path as observer reference
        self.initial_img_path = self.env.current_img_path
        
        # Init score
        h_score, _ = self.cv_evaluator.evaluate_permutation(init_perm, self.grid_n)
        
        root = SearchNode(init_perm, self.initial_img_path, depth=0)
        root.s_heuristic = h_score
        root.s_final = h_score
        
        current_beam = [root]
        visited = {root.state}
        
        for step in range(1, max_depth + 1):
            print(f"\n=== Step {step}/{max_depth} | Beam Size: {len(current_beam)} ===")
            candidates = []
            first_step_uncertainty = 1.0
            
            # --- A. Policy & Execution ---
            for node in current_beam:
                # Action generation (Policy)
                proposed = self._policy_propose(node, samples=beam_width)
                if node.parent: candidates.append(node)
                
                for act in proposed:
                    perm = act['perm']
                    if tuple(perm) in visited: continue
                    
                    # Execute (Executor)
                    tag = f"s{step}_n{hash(tuple(perm))%1000}"
                    new_img = self.env.rearrange_tiles(perm, tag=tag, is_simulation=True)
                    
                    child = SearchNode(perm, new_img, parent=node, action="reorder", depth=step)
                    child.s_policy = act['score'] 
                    
                    # Heuristic (CV Gradient)
                    child.s_heuristic, _ = self.cv_evaluator.evaluate_permutation(perm, self.grid_n)

                    candidates.append(child)
                    visited.add(child.state)
                    
                    if self.verbose:
                        print(f"    [Gen] Perm:{str(perm).replace(' ','')}.. | Policy:{child.s_policy:.2f} | Heuristic:{child.s_heuristic:.2f}")

            if not candidates: 
                print("No valid candidates generated.")
                break
        
            # --- B. Batch Observer (VLM Logits with Comparison) ---
            # Check if heuristic scores are confident enough to skip observer
            heur_scores = [c.s_heuristic for c in candidates]
            skip_observer = False
            if self.entropy_skip_threshold > 0 and heur_scores:
                import numpy as np
                from tool import UncertaintyEstimator
                total = sum(heur_scores) if sum(heur_scores) > 0 else 1.0
                heur_probs = [s / total for s in heur_scores]
                heur_entropy = UncertaintyEstimator.compute_entropy(heur_probs)
                skip_observer = (heur_entropy < self.entropy_skip_threshold)

            if skip_observer:
                print(f"  > Skipping observer (entropy below threshold)")
                for c in candidates:
                    c.s_observer = c.s_heuristic  # Use heuristic as proxy
            else:
                print(f"  > Batch Evaluating {len(candidates)} candidates via VLM (Comparison Logits)...")
                self._batch_observer_evaluate(candidates)

            # --- C. Adaptive Weighting & Selection ---
            uncertainty = np.std([c.s_heuristic for c in candidates]) if candidates else 0
            print(f'total_uncertainty is :{uncertainty}, {[c.s_heuristic for c in candidates]}')
            # uncertainty = np.std([c.s_observer for c in candidates]) if candidates else 0
            if step == 1:
                first_step_uncertainty = candidates[0].s_heuristic
                print(f'first_uncertainty is :{uncertainty}')
            w_p, w_h, w_o = 0, 1, 0
            print(f"  > Weights -> Policy: {w_p:.2f}, CV: {w_h:.2f}, Obs: {w_o:.2f}")

            print(f"  > Detailed Scores Calculation:")
            for c in candidates:
                c.s_final = w_p * c.s_policy + w_h * c.s_heuristic + w_o * c.s_observer
                print(f"    - Perm:{str(c.state[:3]).replace(' ','')}.. -> "
                    f"P:{c.s_policy:.2f}*{w_p:.2f} + "
                    f"H:{c.s_heuristic:.2f}*{w_h:.2f} + "
                    f"O:{c.s_observer:.2f}*{w_o:.2f} + "
                    f"par:{c.parent.s_final:.2f} = "
                    f"Final:{c.s_final:.4f}")

            # Sort and Prune
            current_beam = sorted(candidates, key=lambda x: x.s_final, reverse=True)[:beam_width]
            
            best = current_beam[0]
            print(f"  🏆 Step Best: {best.s_final:.4f}")
            
            if best.s_heuristic >= 0.70:
                print("🎉 Converged!")
                return list(best.state)

        return list(current_beam[0].state), first_step_uncertainty

    def _policy_propose(self, node, samples=3):
        prompt = JigsawPrompts.PREDICT_PERMUTATION.format(
            grid_n=self.grid_n, img_path=node.img_path, 
            max_idx=self.grid_n**2, num_samples=samples
        )
        resp, _ = chat_vlm(prompt, self.model_name, [], temperature=0.7)
        parsed = parse_json_from_response(resp)
        
        actions = []
        if parsed['status']:
            for item in parsed['content'].get('candidates', []):
                mat = item.get('tile_matrix')
                if mat:
                    flat = [x for row in mat for x in row]
                    if len(flat) == self.grid_n**2 and len(set(flat)) == self.grid_n**2:
                        actions.append({'perm': flat, 'score': 0.2})
        if not actions: actions.append({'perm': list(node.state), 'score': 0.1})
        return actions

    # Only accepts a prompt string, as the prompt already contains the image tag
    def get_vlm_logits_score(self, prompt):
        # Use the prompt itself as the cache key
        if prompt in self.vlm_cache:
            return self.vlm_cache[prompt]
        
        try:
            # logit=True, max_tokens=1
            response, _, prob_map = chat_vlm(
                prompt, self.model_name, messages=[], logit=True, max_tokens=1, temperature=0.0
            )
            
            score_yes = prob_map.get('Yes', 0.0)
            score_lower = prob_map.get('yes', 0.0)
            score = max(score_yes, score_lower)

            score = (score - 0.5) * 0.5
            
            self.vlm_cache[prompt] = score
            return score
        except Exception as e:
            print(f"      [VLM Logit Error] {e}", flush=True)
            return 0.0

    # Build a prompt with dual-image comparison for the observer
    def _batch_observer_evaluate(self, nodes):
        def _worker(node):
            prompt = (
                f"Image 1: <img src='{node.parent.img_path}'>\n"
                f"Image 2: <img src='{node.img_path}'>\n"
                "Compare Image 1 and Image 2. "
                "Image 1 is the initial scrambled puzzle. Image 2 is a proposed restoration. "
                "Is Image 2 visually more coherent and better restored than Image 1? "
                "Answer Yes or No."
            )
            score = self.get_vlm_logits_score(prompt)
            return score

        with ThreadPoolExecutor(max_workers=8) as executor:
            scores = list(executor.map(_worker, nodes))
        
        for i, node in enumerate(nodes):
            node.s_observer = scores[i]
            if self.verbose:
                short_perm = str(node.state[:3]) + "..."
                print(f"    [Obs-Logit] Perm:{short_perm} Score(Yes): {node.s_observer:.4f}")
