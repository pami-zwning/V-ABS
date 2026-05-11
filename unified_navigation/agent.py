import os
import shutil
import json
import math
import cv2
import sys
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set matplotlib backend to avoid errors on headless servers
matplotlib.use('Agg')

# --- Path Setup ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# --- Import Utils ---
try:
    from utils import parse_json_from_response
    from utils_llm import chat_vlm, vlm_tracker
except ImportError:
    print(f"Error: Utils not found in {parent_dir}.")
    sys.exit(1)

from env import PixelMazeEnv, GridMazeEnv
from prompt import Prompts

# --- Hyperparameters ---
BASE_W_PRIOR = 0.1
BASE_W_OBSERVE = 0.7
BASE_W_HEURISTIC = 0.2
MAX_TRIES_INIT = 3

# --- Visual Formatting Helpers ---
USE_COLOR = sys.stdout.isatty()

class TColors:
    HEADER = '\033[95m' if USE_COLOR else ''
    OKBLUE = '\033[94m' if USE_COLOR else ''
    OKCYAN = '\033[96m' if USE_COLOR else ''
    OKGREEN = '\033[92m' if USE_COLOR else ''
    WARNING = '\033[93m' if USE_COLOR else ''
    FAIL = '\033[91m' if USE_COLOR else ''
    ENDC = '\033[0m' if USE_COLOR else ''
    BOLD = '\033[1m' if USE_COLOR else ''
    GREY = '\033[90m' if USE_COLOR else ''

def color_score(val_str):
    if val_str == '-': return f"{TColors.GREY}-{TColors.ENDC}"
    try:
        val = float(val_str)
        if val >= 0.8: return f"{TColors.OKGREEN}{val_str}{TColors.ENDC}"
        if val < 0.0: return f"{TColors.FAIL}{val_str}{TColors.ENDC}"
        if val < 0.3: return f"{TColors.GREY}{val_str}{TColors.ENDC}"
        return f"{TColors.WARNING}{val_str}{TColors.ENDC}"
    except:
        return val_str

COL_VISUAL_W = 10
COL_ANSI_W = 9 if USE_COLOR else 0
COL_FMT_W = COL_VISUAL_W + COL_ANSI_W

def print_header(depth, beam_size):
    print(f"\n{TColors.HEADER}{'='*90}{TColors.ENDC}", flush=True)
    print(f"{TColors.BOLD} 🌊 DEPTH {depth} | Current Beam Size: {beam_size}{TColors.ENDC}", flush=True)
    print(f"{TColors.HEADER}{'='*90}{TColors.ENDC}", flush=True)

def print_node_header(pos, action, entropy, weights):
    w_p, w_o, w_h = weights
    print(f"\n{TColors.OKCYAN}📍 [EXPANDING] Node: {pos} (via {action}){TColors.ENDC}", flush=True)
    print(f"   🎲 Uncertainty: {TColors.WARNING}{entropy:.4f}{TColors.ENDC} | Weights: P={w_p:.2f} O={TColors.BOLD}{w_o:.2f}{TColors.ENDC} H={w_h:.2f}", flush=True)
    print(f"   {'-'*88}", flush=True)
    vw = COL_VISUAL_W
    print(f"   {'Action':<8} | {'Prior':<{vw}} | {'Obs':<{vw}} | {'Heur':<{vw}} | {'StepSum':<{vw}} | {'Cumulative':<{vw}} | {'Status'}", flush=True)
    print(f"   {'-'*88}", flush=True)

# --- Caching ---
class VLMCache:
    def __init__(self): self.cache = {}
    def get_key(self, prompt, img_path):
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        img_hash = hashlib.md5(img_path.encode('utf-8')).hexdigest()
        return f"{prompt_hash}_{img_hash}"
    def get(self, prompt, img_path): return self.cache.get(self.get_key(prompt, img_path), None)
    def set(self, prompt, img_path, value): self.cache[self.get_key(prompt, img_path)] = value

vlm_cache = VLMCache()

class SearchNode:
    def __init__(self, curr_pos, prev_pos, env_state, depth, 
                 score_prior=0.0, score_obs=0.0, score_h=0.0, 
                 cumulative_score=0.0, path_actions=None):
        self.curr_pos = curr_pos
        self.prev_pos = prev_pos
        self.env_state = env_state
        self.depth = depth
        self.score_prior = score_prior
        self.score_obs = score_obs
        self.score_h = score_h
        self.cumulative_score = cumulative_score 
        self.path_actions = path_actions if path_actions else []

class UnifiedSearchAgent:
    def __init__(self, output_dir, model_name, beam_width, max_depth, verbose=False, prompts_class=None, **kwargs):
        self.output_dir = output_dir
        self.model_name = model_name
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.verbose = verbose
        self.prompts = prompts_class if prompts_class else Prompts
        self.executor = ThreadPoolExecutor(max_workers=1) 
        
        # Save search type (beam, direct, v_labs)
        self.search_type = kwargs.get('search_type', 'beam')
        self.entropy_skip_threshold = kwargs.get('entropy_skip_threshold', -1.0)
        self.token_set = kwargs.get('token_set', 'yes_no')
        self.mu_threshold = kwargs.get('mu_threshold', 0.5)
        self.no_heuristic = kwargs.get('no_heuristic', False)

        # Score distribution collector (rebuttal E1/E2)
        self.score_distributions = []
        
        # --- Similarity log file paths ---
        self.similarity_log_file = os.path.join(os.path.dirname(self.output_dir), "similarity_log.json")
        self.histogram_save_path = os.path.join(os.path.dirname(self.output_dir), "similarity_histogram.png")
        self.prior_var_save_path = os.path.join(os.path.dirname(self.output_dir), "prior_var.json")
        self.post_var_save_path = os.path.join(os.path.dirname(self.output_dir), "post_var.json")
        self.vabs_var_save_path = os.path.join(os.path.dirname(self.output_dir), "v_abs_var.json")

    def _chat(self, prompt, messages=None, **kwargs):
        if messages is None: messages = []
        response, new_history = chat_vlm(prompt, self.model_name, deepcopy(messages), logit=False, **kwargs)
        return response, new_history

    # --- Similarity computation and plotting ---
    def _record_and_plot_similarity(self, priors, posteriors):
        try:
            vec_p = np.array(priors) - 1.0 / len(priors)
            vec_o = np.array(posteriors) - 1.0 / len(posteriors)
            
            norm_p = np.linalg.norm(vec_p)
            norm_o = np.linalg.norm(vec_o)

            prior_var = np.std(vec_p) ** 2
            post_var = np.std(vec_o) ** 2

            w_p = 1.0 / (1.0 + np.exp(np.std(norm_p) ** 2))
            w_o = 1 - w_p
            vabs_var = np.std(w_p * vec_p + w_o * vec_o) ** 2
            
            if norm_p == 0 or norm_o == 0:
                sim_score = 0.0
            else:
                sim_score = np.dot(vec_p, vec_o) / (norm_p * norm_o)
            
            sim_score = float(sim_score)
            
            existing_scores = []
            if os.path.exists(self.similarity_log_file):
                try:
                    with open(self.similarity_log_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            existing_scores = data
                except Exception:
                    pass

            existing_scores.append(sim_score)
            try:
                with open(self.similarity_log_file, 'w') as f:
                    json.dump(existing_scores, f)
            except Exception:
                pass

            prior_var_list = []
            if os.path.exists(self.prior_var_save_path):
                try:
                    with open(self.prior_var_save_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            prior_var_list = data
                except Exception:
                    pass

            prior_var_list.append(prior_var)
            try:
                with open(self.prior_var_save_path, 'w') as f:
                    json.dump(prior_var_list, f)
            except Exception:
                pass

            vabs_var_list = []
            if os.path.exists(self.vabs_var_save_path):
                try:
                    with open(self.vabs_var_save_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            vabs_var_list = data
                except Exception:
                    pass

            vabs_var_list.append(vabs_var)
            try:
                with open(self.vabs_var_save_path, 'w') as f:
                    json.dump(vabs_var_list, f)
            except Exception:
                pass

            post_var_list = []
            if os.path.exists(self.post_var_save_path):
                try:
                    with open(self.post_var_save_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            post_var_list = data
                except Exception:
                    pass

            post_var_list.append(post_var)
            try:
                with open(self.post_var_save_path, 'w') as f:
                    json.dump(post_var_list, f)
            except Exception:
                pass

            if self.verbose:
                print(f"    📊 Similarity Score: {sim_score:.4f} (Total Count: {len(existing_scores)})", flush=True)

            self._draw_histogram(existing_scores)

        except Exception as e:
            print(f"⚠️ Error in similarity calculation: {e}", flush=True)

    def _draw_histogram(self, scores):
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f'Histogram of Prior-Posterior Similarity Scores (N={len(scores)})')
            plt.xlabel('Cosine Similarity Score')
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.5)
            plt.xlim(-1.0, 1.0)
            if scores:
                avg_score = sum(scores) / len(scores)
                plt.axvline(avg_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_score:.2f}')
                plt.legend()
            plt.savefig(self.histogram_save_path)
            plt.close()
        except Exception as e:
            print(f"⚠️ Error plotting histogram: {e}", flush=True)

    # --- Abstract Methods ---
    def get_heuristic_score(self, curr_pos, goal_pos, prev_pos=None, **kwargs): raise NotImplementedError
    def check_hard_constraints(self, curr_pos, next_pos, env_w, env_h):
        x, y = next_pos
        return 0 <= x < env_w and 0 <= y < env_h
    def predict_next_pos(self, curr_pos, action, **kwargs): raise NotImplementedError
    def execute_action(self, env, curr_pos, action, **kwargs): raise NotImplementedError
    def get_state_key(self, pos, **kwargs): raise NotImplementedError

    def _extract_yes_logit(self, prob_map):
        """Extract positive-direction probability (E4: configurable token set)."""
        if not prob_map: return -10.0
        TOKEN_SETS = {
            "yes_no": ['Yes', ' Yes', 'yes', ' yes', 'YES'],
            "true_false": ['True', ' True', 'true', ' true', 'TRUE'],
            "correct_incorrect": ['Correct', ' Correct', 'correct', ' correct', 'CORRECT'],
            "all": ['Yes', ' Yes', 'yes', ' yes', 'YES',
                     'True', ' True', 'true', ' true', 'TRUE',
                     'Correct', ' Correct', 'correct', ' correct', 'CORRECT'],
        }
        tokens = TOKEN_SETS.get(self.token_set, TOKEN_SETS["yes_no"])
        best_val = -10.0
        for k in tokens:
            if k in prob_map:
                val = prob_map[k]
                if val > best_val: best_val = val
        return best_val
    
    # --- VLM Helper ---
    def get_vlm_logits_score(self, prompt, img_path):
        cached_score = vlm_cache.get(prompt, img_path)
        if cached_score is not None: return cached_score
        try:
            response, _, prob_map = chat_vlm(
                prompt, self.model_name, messages=[], logit=True, max_tokens=1, temperature=0.0
            )
            score = self._extract_yes_logit(prob_map)
            vlm_cache.set(prompt, img_path, score)
            return score
        except Exception as e:
            print(f"      [VLM Error] {e}", flush=True)
            return 0.0

    def calculate_uncertainty(self, scores):
        if not scores: return 0.0
        arr = np.array(scores)
        if np.sum(arr) == 0: return 1.0
        probs = arr / np.sum(arr)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(scores) + 1e-9)
        return entropy / max_entropy if max_entropy > 0 else 0

    # --- CORE SEARCH LOOP ---
    def run_search_loop(self, env, start_pos, goal_pos, actions_set, **kwargs):
        # 1. Init
        init_state = env.save_state()
        root = SearchNode(start_pos, None, init_state, 0, 0, 0, 0, 0, [])
        beam = [root]
        
        finished_nodes = []
        visited = set()
        visited.add(self.get_state_key(start_pos, **kwargs))
        
        env_w, env_h = env.get_resolution()

        for depth in range(self.max_depth):
            if self.verbose:
                print_header(depth, len(beam))
            
            all_candidates_pool = [] 
            first_step_uncertain = 1.0
            if not beam: break

            # --- PROCESS BEAM ---
            for node in beam:
                # Goal Check
                dist = math.hypot(node.curr_pos[0]-goal_pos[0], node.curr_pos[1]-goal_pos[1])
                is_goal = False
                if 'step_size' in kwargs: 
                     if dist < kwargs['step_size'] * 1.5: is_goal = True
                else: 
                     if node.curr_pos == goal_pos: is_goal = True
                
                if is_goal:
                    print(f"{TColors.OKGREEN}🎉 Goal reached at depth {depth}! Path: {node.path_actions}{TColors.ENDC}", flush=True)
                    finished_nodes.append(node)
                    continue

                env.load_state(node.env_state)
                curr_img_path = env.current_img_path
                
                valid_actions = []
                prior_futures = {}
                
                action_report = {a: {'prior': '-', 'obs': '-', 'heur': '-', 'step': '-', 'cum': '-', 'status': 'SKIP'} for a in actions_set}

                # 1. Valid Action Filtering
                for action in actions_set:
                    if node.path_actions and self.is_reverse(action, node.path_actions[-1]): 
                        action_report[action]['status'] = 'REVERSE'
                        continue
                    
                    next_pos = self.predict_next_pos(node.curr_pos, action, **kwargs)
                    if not self.check_hard_constraints(node.curr_pos, next_pos, env_w, env_h): 
                        action_report[action]['status'] = 'WALL'
                        continue
                    
                    state_key = self.get_state_key(next_pos, **kwargs)
                    if state_key in visited: 
                        action_report[action]['status'] = 'VISITED'
                        continue
                    
                    valid_actions.append(action)
                    action_report[action]['status'] = 'PENDING'

                if not valid_actions: 
                    if self.verbose: 
                         print_node_header(node.curr_pos, node.path_actions[-1] if node.path_actions else 'Start', 0.0, (BASE_W_PRIOR, BASE_W_OBSERVE, BASE_W_HEURISTIC))
                         print(f"   {TColors.FAIL}Dead End (No valid actions){TColors.ENDC}\n", flush=True)
                    continue

                # 2. Batch Prior (parallel)
                action_scores_prior = {}
                for action in valid_actions:
                    prompt = self.prompts.PRIOR_CHECK_PROMPT.format(
                        curr_pos=node.curr_pos, goal_pos=goal_pos, 
                        action=action, img_path=curr_img_path
                    )
                    
                    future = self.executor.submit(self.get_vlm_logits_score, prompt, curr_img_path)
                    prior_futures[future] = action
                
                for future in as_completed(prior_futures):
                    act = prior_futures[future]
                    score = future.result()
                    action_scores_prior[act] = score
                    action_report[act]['prior'] = f"{score:.4f}"

                # 3. Adaptive Weights (sigmoid formula: w_p = 1/(1+e^{beta*(H-mu)}))
                raw_scores = list(action_scores_prior.values())
                uncertainty = self.calculate_uncertainty(raw_scores)
                if depth == 0:
                    first_step_uncertain = uncertainty

                beta = 2.0
                w_prior_sigmoid = 1.0 / (1.0 + math.exp(beta * (uncertainty - self.mu_threshold)))
                w_obs_sigmoid = 1.0 - w_prior_sigmoid
                w_heur = 0.0 if self.no_heuristic else BASE_W_HEURISTIC

                # Normalize
                w_total = w_prior_sigmoid + w_obs_sigmoid + w_heur
                curr_w_prior = w_prior_sigmoid / w_total
                curr_w_obs = w_obs_sigmoid / w_total
                curr_w_heur = w_heur / w_total
                
                # 4. Execute & Observe
                # Check if prior is confident enough to skip observer
                prior_entropy = self.calculate_uncertainty(raw_scores)
                skip_observer = (self.entropy_skip_threshold > 0 and prior_entropy < self.entropy_skip_threshold)

                obs_futures = {}
                execution_results = {}

                threshold = 0.00
                actions_to_run = [a for a, s in action_scores_prior.items() if s >= threshold]

                if not actions_to_run and valid_actions:
                    best_act = max(action_scores_prior, key=action_scores_prior.get)
                    actions_to_run = [best_act]
                    action_report[best_act]['status'] = 'RESCUED'

                # Execute (sequential)
                for action, score in action_scores_prior.items():
                    if action not in actions_to_run:
                        action_report[action]['status'] = 'PRUNED'
                        continue

                    env.load_state(node.env_state)
                    exec_res = self.execute_action(env, node.curr_pos, action, **kwargs)
                    if exec_res['status']:
                        execution_results[action] = exec_res
                    else:
                        action_report[action]['status'] = 'EXEC_FAIL'

                # 5. Batch Observer (parallel, or skip if prior is confident)
                if not skip_observer:
                    for action, res in execution_results.items():
                        next_pos = self.predict_next_pos(node.curr_pos, action, **kwargs)
                        prompt = self.prompts.OBSERVER_CHECK_PROMPT.format(
                            action=action, prev_pos=node.curr_pos, curr_pos=next_pos, goal_pos=goal_pos,
                            curr_img=res['curr_img'], crop_img=res['crop_img']
                        )
                        future = self.executor.submit(self.get_vlm_logits_score, prompt, res['crop_img'])
                        obs_futures[future] = action

                candidates_for_this_node = []
                action_scores_obs_temp = {}

                if skip_observer:
                    # Use prior scores as observer scores when skipping
                    for action, res in execution_results.items():
                        score_obs = action_scores_prior[action]
                        action_scores_obs_temp[action] = score_obs
                        score_prior = action_scores_prior[action]
                        next_pos = self.predict_next_pos(node.curr_pos, action, **kwargs)
                        score_h = self.get_heuristic_score(next_pos, goal_pos, prev_pos=node.curr_pos, **kwargs)
                        step_score = (curr_w_prior * score_prior) + (curr_w_obs * score_obs) + (curr_w_heur * score_h)
                        new_cumulative_score = node.cumulative_score + step_score

                        if action_report[action]['status'] != 'RESCUED':
                            action_report[action]['status'] = 'OK'
                        action_report[action].update({
                            'obs': f"{score_obs:.4f}(skip)",
                            'heur': f"{score_h:.4f}",
                            'step': f"{step_score:.4f}",
                            'cum': f"{new_cumulative_score:.4f}"
                        })

                        child_state = deepcopy(node.env_state)
                        child_state['history'].append(res['curr_img'])
                        child_state['cnt'] += 1

                        child = SearchNode(
                            curr_pos=next_pos, prev_pos=node.curr_pos,
                            env_state=child_state, depth=depth+1,
                            score_prior=score_prior, score_obs=score_obs, score_h=score_h,
                            cumulative_score=new_cumulative_score,
                            path_actions=node.path_actions + [action]
                        )
                        candidates_for_this_node.append(child)
                        visited.add(self.get_state_key(next_pos, **kwargs))
                        self.score_distributions.append({
                            'depth': depth, 'prior_score': score_prior, 'posterior_score': score_obs,
                            'heuristic_score': score_h, 'entropy': uncertainty, 'step_score': step_score,
                            'weights': {'prior': curr_w_prior, 'observer': curr_w_obs, 'heuristic': curr_w_heur},
                        })
                else:
                    for future in as_completed(obs_futures):
                        action = obs_futures[future]
                        score_obs = future.result()
                        action_scores_obs_temp[action] = score_obs
                        score_prior = action_scores_prior[action]
                        exec_res = execution_results[action]
                        next_pos = self.predict_next_pos(node.curr_pos, action, **kwargs)

                        score_h = self.get_heuristic_score(next_pos, goal_pos, prev_pos=node.curr_pos, **kwargs)

                        step_score = (curr_w_prior * score_prior) + (curr_w_obs * score_obs) + (curr_w_heur * score_h)
                        new_cumulative_score = node.cumulative_score + step_score

                        if action_report[action]['status'] != 'RESCUED':
                            action_report[action]['status'] = 'OK'

                        action_report[action].update({
                            'obs': f"{score_obs:.4f}",
                            'heur': f"{score_h:.4f}",
                            'step': f"{step_score:.4f}",
                            'cum': f"{new_cumulative_score:.4f}"
                        })

                        # Create Child
                        child_state = deepcopy(node.env_state)
                        child_state['history'].append(exec_res['curr_img'])
                        child_state['cnt'] += 1

                        child = SearchNode(
                            curr_pos=next_pos, prev_pos=node.curr_pos,
                            env_state=child_state, depth=depth+1,
                            score_prior=score_prior, score_obs=score_obs, score_h=score_h,
                            cumulative_score=new_cumulative_score,
                            path_actions=node.path_actions + [action]
                        )
                        candidates_for_this_node.append(child)
                        visited.add(self.get_state_key(next_pos, **kwargs))
                        self.score_distributions.append({
                            'depth': depth, 'prior_score': score_prior, 'posterior_score': score_obs,
                            'heuristic_score': score_h, 'entropy': uncertainty, 'step_score': step_score,
                            'weights': {'prior': curr_w_prior, 'observer': curr_w_obs, 'heuristic': curr_w_heur},
                        })

                # --- Similarity computation ---
                aligned_priors = []
                aligned_obs = []
                for act, o_score in action_scores_obs_temp.items():
                    if act in action_scores_prior:
                        aligned_priors.append(action_scores_prior[act])
                        aligned_obs.append(o_score)
                
                self._record_and_plot_similarity(aligned_priors, aligned_obs)


                all_candidates_pool.extend(candidates_for_this_node)

                # --- PRINT BEAUTIFUL TABLE FOR THIS NODE ---
                if self.verbose:
                    print_node_header(node.curr_pos, node.path_actions[-1] if node.path_actions else 'Start', uncertainty, (curr_w_prior, curr_w_obs, BASE_W_HEURISTIC))
                    for action in actions_set:
                        r = action_report[action]
                        status_str = r['status']
                        
                        if status_str == 'OK': status_str = f"{TColors.OKGREEN}OK{TColors.ENDC}"
                        elif status_str == 'RESCUED': status_str = f"{TColors.WARNING}RESCUED{TColors.ENDC}"
                        elif status_str == 'PRUNED': status_str = f"{TColors.FAIL}PRUNED{TColors.ENDC}"
                        else: status_str = f"{TColors.GREY}{status_str}{TColors.ENDC}"
                        
                        p_str = color_score(r['prior'])
                        o_str = color_score(r['obs'])
                        h_str = color_score(r['heur'])
                        s_str = color_score(r['step'])
                        
                        print(f"   {action:<8} | {p_str:<{COL_FMT_W}} | {o_str:<{COL_FMT_W}} | {h_str:<{COL_FMT_W}} | {s_str:<{COL_FMT_W}} | {r['cum']:<{COL_VISUAL_W}} | {status_str}", flush=True)
                    print(f"   {'-'*88}\n", flush=True)

            # --- SELECTION PHASE ---
            if not all_candidates_pool and not finished_nodes:
                print(f"{TColors.FAIL}No valid candidates found in this depth. Terminating.{TColors.ENDC}", flush=True)
                break
            
            all_candidates_pool.sort(key=lambda x: x.cumulative_score, reverse=True)
            beam = all_candidates_pool[:self.beam_width]

            if self.verbose and beam:
                print(f"{TColors.OKBLUE}🏆 [SELECTED TOP {len(beam)}]{TColors.ENDC}", flush=True)
                for i, b in enumerate(beam):
                    print(f"   {i+1}. Act: {TColors.BOLD}{b.path_actions[-1]:<6}{TColors.ENDC} | Pos: {b.curr_pos} | Total Score: {TColors.OKGREEN}{b.cumulative_score:.4f}{TColors.ENDC}", flush=True)
                print("\n", flush=True)

        final_set = finished_nodes + beam
        if not final_set: return None
        final_set.sort(key=lambda x: x.cumulative_score, reverse=True)
        return final_set[0], first_step_uncertain

    def is_reverse(self, act1, act2):
        pairs = {('Up', 'Down'), ('Down', 'Up'), ('Left', 'Right'), ('Right', 'Left'),
                 ('U', 'D'), ('D', 'U'), ('L', 'R'), ('R', 'L')}
        return (act1, act2) in pairs or (act2, act1) in pairs


# =========================================================================
# Pixel Maze Agent
# =========================================================================
class PixelMazeAgent(UnifiedSearchAgent):
    def __init__(self, task_id, image_path, prompt_text, options, output_dir, model_name, **kwargs):
        super().__init__(output_dir, model_name, kwargs.get('beam_width', 3), kwargs.get('max_depth', 50), kwargs.get('verbose', True), search_type=kwargs.get('search_type', 'beam'))
        self.task_id = task_id
        
        self.task_dir = os.path.join(output_dir, f"task_{task_id}")
        if os.path.exists(self.task_dir): shutil.rmtree(self.task_dir)
        os.makedirs(self.task_dir, exist_ok=True)
        
        self.map_name = "map.png"
        self.local_map_path = os.path.join(self.task_dir, self.map_name)
        shutil.copy(image_path, self.local_map_path)
        
        self.env = PixelMazeEnv(self.task_dir, self.map_name)
        self.prompt_text = prompt_text
        self.options = options

    # --- Initialization Methods (CV + VLM) ---
    def auto_detect_maze_specs(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return None, None, None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            start_pos = self._get_centroid(mask_red)
            mask_green = cv2.inRange(hsv, np.array([35, 70, 50]), np.array([85, 255, 255]))
            goal_pos = self._get_centroid(mask_green)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            step_size = int(np.max(cv2.distanceTransform(thresh, cv2.DIST_L2, 5)) * 2)
            return start_pos, goal_pos, step_size
        except Exception as e:
            print(f"[CV Error] {e}", flush=True)
            return None, None, None

    def _get_centroid(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0: return [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        return None

    def get_maze_specs(self):
        w, h = self.env.get_resolution()
        print(f"\n{'='*20} Phase 1: Spec Calibration {'='*20}", flush=True)
        cv_start, cv_goal, cv_step = self.auto_detect_maze_specs(self.local_map_path)
        if cv_start and cv_goal:
            print(f"✅ [CV] Start:{cv_start}, Goal:{cv_goal}, Step:{cv_step}", flush=True)
            return cv_start, cv_goal, cv_step
            
        print("⚠️ [WARN] CV failed. Using VLM...", flush=True)
        prompt = self.prompts.MAZE_SPECS.format(img_path=self.local_map_path, width=w, height=h)
        s, g, st = [20, 20], [w-20, h-20], 30 
        for _ in range(3):
            resp, _ = self._chat(prompt, [])
            parsed = parse_json_from_response(resp)
            if parsed['status']:
                c = parsed['content']
                s_norm = c.get('start_norm', [50, 50])
                g_norm = c.get('goal_norm', [950, 950])
                step_norm = c.get('step_size_norm', 30)
                s = [int((s_norm[0]/1000)*w), int((s_norm[1]/1000)*h)]
                g = [int((g_norm[0]/1000)*w), int((g_norm[1]/1000)*h)]
                st = int((step_norm/1000) * w)
                if st < 10: st = 30
                print(f"✅ [VLM] Start:{s}, Goal:{g}, Step:{st}", flush=True)
                return s, g, st
        return s, g, st

    # --- Search Overrides ---
    def get_state_key(self, pos, **kwargs):
        step = kwargs.get('step_size', 30)
        grid = max(1, int(step / 2))
        return f"{pos[0] // grid}_{pos[1] // grid}"

    def get_heuristic_score(self, next_pos, goal_pos, prev_pos=None, **kwargs):
        if prev_pos is None:
            dist = math.hypot(next_pos[0]-goal_pos[0], next_pos[1]-goal_pos[1])
            w, h = self.env.get_resolution()
            return max(0.0, 1.0 - (dist / math.hypot(w, h)))
            
        dist_prev = math.hypot(prev_pos[0] - goal_pos[0], prev_pos[1] - goal_pos[1])
        dist_next = math.hypot(next_pos[0] - goal_pos[0], next_pos[1] - goal_pos[1])
        step = kwargs.get('step_size', 30)
        return (dist_prev - dist_next) / step

    def predict_next_pos(self, curr_pos, action, **kwargs):
        step = kwargs.get('step_size', 30)
        x, y = curr_pos
        dx, dy = 0, 0
        if action == 'U': dy = -step
        elif action == 'D': dy = step
        elif action == 'L': dx = -step
        elif action == 'R': dx = step
        return [int(x+dx), int(y+dy)]

    def execute_action(self, env, curr_pos, action, **kwargs):
        step = kwargs.get('step_size', 30)
        next_pos = self.predict_next_pos(curr_pos, action, step_size=step)
        
        current_map_path = env.history[-1]
        crop_path = env.crop_local(current_map_path, next_pos, action, size=step)
        
        draw_res = env.draw_point(next_pos, color='blue', radius=int(step/4))
        
        prev_img = env.history[-2] if len(env.history) >= 2 else env.history[-1]
        
        return {
            'status': draw_res['status'], 
            'curr_img': draw_res['curr_img'], 
            'prev_img': prev_img, 
            'crop_img': crop_path 
        }

    # --- Mode: Direct ---
    def run_direct_baseline(self):
        print(f"\n--- Phase 3: Direct Mode (One-Shot) ---")
        # Ask VLM directly, without history, only the current map
        # prompt_text is already the full question (including A,B,C,D options)
        prompt = f"{self.prompt_text}\nPlease analyze the image and answer the question directly. The format of the output MUST BE 'Answer: ' "
        
        # Use temperature=0.0 for the most deterministic answer
        response, _ = self._chat(prompt, [], temperature=0.0)

        print(f"Direct Response received.")
        return response, "Direct Mode"

    def run(self):
        print(f"\n{'='*20} Task {self.task_id} | Type: {self.search_type} {'='*20}", flush=True)

        if self.search_type == 'direct':
            return self.run_direct_baseline()

        # Reset trackers
        if vlm_tracker:
            vlm_tracker.reset()
        self.score_distributions = []

        # Beam / V-LABS Search
        start_pos, goal_pos, step_size = self.get_maze_specs()
        self.env.draw_point(start_pos, color='blue', radius=int(step_size/3))
        result = self.run_search_loop(self.env, start_pos, goal_pos, ['U', 'D', 'L', 'R'], step_size=step_size)

        if result is None:
            return "Fail", ""
        best_node, first_step_uncertain = result
        if best_node:
            path_str = "".join(best_node.path_actions)

            # Check if goal was reached
            dist = math.hypot(best_node.curr_pos[0]-goal_pos[0], best_node.curr_pos[1]-goal_pos[1])
            reached_goal = dist < step_size * 1.5

            if reached_goal:
                final_prompt = self.prompts.BEAM_SOLVE_VISUAL_FINAL.format(
                    question=self.prompt_text, img_original=self.local_map_path,
                    img_trace=self.env.history[-1], found_path_str=path_str
                )
            else:
                final_prompt = self.prompts.PREDICT_COMPLETE_PATH.format(
                    img_map=self.local_map_path,
                    img_trace=self.env.history[-1],
                    start_pos=start_pos, goal_pos=goal_pos,
                    explored_path=path_str, current_pos=best_node.curr_pos,
                    question=self.prompt_text
                )
            resp, _ = self._chat(final_prompt, [])

            # Save detailed search log
            vlm_stats = vlm_tracker.get_stats() if vlm_tracker else {}
            log_file = os.path.join(self.task_dir, "search_log.json")
            log_data = {
                "task_id": self.task_id,
                "start_pos": list(start_pos),
                "goal_pos": list(goal_pos),
                "step_size": step_size,
                "reached_goal": reached_goal,
                "path": path_str,
                "best_node": {
                    "curr_pos": list(best_node.curr_pos),
                    "depth": best_node.depth,
                    "cumulative_score": best_node.cumulative_score,
                    "scores": {
                        "prior": best_node.score_prior,
                        "observer": best_node.score_obs,
                        "heuristic": best_node.score_h,
                    }
                },
                "score_distributions": self.score_distributions,
                "vlm_stats": vlm_stats,
                "config": {
                    "beam_width": self.beam_width,
                    "max_depth": self.max_depth,
                    "token_set": self.token_set,
                    "mu_threshold": self.mu_threshold,
                    "no_heuristic": self.no_heuristic,
                    "entropy_skip_threshold": self.entropy_skip_threshold,
                }
            }
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

            return resp, path_str
        return "Fail", ""

# =========================================================================
# Grid Navigation Agent
# =========================================================================
class GridNavAgent(UnifiedSearchAgent):
    def __init__(self, task_input, output_dir, model_name, prompts_class=None, **kwargs):
        super().__init__(output_dir, model_name, kwargs.get('beam_width', 3), kwargs.get('max_depth', 30), kwargs.get('verbose', True), prompts_class=prompts_class, search_type=kwargs.get('search_type', 'beam'))
        self.task_name = os.path.basename(task_input.rstrip('/'))
        self.task_dir = os.path.join(output_dir, self.task_name)
        if os.path.exists(self.task_dir): shutil.rmtree(self.task_dir)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copytree(task_input, self.task_dir, dirs_exist_ok=True)
        self.env = GridMazeEnv(self.task_dir)
        self.log_path = os.path.join(self.task_dir, 'output.json')

    # --- Initialization Methods (VLM) ---
    def calibrate_grid(self):
        print("\n--- Phase 1: Grid Calibration ---", flush=True)
        img_w, img_h = self.env.get_resolution()
        messages = []
        prompt_gen = self.prompts.GRID_SIZE_GEN.replace("{configuration_img_path}", self.env.map_path)
        
        for _ in range(MAX_TRIES_INIT):
            resp, messages = self._chat(prompt_gen, messages)
            parsed = parse_json_from_response(resp)
            if not parsed['status']: continue
            rows = int(parsed['content'].get('rows', 0))
            cols = int(parsed['content'].get('cols', 0))
            if rows <=0 or cols <=0: continue
            ratio = (img_w / cols) / (img_h / rows)
            math_info = f"Res: {img_h}x{img_w}. Guess: {rows}x{cols}. Ratio: {ratio:.2f}"
            print(math_info)
            c_resp, _ = self._chat(self.prompts.GRID_CRITIC.format(img_path=self.env.map_path, math_info=math_info), [])
            c_parsed = parse_json_from_response(c_resp)
            if 'frozen' in self.task_name: # TODO
                if img_w / cols != 64 or img_h / rows != 64 :
                    continue
            if c_parsed['status'] and c_parsed['content'].get('is_correct'):
                print(f"✅ Grid Confirmed: {rows}x{cols}", flush=True)
                return rows, cols
                # return img_h // 64, img_w // 64
            else:
                reason = c_parsed['content'].get('feedback', 'Math mismatch') if c_parsed['status'] else "Critic Error"
                messages.append({"role": "user", "content": f"Critic Rejected: {reason}. {math_info}"})
        return 4, 4

    def localize_agents(self, rows, cols):
        print("\n--- Phase 2: Localization ---")
        map_path = self.env.map_path
        prompt_txt = Prompts.LOCATION_GEN.format(img_path=map_path, rows=rows, cols=cols)
        
        messages = []
        confirmed_start, confirmed_goal = None, None
        
        for _ in range(MAX_TRIES_INIT):
            if confirmed_start or confirmed_goal:
                hint = "Update:\n"
                if confirmed_start: hint += f"Start_pos at {confirmed_start} is CORRECT. KEEP IT.\n"
                if confirmed_goal: hint += f"Goal_pos at {confirmed_goal} is CORRECT. KEEP IT.\n"
                messages.append({"role": "user", "content": hint})
            
            resp, messages = self._chat(prompt_txt, messages)
            parsed = parse_json_from_response(resp)
            
            if not parsed['status']: continue
                
            data = parsed['content']
            s_pos = confirmed_start if confirmed_start else data.get('start_pos')
            g_pos = confirmed_goal if confirmed_goal else data.get('goal_pos')
            
            # Crop & Verify
            s_crop = self.env.crop_cell(map_path, rows, cols, s_pos)
            g_crop = self.env.crop_cell(map_path, rows, cols, g_pos)
            
            if not s_crop or not g_crop:
                messages.append({"role": "user", "content": "Coordinates out of bounds."})
                continue
                
            critic_prompt = Prompts.LOCATION_CRITIC.format(start_crop=s_crop, goal_crop=g_crop)
            c_resp, _ = self._chat(critic_prompt, [])
            c_res = parse_json_from_response(c_resp)
            
            if c_res['status']:
                c_data = c_res['content']
                print(c_data)
                if c_data.get('start_correct'): confirmed_start = s_pos
                if c_data.get('goal_correct'): confirmed_goal = g_pos
                
                if confirmed_start and confirmed_goal:
                    print(f"✅ Locations: Start={s_pos}, Goal={g_pos}")
                    return s_pos, g_pos
                
                fb = "Check icons."
                if not c_data.get('start_correct'): fb += f" Start at {s_pos} is WRONG."
                if not c_data.get('goal_correct'): fb += f" Goal at {g_pos} is WRONG."
                messages.append({"role": "user", "content": fb})
            
            else:
                pass
        
        return s_pos, g_pos

    # --- Search Overrides ---
    def get_state_key(self, pos, **kwargs):
        return f"{pos[0]}_{pos[1]}"

    def get_heuristic_score(self, next_pos, goal_pos, prev_pos=None, **kwargs):
        if prev_pos is None: return 0.0

        if next_pos == goal_pos:
            return 100
        
        dist_prev = abs(prev_pos[0] - goal_pos[0]) + abs(prev_pos[1] - goal_pos[1])
        dist_next = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
        
        return float(dist_prev - dist_next) * 0.1

    def predict_next_pos(self, curr_pos, action, **kwargs):
        r, c = curr_pos
        dr, dc = 0, 0
        if action == 'Up': dr = -1
        elif action == 'Down': dr = 1
        elif action == 'Left': dc = -1
        elif action == 'Right': dc = 1
        return [r+dr, c+dc]

    def execute_action(self, env, curr_pos, action, **kwargs):
        rows, cols = kwargs.get('rows'), kwargs.get('cols')
        next_pos = self.predict_next_pos(curr_pos, action)
        
        current_map_path = env.history[-1] 
        crop_path = env.crop_cell(current_map_path, rows, cols, next_pos)
        
        if crop_path is None:
            return {'status': False, 'message': 'Crop failed (out of bounds)'}

        exec_res = env.step_swap(rows, cols, curr_pos, next_pos)
        
        if not exec_res['status']: 
            return {'status': False}
        
        return {
            'status': True, 
            'curr_img': exec_res['curr_img'], 
            'prev_img': exec_res['prev_img'], 
            'crop_img': crop_path 
        }

    # =========================================================================
    # Mode: direct (Direct Prediction)
    # =========================================================================
    def run_direct_baseline(self, rows, cols, start_pos, goal_pos):
        print(f"\n--- Phase 3: Direct Mode (One-Shot) ---")
        
        # 1. Construct Prompt
        # Assume Prompts.DIRECT_NAV_SOLVE exists; if not, add it in prompt.py
        if hasattr(self.prompts, 'DIRECT_NAV_SOLVE'):
            prompt = self.prompts.DIRECT_NAV_SOLVE.format(
                rows=rows, cols=cols,
                start_pos=start_pos,
                goal_pos=goal_pos,
                img_path=self.env.map_path # initial map
            )
        else:
            # Fallback prompt if not defined in Prompts class
            prompt = f"The grid is {rows}x{cols}. Start at {start_pos}, Goal is {goal_pos}. " \
                     f"Provide the path as a JSON list of actions (Up, Down, Left, Right). Image: {self.env.map_path}"
        
        # 2. Call LLM
        # Use temperature=0.0 for the most deterministic answer
        response, _ = self._chat(prompt, [], temperature=0.0)

        # 3. Parse Result
        parsed = parse_json_from_response(response)
        
        full_path = []
        thought = "Parse Failed"
        
        if parsed['status']:
            content = parsed['content']
            full_path = content.get('full_path', [])
            thought = content.get('thought', '')
            print(f"✅ Predicted Path: {full_path}")
        else:
            print(f"❌ Parse Error: {response}")

        # 4. Save Logs Directly
        log_data = {
            "type": "direct",
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "path": full_path,
            "thought": thought,
            "raw_response": response
        }
        
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
    
    def run(self):
        try:
            print(f"Task: {self.task_name} | Type: {self.search_type}", flush=True)
            rows, cols = self.calibrate_grid()
            s_pos, g_pos = self.localize_agents(rows, cols)

            if self.search_type == 'direct':
                self.run_direct_baseline(rows, cols, s_pos, g_pos)
            else:
                # Reset trackers
                if vlm_tracker:
                    vlm_tracker.reset()
                self.score_distributions = []

                best_node, first_step_uncertain = self.run_search_loop(self.env, s_pos, g_pos, ['Up', 'Down', 'Left', 'Right'], rows=rows, cols=cols)
                if best_node:
                    explored_path = best_node.path_actions
                    reached_goal = (best_node.curr_pos == g_pos)

                    if not reached_goal:
                        # Ask VLM to predict remaining path
                        prompt = (
                            f"You are navigating a {rows}x{cols} grid from {s_pos} to {g_pos}.\n"
                            f"Path explored so far: {explored_path}\n"
                            f"Current position: {best_node.curr_pos}\n\n"
                            f"Map: <img src='{self.env.map_path}'>\n\n"
                            f"Predict the remaining moves (Up/Down/Left/Right) to reach the goal {g_pos}.\n"
                            f"Output ONLY a JSON list of the COMPLETE path (explored + remaining), e.g.:\n"
                            f'["Up", "Right", "Right", "Down"]'
                        )
                        resp, _ = self._chat(prompt, [])
                        parsed = parse_json_from_response(resp)
                        if parsed['status'] and isinstance(parsed['content'], list):
                            explored_path = parsed['content']

                    print(f"🏆 Best Path: {explored_path}", flush=True)

                    # Save detailed log (similar to visual search search_log.json)
                    vlm_stats = vlm_tracker.get_stats() if vlm_tracker else {}
                    log_data = {
                        "task": self.task_name,
                        "start_pos": list(s_pos),
                        "goal_pos": list(g_pos),
                        "grid_size": [rows, cols],
                        "reached_goal": reached_goal,
                        "path": explored_path,
                        "best_node": {
                            "curr_pos": list(best_node.curr_pos),
                            "depth": best_node.depth,
                            "cumulative_score": best_node.cumulative_score,
                            "scores": {
                                "prior": best_node.score_prior,
                                "observer": best_node.score_obs,
                                "heuristic": best_node.score_h,
                            }
                        },
                        "score_distributions": self.score_distributions,
                        "vlm_stats": vlm_stats,
                        "config": {
                            "beam_width": self.beam_width,
                            "max_depth": self.max_depth,
                            "token_set": self.token_set,
                            "mu_threshold": self.mu_threshold,
                            "no_heuristic": self.no_heuristic,
                            "entropy_skip_threshold": self.entropy_skip_threshold,
                        }
                    }
                    with open(self.log_path, 'w') as f:
                        json.dump(log_data, f, indent=2)
                return first_step_uncertain
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

# =========================================================================
# Frozen Lake Agent (New)
# =========================================================================
class FrozenLakePromptAdapter(Prompts):
    GRID_SIZE_GEN = Prompts.FROZEN_GRID_SIZE_GEN
    LOCATION_GEN = Prompts.FROZEN_LOCATION_GEN
    LOCATION_CRITIC = Prompts.FROZEN_LOCATION_CRITIC
    PRIOR_CHECK_PROMPT = Prompts.FROZEN_PRIOR_CHECK_PROMPT
    OBSERVER_CHECK_PROMPT = Prompts.FROZEN_OBSERVER_CHECK_PROMPT
    # Assume Frozen Lake uses the generic Direct Prompt or defines its own
    # DIRECT_NAV_SOLVE = Prompts.FROZEN_DIRECT_SOLVE if exists

class FrozenLakeAgent(GridNavAgent):
    def __init__(self, task_input, output_dir, model_name, **kwargs):
        super().__init__(task_input, output_dir, model_name, prompts_class=FrozenLakePromptAdapter, **kwargs)