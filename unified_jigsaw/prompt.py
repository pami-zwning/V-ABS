class JigsawPrompts:
    # --- Policy Model: Generate candidate actions ---
    PREDICT_PERMUTATION = """
Visual Restoration Task:
You are an expert AI specialized in solving Jigsaw Puzzles.
You are given a current state of a {grid_n}x{grid_n} puzzle image.
The goal is to rearrange all tiles to restore the original perfect image.

Current Image State:
<img src='{img_path}'>

**Instruction**:
Output the tile_matrix ({grid_n}×{grid_n}) ordering that you consider reasonable to restore the image.
The output must contain all numbers from 1 to {max_idx} exactly once.

**Output JSON**:
```json
{{
    "candidates": [
        {{
            "tile_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "reasoning": "I see the horizon line connecting row 1..."
        }},
        ... (Produce {num_samples} distinct candidates)
    ]
}}
"""

    # --- Observer Model: Post-hoc scoring ---
    OBSERVER_SCORING = """
Visual Coherence Evaluator:
Evaluate the coherence of this jigsaw puzzle arrangement.
<img src='{curr_img}'>
Score criteria:
0.0: Chaotic, no matching borders.
0.5: Some parts connected, but global structure is wrong.
1.0: Perfectly restored image.
Output JSON:
code
JSON
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "The face is aligned but the background is fragmented..."
}}
"""

class DirectPrompts:
    # --- Direct Mode ---
    DIRECT_SOLVE = """
Instructions: The original image has been divided into {grid_n}x{grid_n} pieces and scrambled.
In the image, each piece is numbered from 1 to {grid_n}x{grid_n}.
Determine the correct permutation to restore the original image.
Image:
<img src='{img_path}'>
Output format:
Return ONLY a JSON object.
{{
"final_permutation": [1, 2, 3, ...],
"reasoning": "..."
}}
"""