class Prompts:

    # ========== Grid Mode Prompts (Navigation) ==========
    
    GRID_SIZE_GEN = """
Visual Analysis Task:
Before we start navigation, you must accurately determine the Grid Size (rows x columns) of this map.

**Instructions**:
1. Look at the map image carefully.
2. Count the number of **Row** (from the upper to the bottom) .
3. Count the number of **Columns** (from left to right).

Map:
<img src='{configuration_img_path}'>

Output a JSON object with your count:
```json
{{
    "rows": <int>,
    "cols": <int>,
    "reasoning": "I counted X rows by... and Y columns by..."
}}
"""
    
    GRID_CRITIC = """
You are a Map Grid Auditor.
An agent has analyzed this map and proposed a Grid Size.
You need to verify this proposal using both Visual Counting and Resolution Mathematics.
Mathematical Analysis:
{math_info}
Visual Analysis:
Map:
<img src='{img_path}'>
Evaluation Task:
Check the Math:
Does the "Cell Aspect Ratio" in Mathematical Analysis look close to 1.0? (Maze cells are usually squares). If not, change the number of row and col in the Mathematical Analysis.

Decision:
If the Math suggests non-square cells, reject it.
If the Math looks like Square cells (the ratio is about 1, like between 0.95 and 1.05), accept it.
If the Math not a Square cells (like 0.9 or 1.1), reject it.
Output JSON:
```json
{{
    "is_correct": <bool>,
    "math_check_comment": "<Does the cell size/ratio make sense?>",
    "feedback": "<Final conclusion>"
}}
"""
    
    LOCATION_GEN = """
**Phase 2: Agent Localization**

**Context**:
We have confirmed the map is a **{rows}x{cols} grid** ({rows} rows by {cols} columns).

**Objective**:
Identify the exact coordinates of the starting point and the destination on the map.
1. The **Home** icon represents the **Start Position** (`start_pos`).
2. The **Office** icon represents the **Goal Position** (`goal_pos`).

**Map Image**:
<img src='{img_path}'>

**Instructions**:
1. **Coordinate System**: Use [row, col] format, 0-indexed. Top-Left is [0, 0].
2. **Visual Alignment**: Mentally divide the image into a {rows}x{cols} grid. Carefully check which specific cell contains the center of the Home icon and the Office icon.
3. **Constraint**: The `start_pos` and `goal_pos` must be **two different locations**.
4. **Output**: Return a strictly valid JSON object. Do not include any conversational text outside the JSON.

**JSON Output Format**:
```json
{{
    "thought": "I can see the grid is {rows}x{cols}. The Home icon is located in the ... row and ... column. The Office icon is located in...",
    "start_pos": [<row_index>, <col_index>],
    "goal_pos": [<row_index>, <col_index>]
}}
"""
    
    LOCATION_CRITIC = """
You are a Location Auditor.
I have cropped two cells based on the agent's coordinates.
Image 1 (Candidate for start Home):
<img src='{start_crop}'>
Image 2 (Candidate for goal Office):
<img src='{goal_crop}'>
Verification Task:
Does Image 1 clearly contain the Home icon? If yes, then the start_correct is True. Else, the start_correct is False.
Does Image 2 clearly contain the goal Office icon? If yes, then the goal_correct is True. Else, the goal_correct is False.
Output JSON:
```json
{{
    "start_correct": <bool>,
    "goal_correct": <bool>,
    "feedback": "<Explain if icons are missing or cut off>"
}}
"""
    
    NAV_INIT = """
Visual Navigation Task (Tree Search Mode):
Grid Size: {rows}x{cols}
Home: {start_pos}
Office: {goal_pos}
Goal: Navigate to Office.
Strategy:
Identify ALL possible moves. Do not turn back immediately.
Map:
<img src='{img_path}'>
Please output ALL valid Next Actions available from your current position.
Status:
Grid Size: {rows}x{cols}
Home (Fixed): {start_pos}
Office (Fixed): {goal_pos}
Current Pos (Dynamic): {start_pos}
Previous Pos: "None"
Constraints:
Unidirectional: Move in one direction (Up/Down/Left/Right).
Obstacle Avoidance: DO NOT land on or cross walls or obstacles.
Move one step: Once you choice a valid direction, please move one step and make sure you will not attack the Wall or Obstacle.
NO BACKTRACKING: Do NOT move back to Previous_Pos immediately.
Thinking Process (Mandatory):
Before generating the JSON, you must analyze the scene step-by-step:
Analyze Neighbors: Check all 4 directions (Up, Down, Left, Right) from {start_pos}.
Is the direction near a wall or obstacle?
Is it out of bounds?
Calculate End Points: For each VALID direction, calculate the stopping coordinate based on the "Cautious Step" rule (make sure you will not hit anything).
Evaluation: Compare these end points. Which one helps explore new areas?
Task:
List valid directions. If a direction is blocked by obstacle or map immediately, do not list it.
Output STRICTLY in the following JSON format:
```json
{{
    "current_pos": {start_pos},
    "grid_size": [{rows}, {cols}],
    "candidates": [
        {{
            "direction": "Up",
            "next_pos": [<row>, <col>],
            "reason": "Stops at row X because of wall at X-1"
        }},
        {{
            "direction": "Right",
            "next_pos": [<row>, <col>],
            "reason": "Reaches map edge"
        }}
    ]
}}
"""
    
    NAV_STEP = """
Please output ALL valid Next Actions available from your current position.

Status:
Grid Size: {rows}x{cols}
Office (Fixed): {goal_pos}
Current Pos (Dynamic): {current_pos}

Constraints:
Unidirectional: Move in one direction (Up/Down/Left/Right).
Obstacle Avoidance: DO NOT land on or cross walls or obstacles.
Move one step: Once you choice a valid direction, please move one step and make sure you will not attack the Wall or Obstacle.

Thinking Process (Mandatory):
Before generating the JSON, you must analyze the scene step-by-step:
Analyze Neighbors: Check all 4 directions (Up, Down, Left, Right) from {current_pos}.
Is the direction near a wall or obstacle?
Is it out of bounds?

Task:
List valid directions. If a direction is blocked by obstacle or map immediately, do not list it.
Output STRICTLY in the following JSON format:
```json
{{
    "current_pos": [<row>, <col>],
    "grid_size": [{rows}, {cols}],
    "candidates": [
        {{
            "direction": "Up",
            "next_pos": [<row>, <col>],
            "reason": "Stops at row X because of wall at X-1"
        }},
        {{
            "direction": "Right",
            "next_pos": [<row>, <col>],
            "reason": "Reaches map edge"
        }}
    ]
}}
"""
    
    NAV_OBSERVER = """
You are a rigorous Visual Maze Adjudicator.
Your task is to evaluate the validity of a navigation move based on visual evidence.

Visual Evidence:
Previous Map: <img src='{prev_img}'>
Target Cell Close-up: <img src='{target_crop}'>
Current Map: <img src='{curr_img}'> (After potential move)

Context Info:
Agent's New Coordinate: {curr_pos}
Goal Coordinate: {goal_pos}

CRITICAL EVALUATION STEPS:

Step 1: Terrain Safety Check (The most important check)
Focus: Look strictly at the **Target Cell Close-up**.
Note: This image shows the terrain BEFORE the agent lands there.
Question: Is this cell a Wall (🚧) or a clear Road (⬜)?
- If it is a Black Wall / Obstacle: The move is INVALID. Score = 0.
- If it is a White Path / Floor: The move is VALID.

Step 2: Map Integrity Check
Focus: Compare Previous Map vs Current Map.
Verdict: If the map looks broken/shifted, validity_score = 0.

Output Format:
Provide your judgment in the following strict JSON format:
```json
{{
    "validity_score": <float, 0.0 if wall/broken, 1.0 if valid road>,
    "goal_reached": <0 or 1>,
    "reasoning": "<I see a black wall in the close-up... / I see a clear white path...>"
}}
"""
    
    DIRECT_NAV_SOLVE = """
Visual Navigation Task:
You are an expert navigator.
You are given a map with a grid structure.

**Information**:
- Grid Size: {rows} Rows x {cols} Columns.
- Start Position (Home or Person): {start_pos} (Row, Col)
- Goal Position (Office or Gift): {goal_pos} (Row, Col)

**Map**:
<img src='{img_path}'>

**Task**:
Plan a **complete, step-by-step path** from Home to Office.
- Avoid walls and obstacles.
- Do not go out of bounds.
- Use only directions: "Up", "Down", "Left", "Right".

**Output**:
Return a JSON object containing the full list of actions to reach the goal.
```json
{{
    "thought": "I see the path starts at... then I need to go around the wall...",
    "full_path": ["Up", "Right", "Right", "Up", "Left", ...]
}}
"""
    
    # ========== Pixel Mode Prompts (Maze) ==========
    
    DIRECT_SOLVE = """
Image:
<img src='{img_path}'>

Question:
{question}

Output your answer clearly, e.g., "Answer: A".
"""
    
    MAZE_SPECS = """
Visual Analysis Task:
Analyze the Maze image layout.

Tasks:
1. Locate the **Center** of the Start Point (Red Ball/Circle).
2. Locate the **Center** of the Goal Point (Green Ball/Circle).
3. Estimate the **Corridor Width** (Step Size).

**CRITICAL INSTRUCTION ON COORDINATES**:
Do NOT output raw pixel coordinates.
Output **Normalized Coordinates** in the range **[0, 1000]**.
(0,0) is Top-Left, (1000,1000) is Bottom-Right.

For Step Size, estimate it relative to the image width (0-1000 scale).

Image:
<img src='{img_path}'>

Output JSON:
```json
{{
    "start_norm": [<x_0to1000>, <y_0to1000>],
    "goal_norm": [<x_0to1000>, <y_0to1000>],
    "step_size_norm": <int_0to1000>,
    "reasoning": "..."
}}
"""
    
    NAV_BATCH_OBSERVER = """
You are a precision Maze Navigation Agent. Your sole task is to evaluate the physical traversability of the potential next steps.

### Visual Definitions:
1. **Blue Dot / Marker**: This represents the agent's position AFTER taking a specific step (Up, Down, Left, or Right).
2. **Black/Dark Areas**: These are **WALLS**. They are impassable obstacles.
3. **White/Light Areas**: These are **SAFE PATHS**.

### Your Task:
Carefully examine the candidate images provided below. For each option, perform a "Collision Test":
1. Look at the **Blue Dot** in the center of the crop.
2. **CRITICAL CHECK**: Does the Blue Dot overlap with, touch, or get dangerously close to a Black Wall?
   - If the Blue Dot overlaps a black line or sourronded by the black area: The move is INVALID. **Score = 0.0**.
   - If the Blue Dot partly overlap with the black area: The move is RISKY. **Score = 0.2 to 0.4**.
   - If the Blue Dot is clearly on white space with margin: The move is SAFE. **Score = 1.0**.

Candidate Images:
{candidate_images_block}

### Output Format:
Provide a JSON object. For each direction, provide:
- "collision_check": A short string describing if the dot touches a wall (e.g., "Clear", "Touching Wall", "Near Wall").
- "validity_score": A float from 0.0 to 1.0 based on the safety rules above.

```json
{{
    "U": {{
        "collision_check": "...",
        "validity_score": <float>
    }},
    "D": {{
        "collision_check": "...",
        "validity_score": <float>
    }},
    ...
}}
"""
    
    BEAM_SOLVE_VISUAL_FINAL = """
I have performed an automated pathfinding search on the maze.
Here is the gived partion paths I found physically valid:
**"{found_path_str}"**

I need you to match this sequence with the options provided in the question.

Original Question:
{question}

Instructions:
1. Read the options (A, B, C, D...) in the Original Question carefully. They usually describe a entire sequence of moves (e.g., "R, R, D...").
2. Compare the partion path "{found_path_str}" with the beginning of each option.
3. Find the option whose begining path is similar with the given partion path.

Output your decision clearly.
- If a similar match is found, output e.g., "Answer: A".
- If no similar match is found, output "Answer: F".
"""

    # Used when search depth is exhausted before reaching the goal.
    # Asks VLM to predict the complete path based on partial exploration + map.
    PREDICT_COMPLETE_PATH = """I have partially explored a maze/grid navigation task.

Map image:
<img src='{img_map}'>

Current exploration trace:
<img src='{img_trace}'>

Known information:
- Start position: {start_pos}
- Goal position: {goal_pos}
- Path explored so far: {explored_path}
- Current position after exploration: {current_pos}

Based on the map and the partial path above, predict the COMPLETE path from start to goal.
Combine the already-explored steps with your best guess for the remaining steps.

Original Question:
{question}

Instructions:
1. The explored path "{explored_path}" is verified to be physically valid.
2. Look at the map to figure out the remaining route from {current_pos} to {goal_pos}.
3. Combine them into a full path and match with the options in the question.

Output your answer (e.g., "Answer: A").
"""
    
    # ========== V-LABS Prompts (shared by both modes) ==========
    
    VLABS_IMAGINE = """
Visual Imagination Task:
You are at Current Position: {curr_pos}.
You are considering moving: **{direction}** to Next Position: {next_pos}.

Current Map View:
<img src='{curr_img}'>

**Task**:
Before you move, IMAGINE what the visual result will be.
1. Will you hit a wall?
2. What will the cell at {next_pos} look like? (Road, Obstacle, Icon?)
3. If you move {direction}, how will the agent's position relative to the grid lines change?

Output JSON:
```json
{{
    "prediction": "I expect to land on a white road cell. To my right there is a wall...",
    "expected_safety": "Safe/Crash",
    "feature_description": "<Brief visual description of the target cell>"
}}
"""
    
    VLABS_CONSISTENCY = """
Visual Consistency Verification (System 2 Check):
We have executed the move **{direction}**.

1. **Agent's Imagination** (Before Move):
"{imagination}"

2. **Reality** (After Move):
<img src='{next_img_crop}'> (Target Cell Close-up)
<img src='{next_img_full}'> (Full Map)

**Task**:
Compare Imagination vs. Reality.
1. Did the agent actually land where it expected?
2. Is the cell type (Road/Wall) consistent with the prediction?
3. Does the move look valid visually?

Output JSON:
```json
{{
    "consistency_score": <float, 0.0 to 1.0>,
    "reality_check": "Match/Mismatch",
    "reasoning": "<Why does reality match/mismatch imagination?>"
}}
"""

    PRIOR_CHECK_PROMPT = """
Current Status:
- Position: {curr_pos}
- Goal: {goal_pos}
- Proposed Action: **{action}**

Map Image:
<img src='{img_path}'>

Task:
Analyze if the Proposed Action is valid and logical.
1. Is the direction BLOCKED by a wall immediately?
2. Does it move drastically away from the goal?

Question: Is moving "{action}" a valid and reasonable step to take right now?
Answer (Yes/No):"""

    # [Step 4: Observer] Observation scoring prompt (generic version)
    OBSERVER_CHECK_PROMPT = """
Action Execution Assessment:
- Action Taken: **{action}**
- Previous Position: {prev_pos}
- New Position: {curr_pos}
- Goal: {goal_pos}

Images:
1. Full Map (After Move): <img src='{curr_img}'>
2. Zoom-in View (Target Cell): <img src='{crop_img}'>

Task:
Verify the safety of the New Position.

**CRITICAL CHECK**:
Look at the Zoom-in View.
This shows the terrain of the target location.
- Is it a **WALL** (Black or Obstacle Area)? 
- Or is it a **SAFE PATH** (White/Light Area)?

The Agent is NOT explicitly shown in the zoom-in view. You must judge based on the background terrain.

Question: Is the target position clearly a SAFE road (not a wall)?
Answer (Yes/No):"""

    # ========== Frozen Lake Mode Prompts (new) ==========

    FROZEN_GRID_SIZE_GEN = """
Visual Analysis Task (Frozen Lake):
We are playing Frozen Lake.
1. Count the number of rows and columns of the grid with {grid_n}x{grid_n}.
The map usually contains distinct square tiles (Ice or Holes).

Map:
<img src='{configuration_img_path}'>

Output JSON:
```json
{{
    "rows": <int>,
    "cols": <int>,
    "reasoning": "..."
}}
"""

    FROZEN_LOCATION_GEN = """
Visual Analysis Task (Step 2):
The Grid Size is **{rows}x{cols}**.

**Task**:
Locate the **Start Position** (usually an Elf/Person 🧝 or marked 'S') and the **Goal Position** (usually a Gift/Frisbee 🎁 or marked 'G') on this frozen lake.
- Output the [row, col] coordinates (0-indexed).
- Top-Left is [0, 0].

**Map**:
<img src='{img_path}'>

Output JSON:
```json
{{
    "thought": "<I see the Elf at...>",
    "start_pos": [<row>, <col>],
    "goal_pos": [<row>, <col>]
}}
"""

    FROZEN_LOCATION_CRITIC = """
You are a Location Auditor.
Image 1 (Candidate for Start/Elf):
<img src='{start_crop}'>
Image 2 (Candidate for Goal/Gift):
<img src='{goal_crop}'>

Verification Task:
Does Image 1 clearly contain the Elf/Person/Start marker?
Does Image 2 clearly contain the Gift/Frisbee/Goal marker?

Output JSON:
```json
{{
    "start_correct": <bool>,
    "goal_correct": <bool>,
    "feedback": "..."
}}
"""

    FROZEN_PRIOR_CHECK_PROMPT = """
Current State:
- Position: {curr_pos}
- Goal: Row {goal_pos}
- Proposed Action: {action}

Map Image:
<image>\n

Task:
You are navigating on a frozen lake towards the Goal (G).
Verify if the proposed action is the **CORRECT** step to reach the goal.

Rules:
- The map has a unique path to the goal.
- You must move closer to the goal.
- Moving into a Hole (H), Wall, or moving backwards/away from the goal is INCORRECT.

Question: Is moving "{action}" the correct step to reach the goal?
Answer (Yes/No):
"""


    FROZEN_OBSERVER_CHECK_PROMPT = """
Road Surface Smoothness Check

Image:
<img src='{crop_img}'>

Task:
Determine whether the road surface is uniformly flat and continuous.

STRICT Judgment Rules:
- If there is ANY localized region that is visually distinct from the surrounding surface,
  including but not limited to:
  • circular or oval shapes
  • enclosed regions with clear boundaries
  • darker or brighter patches
  • bowl-like, sunken, or pit-like structures
  • puddles, holes, or depressions (even if frozen or smooth)
  → NO

- ONLY answer YES if the ENTIRE image shows a single, continuous, uniform, flat ice/snow surface
  with no isolated shapes, no boundaries, and no local structural variation.

Important:
A smooth-looking pit, frozen puddle, or circular ice hole is still NOT flat ground.

Question: Does the image show a completely uniform, flat surface with no localized structures?
Answer (Yes/No):
"""


