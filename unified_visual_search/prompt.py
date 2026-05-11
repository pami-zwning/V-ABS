"""
Prompt Templates for Visual Search V2
"""

class VisualSearchPrompts:

    # ========== Phase 1: Target Extraction & Grounding ==========

    EXTRACT_TARGETS_WITH_RELATIONS = """
You are an expert visual assistant tasked with analyzing a visual question.

Your job is to:
1. Extract all key visual targets (nouns with descriptions) from the question
2. Identify spatial relationships between targets if any exist
3. Return a structured JSON format

Examples:

Q: "Is the bowl on the left or right side of the faucet?"
Output: ```json
{{
    "targets": ["bowl", "faucet"],
    "relation": "spatial",
    "relation_type": "left-right",
    "relation_pairs": [["bowl", "faucet"]]
}}
```

Q: "What is the color of the mailbox?"
Output: ```json
{{
    "targets": ["mailbox"],
    "relation": "none"
}}
```

Q: "Is the woman with a backpack on the left or right side of the person with a beanie?"
Output: ```json
{{
    "targets": ["woman with a backpack", "person with a beanie"],
    "relation": "spatial",
    "relation_type": "left-right",
    "relation_pairs": [["woman with a backpack", "person with a beanie"]]
}}
```

Q: "What is written on the white board in front of the girl in purple shorts?"
Output: ```json
{{
    "targets": ["white board", "girl in purple shorts"],
    "relation": "spatial",
    "relation_type": "front-back",
    "relation_pairs": [["white board", "girl in purple shorts"]]
}}
```

Q: "What kind of animal is in the poster?"
Output: ```json
{{
    "targets": ["animal", "poster"],
    "relation": "containment",
    "relation_pairs": [["animal", "poster"]]
}}
```

Current Question: "{question}"
Please analyze and provide the JSON output:"""

    # ========== Phase 2: Region Scoring ==========

    PRIOR_SCORE_WITH_CONTEXT = """Look at this image. It has been divided into sub-regions: {direction} is one of them.

Question: "{question}"
Target object: "{target}"

Based on the overall scene layout, colors, and spatial arrangement visible in this image, is the {direction} area the most promising location to find the "{target}"?

Answer Yes or No."""

    POSTERIOR_SCORE_CONTRASTIVE = """This is a cropped close-up view of a specific image region.

Question: "{question}"
Target object: "{target}"

Examine the details in this cropped region carefully. Can you clearly see the "{target}" in this image?

Answer Yes or No."""

    # ========== Phase 3: Self-Reflection ==========

    SELF_REFLECTION = """
Question: "{question}"

You have completed a visual search and arrived at these findings:
{search_summary}

Current answer confidence: {confidence}

Now, critically reflect:
1. Did I search in the most promising regions?
2. Could I have missed the target due to incorrect early decisions?
3. Are there unexplored regions that might contain better evidence?

If confidence is below 0.7, suggest an alternative search path:
```json
{{
    "should_backtrack": true/false,
    "alternative_region": "which region to try instead",
    "reasoning": "why this alternative might be better"
}}
```

Your response:"""

    # ========== Phase 4: Final Answer Generation ==========

    ANSWER_WITH_EVIDENCE_V2 = """
You are an intelligent visual question answering agent with enhanced reasoning capabilities.

Question: {question}

You have completed a hierarchical beam search and collected the following evidence:
{evidence_descriptions}

Visual Evidence Summary:
- Number of targets located: {num_targets}
- Search depth reached: {max_depth}
- Average confidence: {avg_confidence}
- Spatial relations verified: {relations_verified}

Based on the comprehensive evidence above, provide your final answer.

Instructions:
1. If multiple-choice, output the option letter (A/B/C/D)
2. If open-ended, provide a concise answer (1-5 words)
3. If uncertain, state your best guess with confidence level

Answer format:
```json
{{
    "answer": "your answer here",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation based on evidence"
}}
```

Your response:"""
