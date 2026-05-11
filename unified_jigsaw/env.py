import os
import numpy as np
from PIL import Image

class JigsawEvaluator:
    """
    Efficient CV evaluator with precomputed cost matrix.
    """
    def __init__(self, tiles_cache):
        # Preprocess: convert to int16 to avoid overflow during computation
        self.tiles_data = [np.array(img.convert('RGB'), dtype=np.int16) for img in tiles_cache]
        self.num_tiles = len(tiles_cache)
        
        self.cost_matrix = np.zeros((self.num_tiles, self.num_tiles, 2), dtype=np.float32)
        self._precompute_all_costs()

    def _calculate_pairwise_cost(self, idx_a, idx_b, direction):
        img_a = self.tiles_data[idx_a]
        img_b = self.tiles_data[idx_b]
        
        # Edge-weighted computation (Robust Boundary Matching)
        if direction == 'horizontal': 
            edge_a = 0.6 * img_a[:, -1, :] + 0.4 * img_a[:, -2, :]
            edge_b = 0.6 * img_b[:, 0, :]  + 0.4 * img_b[:, 1, :]
        else: # vertical
            edge_a = 0.6 * img_a[-1, :, :] + 0.4 * img_a[-2, :, :]
            edge_b = 0.6 * img_b[0, :, :]  + 0.4 * img_b[1, :, :]
            
        diff = edge_a - edge_b
        rmse = np.sqrt(np.mean(np.square(diff)))
        return rmse

    def _precompute_all_costs(self):
        # Precompute all pairwise costs, O(N^2)
        for i in range(self.num_tiles):
            for j in range(self.num_tiles):
                if i == j: continue
                self.cost_matrix[i, j, 0] = self._calculate_pairwise_cost(i, j, 'horizontal')
                self.cost_matrix[i, j, 1] = self._calculate_pairwise_cost(i, j, 'vertical')

    def evaluate_permutation(self, permutation, grid_n, threshold=25.0):
        perm_0 = [x - 1 for x in permutation]
        total_cost = 0.0
        match_count = 0
        
        for idx in range(len(perm_0)):
            r, c = divmod(idx, grid_n)
            curr = perm_0[idx]
            
            # Check Right
            if c < grid_n - 1:
                right = perm_0[idx + 1]
                cost = self.cost_matrix[curr, right, 0]
                if cost < threshold: match_count += 1
                total_cost += cost

            # Check Bottom
            if r < grid_n - 1:
                bottom = perm_0[idx + grid_n]
                cost = self.cost_matrix[curr, bottom, 1]
                if cost < threshold: match_count += 1
                total_cost += cost

        max_matches = 2 * grid_n * (grid_n - 1)
        score = match_count / max_matches if max_matches > 0 else 0
        return score, total_cost

class JigsawEnvironment:
    def __init__(self, img_path: str, res: int, output_root: str):
        self.original_path = img_path
        self.img_name = os.path.basename(img_path)
        self.stem = os.path.splitext(self.img_name)[0]
        self.working_dir = os.path.join(output_root, self.stem)
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.current_img_path = os.path.join(self.working_dir, 'init.png')
        self.rows = res
        self.cols = res
        self.total_tiles = res * res
        
        # Load and Resize
        self.base_img = Image.open(self.original_path).convert('RGB')
        self.base_img = self.base_img.resize((900, 900), Image.Resampling.LANCZOS)
        self.base_img.save(self.current_img_path)
        
        self.tiles_cache = self._slice_tiles()
        self.evaluator = JigsawEvaluator(self.tiles_cache)  # Initialize CV evaluator

    def _slice_tiles(self):
        w, h = self.base_img.size
        cw, ch = w // self.cols, h // self.rows
        tiles = []
        for i in range(self.total_tiles):
            r, c = divmod(i, self.cols)
            box = (c*cw, r*ch, (c+1)*cw, (r+1)*ch)
            tiles.append(self.base_img.crop(box))
        return tiles

    def rearrange_tiles(self, permutation, tag="temp", is_simulation=True):
        # permutation is 1-based
        new_img = Image.new('RGB', self.base_img.size)
        w, h = self.base_img.size
        cw, ch = w // self.cols, h // self.rows

        for i, tid in enumerate(permutation):
            r, c = divmod(i, self.cols)
            tile_data = self.tiles_cache[tid - 1] # 1-based -> 0-based
            new_img.paste(tile_data, (c*cw, r*ch))
        
        fname = f"{tag}.jpg"
        path = os.path.join(self.working_dir, fname)
        new_img.save(path, quality=90)
        return path