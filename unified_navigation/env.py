import os
from PIL import Image, ImageDraw
from copy import deepcopy
import shutil

# ==========================
# Task 1 Environment: Pixel-based (Maze)
# ==========================
class PixelMazeEnv:
    def __init__(self, working_dir: str, map_name: str):
        self.working_dir = working_dir
        self.map_path = os.path.join(working_dir, map_name)
        self.current_img_path = os.path.join(self.working_dir, 'current_state.png')
        
        # Init Image
        if os.path.exists(self.map_path):
            img = Image.open(self.map_path).convert('RGB')
            img.save(self.current_img_path)
        else:
            raise FileNotFoundError(f"Map not found: {self.map_path}")
            
        self.step_cnt = 0
        self.history = [self.current_img_path]

    def get_resolution(self):
        with Image.open(self.map_path) as img:
            return img.size 

    def crop_local(self, img_path, pre_center, dir, size=100):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                cx, cy = pre_center
                
                half = size // 2
                left = max(0, cx - half)
                top = max(0, cy - half)
                right = min(w, cx + half)
                bottom = min(h, cy + half)
                
                crop = img.crop((left, top, right, bottom))
                
                save_name = f"crop_x{cx}_y{cy}_{dir}.png"
                save_path = os.path.join(self.working_dir, save_name)
                crop.save(save_path)
                return save_path
        except Exception as e:
            print(f"Crop error: {e}")
            return img_path 

    def draw_point(self, pos, color='blue', radius=5):
        prev_pth = self.history[-1]
        try:
            img = Image.open(prev_pth).convert('RGB')
            draw = ImageDraw.Draw(img, 'RGBA')
            
            x, y = pos
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=None)
            
            self.step_cnt += 1
            new_name = f"step_{self.step_cnt}_x{x}_y{y}.png"
            new_path = os.path.join(self.working_dir, new_name)
            img.save(new_path)
            
            self.history.append(new_path)
            return {
                'status': True,
                'curr_img': new_path
            }
        except Exception as e:
            return {'status': False, 'message': str(e)}

    def save_state(self):
        return {'history': deepcopy(self.history), 'cnt': self.step_cnt}

    def load_state(self, state):
        self.history = deepcopy(state['history'])
        self.step_cnt = state['cnt']


# ==========================
# Task 2 Environment: Grid-based (Navigation)
# ==========================
class GridMazeEnv:
    def __init__(self, task_path: str):
        self.task_path = task_path
        self.working_dir = task_path 
        self.map_path = os.path.join(task_path, "map.png")
        self.current_img_path = os.path.join(self.working_dir, 'current_state.png')
        
        # Init Image
        if os.path.exists(self.map_path):
            img = Image.open(self.map_path).convert('RGB')
            img.save(self.current_img_path)
        else:
            raise FileNotFoundError(f"Map not found: {self.map_path}")
            
        self.step_cnt = 0
        self.history = [self.current_img_path]

    def get_resolution(self):
        with Image.open(self.map_path) as img:
            return img.size # (width, height)

    def crop_cell(self, img_path, rows, cols, pos):
        """Crop a single cell for observation"""
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                cell_w, cell_h = w // cols, h // rows
                r, c = pos
                
                if not (0 <= r < rows and 0 <= c < cols):
                    return None
                    
                left, top = c * cell_w, r * cell_h
                crop = img.crop((left, top, left + cell_w, top + cell_h))
                
                save_name = f"crop_r{r}_c{c}_{os.path.basename(img_path)}"
                save_path = os.path.join(self.working_dir, save_name)
                crop.save(save_path)
                return save_path
        except Exception as e:
            print(f"Crop error: {e}")
            return None

    def step_swap(self, rows, cols, current_pos, next_pos):
        """Execute swap operation (specific to navigation task logic)"""
        prev_pth = self.history[-1]
        try:
            img = Image.open(prev_pth).convert('RGB')
            w, h = img.size
            cell_w, cell_h = w // cols, h // rows
            
            y1, x1 = current_pos
            y2, x2 = next_pos
            
            box1 = (x1 * cell_w, y1 * cell_h, (x1 + 1) * cell_w, (y1 + 1) * cell_h)
            box2 = (x2 * cell_w, y2 * cell_h, (x2 + 1) * cell_w, (y2 + 1) * cell_h)
            
            region1 = img.crop(box1).copy()
            region2 = img.crop(box2).copy()
            img.paste(region2, box1)
            img.paste(region1, box2)
            
            self.step_cnt += 1
            new_name = f"step_{self.step_cnt}_{y1}_{x1}_to_{y2}_{x2}.png"
            new_path = os.path.join(self.working_dir, new_name)
            img.save(new_path)
            
            self.history.append(new_path)
            return {
                'status': True,
                'curr_img': new_path,
                'prev_img': prev_pth
            }
        except Exception as e:
            return {'status': False, 'message': str(e)}

    def save_state(self):
        return {'history': deepcopy(self.history), 'cnt': self.step_cnt}

    def load_state(self, state):
        self.history = deepcopy(state['history'])
        self.step_cnt = state['cnt']