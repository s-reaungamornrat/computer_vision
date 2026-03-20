import numpy as np

class Cell:
    def __init__(self, num_masks, num_patches):
        self.num_masks=num_masks
        self.num_patches=num_patches
        self.size=num_masks+num_patches
        self.queue=np.hstack([np.ones(num_masks, dtype=np.uint8), np.zeros(num_patches, dtype=np.uint8)])
        self.queue_ptr=0
        
    def set_ptr(self, pos=-1):
        self.queue_ptr=np.random.randint(self.size) if pos<0 else pos
        
    def get_cell(self):
        cell_idx=(np.arange(self.size)+self.queue_ptr)%self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr+=1

class RunningCellMaskingGenerator:
    """
    Input video is divided into a grid of patches (typically 16x16 pixels each). 
    Args:
        input_size (tuple[int, int, int]): A tuple of number of patches along frame, height, and width dimension. In other words, 
            grid cell dimension along frame, height, width
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/masking_generator.py#L56
    """
    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width=input_size # number of patches along frame, height, and width dimension
        self.mask_ratio=mask_ratio

        # consider a grid of 2x2 and divide it for mask and not-mask based on `mask_ratio`
        num_masks_per_cell=int(4*self.mask_ratio)
        assert 0<num_masks_per_cell<4
        num_patches_per_cell=4-num_masks_per_cell # for the 2x2, compute the not-mask cells

        self.cell=Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size=self.cell.size # 4 since it is 2x2 grid
        
        mask_list=[]
        for ptr_pos in range(self.cell_size): # 4 cells
            self.cell.set_ptr(ptr_pos) 
            mask=[]
            for _ in range(self.frames):
                self.cell.run_cell() 
                mask_unit=self.cell.get_cell().reshape(2,2)
                mask_map=np.tile(mask_unit, [self.height//2, self.width//2])
                mask.append(mask_map.flatten()) # (self.height*self.width,), typically (14*14,)=(196,)
            mask=np.stack(mask, axis=0) # (num_frames, self.height*self.width), typically, (8, 196)
            mask_list.append(mask)
        self.all_mask_maps=np.stack(mask_list, axis=0) # (num_cells,num_frames, self.height*self.width), typically, (4, 8, 196)

    def __repr__(self):
        return f"Running cell masking with mask ratio {self.mask_ratio}"

    def __call__(self):
        mask=self.all_mask_maps[np.random.randint(self.cell_size)]
        return np.copy(mask)
        
class TubeMaskingGenerator:
    """
    Input video is divided into a grid of patches (typically 16x16 pixels each). 
    Args:
        input_size (tuple[int, int, int]): A tuple of number of patches along frame, height, and width dimension. In other words, 
            grid cell dimension along frame, height, width
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/masking_generator.py#L56
    """
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width=input_size # number of patches along frame, height, and width dimension
        self.num_patches_per_frame=self.height*self.width # total number of patches in each frame, typically 14x14 grid cells
        self.total_patches=self.frames*self.num_patches_per_frame
        self.num_masks_per_frame=int(mask_ratio*self.num_patches_per_frame)
        self.total_masks=self.frames*self.num_masks_per_frame
        
    def __repr__(self):
        return f"Tube Masking : total patches {self.total_patches}, mask patches {self.total_masks}"
        
    def __call__(self):
        """
        Returns:
            (np.ndarray): Tube mask of shape (n_frames, n_grids) where n_frames is the temporal length for which the tube spans,
                and n_grid is the number of grid/patch cell in each time frame (grid_h*grid_w). Note grid_w is not the width of each
                patch and likewise for grid_h. grid_w is the number of patches along the x-direction, and the same for grid_h
        """
        mask_per_frame=np.hstack([np.zeros(self.num_patches_per_frame-self.num_masks_per_frame, dtype=np.uint8), 
                                  np.ones(self.num_masks_per_frame, dtype=np.uint8)])
        np.random.shuffle(mask_per_frame)
        mask=np.tile(mask_per_frame, (self.frames, 1)) # (frames, grid_h*grid_w), e.g., (8, 14x14=196)
        return mask