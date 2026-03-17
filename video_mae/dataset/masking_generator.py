import numpy as np

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
        mask_per_frame=np.hstack([np.zeros(self.num_patches_per_frame-self.num_masks_per_frame), 
                                  np.ones(self.num_masks_per_frame)])
        np.random.shuffle(mask_per_frame)
        mask=np.tile(mask_per_frame, (self.frames, 1)) # (frames, grid_h*grid_w), e.g., (8, 14x14=196)
        return mask