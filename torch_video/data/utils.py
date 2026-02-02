import os
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import torch

# def compute_timestamp_clip_sampling_params(video_duration, average_fps, frame_rate=8, clip_duration=2, step_duration=1):
#     """Compute parameters for clips_at_regular_timestamps and clips_at_random_timestamps
#     Args:
#         video_duration (float): Duration of the video in seconds
#         average_fps (float): Average frame rate of the original input video in frame per second
#         frame_rate (float): Desired/output frame rate
#         clip_duration (float): How long each clip captures in second
#         step_duration (float): Distance between the start of each clip in seconds.
#     Returns:
#         clip_start_times (torch.Tensor): Start times in seconds of each clip as 1D float tensor of size N where the number of clips
#         num_frames_per_clip (int): Number of frames per clip
#         seconds_between_frames (float): Distance between each frame in seconds
#     """
#     if video_duration<clip_duration: return None,None,None
def compute_clip_start_times(video_duration, clip_duration=2, step_duration=1):
    """Compute parameters for clips_at_regular_timestamps and clips_at_random_timestamps
    Args:
        video_duration (float): Duration of the video in seconds
        clip_duration (float): How long each clip captures in second
        step_duration (float): Distance between the start of each clip in seconds.
    Returns:
        clip_start_times (torch.Tensor): Start times in seconds of each clip as 1D float tensor of size N where the number of clips
    """
    if video_duration<clip_duration: return None
        
    # # Determine the interval between frames within a clipp
    # sampling_fps=frame_rate if frame_rate else average_fps
    # seconds_between_frames=1./sampling_fps
    
    # # Determine number of frames per clip
    # num_frames_per_clip=int(clip_duration*sampling_fps)
    
    # Calculate all possible clip-start time: we stop before the end to ensure the full clip can be extracted
    stop_time=max(0, video_duration-clip_duration)
    clip_start_times=torch.arange(0, stop_time, step_duration)
    
    return clip_start_times
    
def has_file_allowed_extension(filename:str, extension:Union[str, tuple[str, ...]])->bool:
    """Check if a file is an allowed extension
    Args:
        filename (str): Path to a file
        extensions (str|tuple[str,...]): File extensions to consider, must be lowercase
    Returns:
        (bool): True if the filename ends with one of the given extensions
    """
    return filename.lower().endswith(extension if isinstance(extension, str) else tuple(extension))

def find_classes(directory:Union[str, Path])->tuple[list[str], dict[str, int]]:
    """Find the class folders in a dataset. Assuming the input ``directory`` contains subdirectories whose
    names corresponding to the class names
    Args:
        directory (str|Path): A path to the directory containing subdirectories of videos of each class
    Returns:
        classes (list[str]): Sorted name of classes
        class_to_idx (dict[str, int]): Mapping between class names and class indices, starting from index=0
    """
    classes=sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes: raise FileNotFoundError(f"Could not find any class folder in {directory}")
    class_to_idx={cls_name:i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(directory:Union[str,Path], class_to_idx:Optional[dict[str,int]]=None, 
                 extensions:Optional[Union[str, tuple[str,...]]]=None,
                 is_valid_file:Optional[Callable[[str],bool]]=None, allow_empty:bool=False)->list[tuple[str,int]]:
    """Generate a list of samples of a form (path_to_sample, class).
    Args:
        directory (str|Pat): Path to directory containing subdirectories of class-specific videos
        class_to_idx (dict[str,int], optional): Mapping between class names and class indices. 
            If not provided, ``find_classes`` will be used to find it
        extensions (str|tuple[str,...]],optional): Video file extensions, such as avi, mp4
        is_valid_file (callable[[str],bool], optional): A function to check whether file is valid
        allow_empty (bool,optional): Whether to allow empty class, e.g., subfolders without videos
    Returns:
        (list[tuple[str|Path, int]]): A list of tuples of paths to video and class indices
    """
    directory=os.path.expanduser(directory)
    
    if class_to_idx is None: _, class_to_idx=find_classes(directory)
    elif not class_to_idx: raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
    
    both_none=extensions is None and is_valid_file is None
    both_something=extensions is not None and is_valid_file is not None
    if both_none or both_something: raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    
    if extensions is not None:
        def is_valid_file(x:str|Path)->bool: return has_file_allowed_extension(str(x) if isinstance(x, Path) else x, extensions)
    # for type checker (not for Python runtime), declaring that is_valid_file is a function that takes a string and returns a boolean
    is_valid_file=cast(Callable[[str],bool], is_valid_file) 
    
    instances=[]
    available_classes=set()
    for target_class in sorted(class_to_idx.keys()):
        class_index=class_to_idx[target_class]
        target_dir=os.path.join(directory, target_class)
        if not os.path.isdir(target_dir): continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path=os.path.join(root, fname)
                if is_valid_file(path):
                    item=path, class_index
                    instances.append(item)
                    if target_class not in available_classes: available_classes.add(target_class)
                        
    empty_classes=set(class_to_idx.keys())-available_classes
    if empty_classes and not allow_empty:
        msg=f"Found no valid file for the classes {', '.join(empty_classes)}"
        if extensions is not None:
            msg+=f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)
    return instances

def unfold(tensor:torch.Tensor, size:int, step:int, dilation:int=1)->torch.Tensor:
    """Similar to torch.unfold, but with the dilation and specialized for 1d tensor

    Returns all consecutive windows of `size` elements, with `step` between windows. The distance between each element in a window is given by 
    `dilation`. In other words, it returns a 2D tensor where each row is a window of length `size` taken from the input 1D tensor, spaced `step` elements 
    apart, and with elements inside each window spaced by `dilation`
    output[i, j] = tensor[i*step + j*dilation]
    
    Returns:
        (torch.Tensor): 2D tensors where each row represents a window of `size` elements, i.e., tensor of shape (N, size) where N is the
            number of windows and `size` is the size of each window
    """
    if tensor.dim()!=1: raise ValueError(f'tensor should have 1 dimension instead of {tensor.dim()}')
    o_stride=tensor.stride(0)
    numel=tensor.numel()
    # step*o_stride: moving between windows (row) jumps `step` elements
    # dilation*o_stride: moving within a window (columns) jumps `dilation` elements
    new_stride=(step*o_stride, dilation*o_stride)
    # new_size[0]: number of valid windows
    # new_size[1]: window length (size)
    # Understand (numel - (dilation * (size - 1) + 1)) // step + 1
    # - window_span=dilation*(size - 1)+1 : width of window on the original tensor (i.e., indices of window on the original tensor is
    #   starting from start, start+dilation, start+2*dilation, ..., start+dilation*(size - 1)
    #  A window of `size` elements has `size`-1 gaps and each gap has a length of `dilation` so total distance from the 1st element to the last element
    #  in a single window is window_span=dilation*(size-1)+1 [total gap length +1 element]
    # Original valid indices 0,...,numel-1
    # For a window to fit the original tensor
    # start+(window_span-1)<=numel-1  --> start<= numel-window_span --> max_start=numel-window_span 
    # Windows start at 0, 1*step, 2*step, ..., k*step where k*step<=max_start
    # Thus, maximum k is k_max=(max_start//step). Thus, maximum number of windows is k_max+1 since k counts from 0
    new_size=((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0]<1: new_size=(0, size)
    return torch.as_strided(tensor, new_size, new_stride)