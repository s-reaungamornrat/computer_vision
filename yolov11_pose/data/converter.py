import numpy as np

def min_index(arr1:np.ndarray, arr2:np.ndarray):
    """Find a pair of indices with the shortest distance between two arrays of 2D points
    Args:
        arr1 (np.ndarray): A numpy array of shape (N,2) representing N 2D points
        arr2 (np.ndarray): A numpy array of shape (M,2) representing M 2D points
    Returns:
        idx1 (int): Index of the point in arr1 with the shortest distance
        idx2 (int): Index of the point in arr2 with the shortest distance
    """
    #     (Nx1x2-1xMx2)->(NxMx2)-sum->NxM
    dis=((arr1[:,None,:]-arr2[None,:])**2).sum(-1) 
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)
    
def merge_multi_segment(segments: list[list]):
    """Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment

    This function takes multiple polygon segments, finds the closest points between them, reorders and connects them, and out[ut a single
    continuous segmentation path.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like 
            [segmentation1, segmentation2, ...]
    Returns:
        (list[np.ndarray]): A list of connected segments represented as numpy array
    """
    s=[]
    segments=[np.array(i).reshape(-1, 2) for i in segments] # list of (N,2)
    idx_list=[[] for _ in range(len(segments))]

    # Record the indices with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2=min_index(segments[i-1], segments[i])
        idx_list[i-1].append(idx1) # keep track of where segment i-1 should connect to its neighbors
        idx_list[i].append(idx2)

    # Use two rounds to connect all the segments
    for k in range(2):
        # forward connection
        if k==0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indices, reverse the index of middle segments
                if len(idx)==2 and idx[0]>idx[1]:
                    idx=idx[::-1]
                    segments[i]=segments[i][::-1]
                # Rotate (np.roll) the segment so the connection point becomes the start
                segments[i]=np.roll(segments[i], -idx[0], axis=0) # so the row corresponding to idx[0] will be the first row
                segments[i]=np.concatenate([segments[i], segments[i][:1]]) # close the segment by repeating the first point
                # Deal with the first segment and the last one
                if i in {0, len(idx_list)-1}: s.append(segments[i]) # append the full segment for the first and last segment
                else: 
                    idx=[0, idx[1]-idx[0]]
                    s.append(segments[i][idx[0]:idx[1]+1]) # append only connecting portion for middle segments
        else: # backward: walk backward through the segments
            for i in range(len(idx_list)-1,-1,-1):
                if i not in {0,len(idx_list)-1}: # add the remanining unused points of middle segments so no ports are lost
                    idx=idx_list[i]
                    nidx=ans(idx[1]-idx[0])
                    s.append(segments[i][nidx:])
    return s

def coco80_to_coco91_class() -> list[int]:
    r"""Convert 80-index (val2014) to 91-index (paper).

    Returns:
        (list[int]): A list of 80 class IDs where each value is the corresponding 91-index class ID.

    Examples:
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]

    References:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
