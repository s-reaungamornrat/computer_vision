from __future__ import annotations
from typing import Any

import cv2
import matplotlib.pyplot as plt

# Source : https://www.kaggle.com/code/gibsonxue/yolov8-skeleton-draw
bbox_color=(150,0,0)
bbox_thickness=6
# bbox_labelstr={'font_size':6, 'font_thickness':14, 'offset_x':0, 'offset_y':-80}
bbox_labelstr={'font_size':1, 'font_thickness':5, 'offset_x':0, 'offset_y':-80}

#Key point: BGR color scheme
kpt_color_map = {
    0:{'name':'Nose', 'color':[0, 0, 255], 'radius':10},                
    1:{'name':'Right Eye', 'color':[255, 0, 0], 'radius':10},           
    2:{'name':'Left Eye', 'color':[255, 0, 0], 'radius':10},            
    3:{'name':'Right Ear', 'color':[0, 255, 0], 'radius':10},           
    4:{'name':'Left Ear', 'color':[0, 255, 0], 'radius':10},            
    5:{'name':'Right Shoulder', 'color':[193, 182, 255], 'radius':10},  
    6:{'name':'Left Shoulder', 'color':[193, 182, 255], 'radius':10},   
    7:{'name':'Right Elbow', 'color':[16, 144, 247], 'radius':10},      
    8:{'name':'Left Elbow', 'color':[16, 144, 247], 'radius':10},       
    9:{'name':'Right Wrist', 'color':[1, 240, 255], 'radius':10},       
    10:{'name':'Left Wrist', 'color':[1, 240, 255], 'radius':10},       
    11:{'name':'Right Hip', 'color':[140, 47, 240], 'radius':10},       
    12:{'name':'Left Hip', 'color':[140, 47, 240], 'radius':10},       
    13:{'name':'Right Knee', 'color':[223, 155, 60], 'radius':10},      
    14:{'name':'Left Knee', 'color':[223, 155, 60], 'radius':10},      
    15:{'name':'Right Ankle', 'color':[139, 0, 0], 'radius':10},       
    16:{'name':'Left Ankle', 'color':[139, 0, 0], 'radius':10},         
}
kpt_labelstr={'font_size':1, 'font_thickness':2, 'offset_x':0, 'offset_y':150}

# Skeleton connection
skeleton_map = [
    {'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':5},       
    {'srt_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':5},         
    {'srt_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':5},         
    {'srt_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':5},         
    {'srt_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':5},     
    {'srt_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':5},        
    {'srt_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':5},      
    {'srt_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':5},       
    {'srt_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':5},          
    {'srt_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':5},         
    {'srt_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':5},         
    {'srt_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':5},        
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':5},        
    {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':5},         
    {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':5},         
    {'srt_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':5},       
    {'srt_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':5},          
    {'srt_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':5},         
    {'srt_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':5}         
]

def skeletons_overlay(img_bgr, results, output_filename=None, show=False):
    """
    Draw a skeleton of each set of detected keypoints
    Args:
        img_bgr (np.ndarry): Image array of size HxWxC in BGR format
        results (engine.results.Results): containing attrbutes
            - boxes (engine.results.Boxes) for bounding boxes
            - keypoints (engine.results.Keypoints) for keypoints
        output_filename (str): Path to save image. If None, show image using matplotlib
        show (bool): Whether to show output in plt.imshow
    """
    img_bgr=img_bgr.copy() # we do not want to modify the input image
    
    num_bbox=len(results.boxes.cls)
    # Convert to integer
    bboxes_xyxy=results.boxes.xyxy.cpu().numpy().astype(int) # Nx4
    # NxMx3 where N is the number of boxes and M is the number of points in each box
    bboxes_keypoints=results.keypoints.data.cpu().numpy().astype(int) 
    
    for idx in range(num_bbox):
        # Get the coordinates of the box
        bbox_xyxy=bboxes_xyxy[idx]
        # Get the predicted category of the bounding box (for keypoint detection, there is only one category i.e., person).
        bbox_label=results.names[0]
        img_bgr=cv2.rectangle(img_bgr,(bbox_xyxy[0],bbox_xyxy[1]),(bbox_xyxy[2],bbox_xyxy[3]),bbox_color,bbox_thickness)
        # Put text on the top-left corner
        img_bgr=cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), 
                            cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
    
        # The coordinates and confidence levels of all key points in this box.
        bbox_keypoints=bboxes_keypoints[idx] # Mx3
        # Draw the skeleton connection of this box
        for skeleton in skeleton_map:
            # Each skeleton containing only 1 of each keypoints, i.e., 1 left shoulder
            # Get starting point coordinates
            srt_kpt_id=skeleton['srt_kpt_id']
            srt_kpt_x=bbox_keypoints[srt_kpt_id][0] # scalar value
            srt_kpt_y=bbox_keypoints[srt_kpt_id][1] # scalar value
            # Get the coordinates of the termination point
            dst_kpt_id=skeleton['dst_kpt_id']
            dst_kpt_x=bbox_keypoints[dst_kpt_id][0] # scalar value
            dst_kpt_y=bbox_keypoints[dst_kpt_id][1] # scalar value
            # Get skeleton connection color
            skeleton_color=skeleton['color']
            # Get skeleton connection line width
            skeleton_thickness=skeleton['thickness']
            # Draw skeleton connection
            img_bgr=cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,thickness=skeleton_thickness)
        # Key points for drawing this box
        for kpt_id in kpt_color_map:
            # Obtain the color, radius, and XY coordinates of the key point.
            kpt_color=kpt_color_map[kpt_id]['color']
            kpt_radius=kpt_color_map[kpt_id]['radius']
            kpt_x=bbox_keypoints[kpt_id][0]
            kpt_y=bbox_keypoints[kpt_id][1]
            # Draw a circle: image, XY coordinates, radius, color, line width (-1 for fill).
            img_bgr=cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
            
    if output_filename is not None: cv2.imwrite(output_filename, img_bgr)
    elif show: plt.imshow(img_bgr[:,:,::-1])
    else: return img_bgr


def general_skeleton_overlay(img_bgr:np.ndarray, class_names:list[str], bboxes:np.ndarray, keypoints:np.ndarray):
    """Overlay skeleton on image
    Args:
        img_bgr (np.ndarray): Image array in BGR of size (H,W,3) where 3 is BGR channels
        class_names (list[str]): Class names of instances/objects, i.e., people
        bboxes (np.ndarray): Bounding box array in xyxy format in absolute coordinates in pixel unit of size (N, 4) where N is the number of boxes/instances
        keypoints (np.ndarray): Keypoint array in absolute coordinates in pixel units of size (N, 17,3) where N is the number of instances, 17 for landmarks
            on people and 3 for x,y,visibility
    Returns:
        (np.ndarray): Image with skeleton overlaid
    """
    assert len(class_names)==len(bboxes)==len(keypoints), 'Number of objects must be equal'
    
    img_bgr=img_bgr.copy() # we do not want to modify the input image
    
    num_bbox=len(bboxes)
    # Convert to integer
    bboxes_xyxy=bboxes.astype(int) # Nx4
    # NxMx3 where N is the number of boxes and M is the number of points in each box
    bboxes_keypoints=keypoints.astype(int) 
    
    for idx in range(num_bbox):
        # Get the coordinates of the box
        bbox_xyxy=bboxes_xyxy[idx]
        # Get the predicted category of the bounding box (for keypoint detection, there is only one category i.e., person).
        bbox_label=class_names[idx]
        img_bgr=cv2.rectangle(img_bgr,(bbox_xyxy[0],bbox_xyxy[1]),(bbox_xyxy[2],bbox_xyxy[3]),bbox_color,bbox_thickness)
        # Put text on the top-left corner
        img_bgr=cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), 
                            cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
    
        # The coordinates and confidence levels of all key points in this box.
        bbox_keypoints=bboxes_keypoints[idx] # Mx3
        # Draw the skeleton connection of this box
        for skeleton in skeleton_map:
                
            # Each skeleton containing only 1 of each keypoints, i.e., 1 left shoulder
            # Get starting point coordinates
            srt_kpt_id=skeleton['srt_kpt_id']
            if bbox_keypoints[srt_kpt_id][2]==0: continue # check visibility, if not ignore
            srt_kpt_x=bbox_keypoints[srt_kpt_id][0] # scalar value
            srt_kpt_y=bbox_keypoints[srt_kpt_id][1] # scalar value
            # Get the coordinates of the termination point
            dst_kpt_id=skeleton['dst_kpt_id'] 
            if bbox_keypoints[dst_kpt_id][2]==0: continue  # check visibility, if not ignore
            dst_kpt_x=bbox_keypoints[dst_kpt_id][0] # scalar value
            dst_kpt_y=bbox_keypoints[dst_kpt_id][1] # scalar value
            # Get skeleton connection color
            skeleton_color=skeleton['color']
            # Get skeleton connection line width
            skeleton_thickness=skeleton['thickness']
            # Draw skeleton connection
            img_bgr=cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,thickness=skeleton_thickness)
        # Key points for drawing this box
        for kpt_id in kpt_color_map:
            # check visibility, if not ignore
            if bbox_keypoints[kpt_id][2]==0: continue
            # Obtain the color, radius, and XY coordinates of the key point.
            kpt_color=kpt_color_map[kpt_id]['color']
            kpt_radius=kpt_color_map[kpt_id]['radius']
            kpt_x=bbox_keypoints[kpt_id][0]
            kpt_y=bbox_keypoints[kpt_id][1]
            # Draw a circle: image, XY coordinates, radius, color, line width (-1 for fill).
            img_bgr=cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

    return img_bgr