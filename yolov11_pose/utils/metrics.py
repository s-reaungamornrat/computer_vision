from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import torch
import numpy as np

def bbox_ioa(box1:np.ndarray, box2:np.ndarray, iou:bool=False, eps:float=1e-7)->np.ndarray:
    """Calculate the intersection over box2 area given box1 and box2
    Args:
        box1 (np.ndarray): A numpy array of shape (N,4) representing N bounding boxes in x1y1x2y2 format
        box2 (np.ndarray): A numpy array of shape (M,4) representing M bounding boxes in x1y1x2y2 format
        iou (bool, optional): Calculate the standard IoU if True else inter_area/box2_area
        eps (float): A small value to avoid division by zero.
    Returns:
        (np.ndarray): A numpy array of shape (N,M) representing the intersection over box2 area
    """
    # Get coordinates of the bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2=box1.T
    b2_x1, b2_y1, b2_x2, b2_y2=box2.T

    # Intersection area
    inter_area=(np.minimum(b1_x2[:,None], b2_x2)-np.maximum(b1_x1[:,None], b2_x1)).clip(0)*(
        np.minimum(b1_y2[:,None], b2_y2)-np.maximum(b1_y1[:,None], b2_y1)).clip(0)
    # Box2 area
    area=(b2_x2-b2_x1)*(b2_y2-b2_y1)
    if iou:
        box1_area=(b1_x2-b1_x1)*(b1_y2-b1_y1)
        area=area+box1_area[:,None]-inter_area
    # Intersection over box2 are
    return inter_area/(area+eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    
def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def compute_ap(recall:list[float], precision:list[float])->tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves
    Args:
        recall (list): The recall curve
        precision (list): The precision curve
    Returns:
        ap (float): Average precision
        mpre (np.ndarray): Precision envelope curve
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end
    """
    # Append sentinel valies to beginning and end
    mrec=np.concatenate(([0.], recall, [1.]))
    mpre=np.concatenate(([1.],precision,[0.]))

    # Compute the precision envelope
    mpre=np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method='interp' # methods: 'continuous', 'interp'
    if method=='interp':
        x=np.linspace(0,1,101) # 101-point interp (COCO)
        func=np.trapezoid if int(np.__version__[:np.__version__.find('.')])>=2 else np.trapz # np.trapz deprecated
        ap=func(np.interp(x, mrec, mpre), x) # integrate
    else: # continuous
        i=np.where(mrec[1:]!=mrec[:-1])[0] # points where x-axis (recall) changes
        ap=np.sum((mrec[i+1]-mrec[i])*mpre[i+1]) # area under curve
    return ap, mpre, mrec

def ap_per_class(tp:np.ndarray, conf:np.ndarray, pred_cls:np.ndarray, target_cls:np.ndarray,plot:bool=False,
                save_dir:Path=Path(), names:dict[int, str]={}, eps:float=1e-16, prefix:str='')->tuple:
    """Compute the average precision per class for object detection evaluation
    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False)
        conf (np.ndarray): Array of confidence scores of the detections
        pred_cls (np.ndarray): Array of predicted classes of the detections
        target_cls (np.ndarray): Array of true classes of the detections
        plot (bool, optional): Whether to plot PR curves or not
        save_dir (Path, optional): Directory to sabe the PR curves
        names (dict[int, str], optional): Dict of class names to plot PR curves
        eps (float, optional): A small value to avoid division by zero
        prefix (str, optional): A prefix string for saving the plot files
    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class
        ap (np.ndarray): Average precision for each class at different IoU thresholds
        unique_classes (np.ndarray): An array of unique class that  have data
        p_curve (np.ndarray): Precision curves for each class
        r_curve (np.ndarray): Recall curves for each class
        f1_curve (np.ndarray): F1-score curves for each class
        x (np.ndarray): X-axis values for the curves
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class
    """
    # Sort by objectness
    i=np.argsort(-conf)
    tp, conf, pred_cls=tp[i],conf[i],pred_cls[i]

    # Find unique classes and number of detections per class
    unique_classes, nt=np.unique(target_cls, return_counts=True)
    nc=unique_classes.shape[0] # number of classes

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values=np.linspace(0,1,1000), []

    # Average precision, precision, and recall curves
    ap, p_curve, r_curve=np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i=pred_cls==c
        n_l=nt[ci] # number of labels
        n_p=i.sum() # number of predictions
        if n_p==0 or n_l==0: continue
            
        # Accumulate FPs and TPs
        fpc=(1-tp[i]).cumsum(0)
        tpc=tp[i].cumsum(0)

        # Recall
        recall=tpc/(n_l+eps) # recall curve
        r_curve[ci]=np.interp(-x, -conf[i], recall[:,0], left=0) # negative x, xp because xp decreases

        # Precision
        precision=tpc/(tpc+fpc) # precision curve
        p_curve[ci]=np.interp(-x, -conf[i], precision[:,0], left=1) # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci,j],mpre,mrec=compute_ap(recall[:,j], precision[:,j])
            if j==0: prec_values.append(np.interp(x, mrec, mpre)) # precision at mAP@0.5

    prec_values=np.array(prec_values) if prec_values else np.zeros((1,1000)) # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve=2*p_curve*r_curve/(p_curve+r_curve+eps)
    names={i:names[k] for i, k in enumerate(unique_classes) if k in names} # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir/f"{prefix}PR_curve.png", names)
        plot_mc_curve(x, f1_curve, save_dir/f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(x, p_curve, save_dir/f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(x, r_curve, save_dir/f'{prefix}R_curve.png', names, ylabel='Recall')
    i=smooth(f1_curve.mean(0), 0.1).argmax() # max F1 index
    p, r, f1=p_curve[:, i], r_curve[:,i], f1_curve[:,i] # max-F1 precision, recall, F1 values
    tp=(r*nt).round() # true positives
    fp=(tp/(p+eps) -tp).round() # false positive
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values

class Metric:
    """Class for computing evaluation metrics
    """
    def __init__(self)->None:
        """Initialize a Metric instance for computing evaluation metrics """
        self.p= [] # (nc,) precision for each class
        self.r=[] # (nc,) recall for each class
        self.f1=[] # (nc,) f1 score for each class
        self.all_ap=[] # (nc, 10) AP scores for all classes and all IoU thresholds for AP 0.5 to 0.95 with step
        self.ap_class_index=[] # (nc,) index of class for each AP scores
        self.nc=0 # number of classes

    @property
    def ap50(self)->np.ndarray|list:
        """Return the average precision (AP) at an IoU threshold of .5 for all classes
        Returns:
            (np.ndarray|list): Array of shape (nc,) with AP50 values per class or an empty list if not available
        """
        return self.all_ap[:,0] if len(self.all_ap) else []
        
    @property
    def ap(self)->np.ndarray | list:
        """Return the average precision (AP) at an IoU of .5-.95 for all classes
        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []
    @property
    def mp(self)->float:
        """ Return the mean precision of all classes
        Returns:
            (float): The mean precision of all classes
        """
        return self.p.mean() if len(self.p) else 0.
    @property
    def mr(self)->float:
        """Return the mean recall of all class
        Returns:
            (float): The mean recall of all classes
        """
        return self.r.mean() if len(self.r) else 0.
    @property
    def map50(self)->float:
        """Return the mean average precision (mAP) at an IoU threshold of .5
        Returns:
            (float): The mAP at an IoU threshold of .5
        """
        return self.all_ap[:,0].mean() if len(self.all_ap) else 0.
    @property
    def map75(self)->float:
        """Return the mean average precision (mAP) at an IoU threshold of .75
        Returns:
            (float): The mAP at an IoU threshold of .5
        """
        return self.all_ap[:,5].mean() if len(self.all_ap) else 0.

    @property
    def map(self)->float:
        """Return the mean average precision (mAP) over IoU thresholds of .5-.95 in steps of .05 
        Returns:
            (float): The mAP over IoU thresholds of .5-.95 in steps of .05
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.

    def mean_results(self)->list[float]:
        """Return mean of results, mp, mr, map50, map
        """
        return [self.mp, self.mr, self.map50, self.map]
        
    def class_result(self, i:int)->tuple[float, float, float, float]:
        """Return class-aware result, p[i], r[i], ap50[i], ap[i]"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self)->np.ndarray:
        """Return mAP of each class"""
        maps=np.zeros(self.nc)+self.map
        for i, c in enumerate(self.ap_class_index): maps[c]=self.ap[i]
        return maps

    def fitness(self)->float:
        """Return model fitness as a weighted combination of metrics"""
        w=[0.,0.,0.,1.] # weights for [P,R, mAP@0.5, mAP@0.5:0.95]
        return (np.nan_to_num(np.array(self.mean_results()))*w).sum()

    def update(self, results:tuple):
        """Update the evaluation metrics with a new set of results
        Args:
            results (tuple): A tuple cobtaining evaluation metrics:
                - p (list): Precision for each class
                - r (list): Recall for each class
                - f1 (list): F1 score for each class
                - all_ap (list): AP scores for all classes and all IoU thresholds
                - p_curve (list): Precision curve for each class
                - r_curve (list): Recall curve for each class
                - f1_curve (list): F1 curve for each class
                - px (list): X values for the curves
                - prec_values (list): Precision values for each class
        """
        (self.p,
        self.r,
        self.f1, 
        self.all_ap,
        self.ap_class_index,
        self.p_curve,
        self.r_curve,
        self.f1_curve,
        self.px,
        self.prec_values)=results

    @property
    def curves(self)->list:
        """Return a list of curves for accessing specific metrics curves"""
        return []
        
    @property
    def curves_results(self)->list[list]:
        """Return a list of curves for accessing specific metrics curves"""
        return [
            [self.px, self.prec_values, 'Recall', 'Precision'],
            [self.px, self.f1_curve, 'Confidence', 'F1'],
            [self.px, self.p_curve, 'Confidence', 'Precision'],
            [self.px, self.r_curve, 'Confidence', 'Recall']
        ]

class DetMetrics:
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP)"""
    
    def __init__(self, names:dict[int, str]={})->None:
        """Initialize a DetMetrics instance with a save dict, plot flag, and class names
        Args:
            names (dict[int, str], optional): dict of class names
        """
        self.names=names # dict of class names
        self.box=Metric() # an instance of the Metric class for storing detection results
        # A dict for storing execution times of different parts of the detection process
        self.speed={'preprocess':0., 'inference':0., 'loss':0., 'postprocess':0.}
        self.task='detect' # the task type
        # A dict containing lists for true positives, confidence scores, predicted classes, target classes, and target images
        self.stats=dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        # number of targets per class
        self.nt_per_class=None
        # number of targets per image
        self.nt_per_image=None
        
    def update_stats(self, stat:dict[str, Any])->None:
        """Update statistics by appending new values to existing stat collections.
        Args:
            stat (dict[str, Any]): Dict containing new statistical values to append. Keys should match existing keys in 
                self.stats
        """
        for k in self.stats.keys(): self.stats[k].append(stat[k])

    def process(self, save_dif:Path=Path('.'), plot:bool=False)->dict[str, np.ndarray]:
        """Process predicted results for object detection and update metrics
        Args:
            save_dir (Path): Directory to save plots. Default to Path(".")
            plot (bool): Whether to plot precision-ercall curves. Default to False
        Returns:
            (dict[str, np.ndarray]): Dict containing concatenated statistics array
        """
        stats={k:np.concatenate(v,0) for k, v in self.stats.items()} # to numpy
        if not stats: return stats
        results=ap_per_class(stats['tp'], stats['conf'], stats['pred_cls'], stats['target_cls'],
                             plot=plot, save_dir=save_dir, names=self.names, prefix="box")[2:]
        self.box.nc=len(self.names)
        self.box.update(results)
        self.nt_per_class=np.bincount(stats['target_cls'].astype(int), minlength=len(self.names))
        self.nt_per_image=np.bincount(stats['target_img'].astype(int), minlength=len(self.names))

        return stats

    def clear_stats(self):
        """Clear the stored statistics"""
        for v in self.stats.values(): v.clear()
    @property
    def keys(self)->list[str]:
        """Return a list of keys for accessing specific metrics"""
        return ["metrics/precision(B)", "metrics/recall(B)", 'metrics/mAP50(B)', "metrics/mAP50-95(B)"]

    def mean_results(self)->list[float]:
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95"""
        return self.box.mean_results()
        
    def class_result(self, i:int)->tuple[float, float, float, float]:
        """Return the result of evaluating the performance of an object detection model on a specific class"""
        return self.box.class_result(i)

    @property
    def maps(self)->np.ndarray:
        """Return mean Average Precision (mAP) scores per class"""
        return self.box.maps

    @property
    def fitness(self)->float:
        """Return the fitness of box object"""
        return self.box.fitness()

    @property
    def ap_class_index(self)->list:
        """Return the average precision index per class"""
        return self.box.ap_class_index
        
    @property
    def results_dict(self)->dict[str, float]:
        """Return dict of computed performance metrics and statistics"""
        keys=[*self.keys, 'fitness']
        values=((float(x) if hasattr(x, 'item') else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))
    @property
    def curves(self)->list[str]:
        """Return a list of curves for accessing specific metric curves"""
        return ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
    @property
    def curves_results(self)->list[list]:
        """Return a list of computed performance metrics and statistics"""
        return self.box.curves_results
    def summary(self, normalize:bool=True, decimals:int=5)->list[dict[str, Any]]:
        """Generate a summarized representatioon of per-class detection metrics as a list of dictionaries. Include
        shared scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class
        
        Args:
            normalize (bool): For Detect metrics, everything is normalized by default [0-1]
            decimals (int): Number of decimal places to round the metrics values to
        Returns:
            (list[dict[str,Any]]): A list of dict, each representing one class with corresponding metric values
        Examples:
            >>> results=model.val(data='coco8.yaml')
            >>> detection_summary=results.summary()
            >>> print(detection_summary)
        """
        per_class={'Box-P':self.box.p, 'Box-R':self.box.r, 'Box-F1':self.box.f1}
        return [
            {'Class':self.names[self.ap_class_index[i]],
             'Images':self.nt_per_image[self.ap_class_index[i]],
             'Instances':self.nt_per_class[self.ap_class_index[i]],
             **{k:round(v[i], decimals) for k, v in per_class.items()},
             'mAP50': round(self.class_result(i)[2], decimals),
             'mAP50-95':round(self.class_result(i)[3], decimals)
            }
            for i in range(len(per_class['Box-P']))
        ]

class PoseMetrics(DetMetrics):
    """Calculate and aggregate detection and pose metrics over a given set of classes
    """
    def __init__(self, names:dict[int, str]={})->None:
        """Initialize the PoseMetric class with class names
        Args:
            names (dict[int, str], optional): Dict of class names
        """
        super().__init__(names)
        self.pose=Metric()
        self.task='pose'
        self.stats['tp_p']=[] # add additinal stats for pose

    def process(self, save_dir:Path=Path('.'), plot:bool=False)->dict[str, np.ndarray]:
        """Process the detection and pose metrics over the given set of predictions

        Args:
            save_dir (Path): Directory to save plots. Default to Path('.')
            plot (bool): Whether to plot precision-recall curve. Default to False
        Returns:
            (dict[str, np.ndarray]): Dict containing concatenated statistics arrays
        """
        stats=DetMetrics.process(self, save_dir, plot) # process box stats
        results_pose=ap_per_class(stats['tp_p'], stats['conf'], stats['pred_cls'], stats['target_cls'], plot=plot, 
                                 save_dir=save_dir, names=self.names,prefix='Pose')[2:]
        self.pose.nc=len(self.names)
        self.pose.update(results_pose)
        return stats
    @property
    def keys(self)->list[str]:
        """Return a list of evaluation metric keys"""
        return [*DetMetrics.keys.fget(self),
               "metrics/precision(P)","metrics/recal(P)","metrics/mAP50(P)","metrics/mAP50-95(P)"]
    def mean_results(self)->list[float]:
        """Return the mean results of box and pose"""
        return DetMetrics.mean_results(self)+self.pose.mean_results()
        
    def class_result(self, i:int)->list[float]:
        """Return the class-wise detection results for a specific class i"""
        return DetMetrics.class_result(self, i)+self.pose.class_result(i)
    @property
    def maps(self)->np.ndarray:
        """Return the mean average precision (mAP) per class for both box and pose detections"""
        return DetMetrics.maps.fget(self)+self.pose.maps
    @property
    def fitness(self)->float:
        """Return combined fitness score for pose and box detection"""
        return DetMetrics.fitness.fget(self)+self.pose.fitness()
    @property 
    def curves(self)->list[str]:
        """Return a list of curves for accessing specific metrics curves"""
        return [*DetMetrics.curves.fget(self), "Precision-Recall(B)", "F1-Confidence(B)", 'Precision-Confidence(B)',
               "Recall-Confidence(B)", "Precision-Recall(P)", "F1-Confidence(P)", "Precision-Confidence(P)",
               "Recall-Confidence(P)"]
    @property
    def curves_results(self)->list[list]:
        """Return a list of computed performance metrics and statistics"""
        return DetMetrics.curves_results.fget(self)+self.pose.curves_results

    def summary(self, normalize:bool=True, decimals:int=5)->list[dict[str, Any]]:
        """Generate a summarized representation of per-class pose metrics as a list of dicts. Includes both box
        and pose scalar metrics (mAp, mAP50, mAP50-95) alogside precision, recall, and F1-score for each class

        Args:
            normalize (bool): For Pose metrics, everything is normalized by default [0-1]
            decimals (int): Number of decimal places to round the metrics values to
        Returns:
            (list[dict[str,Any]]): A list of dict, each representing one class with corresponding metric values.
        Examples:
            >>> results=model.val(data='coco8-pose.yaml')
            >>> pose_summary=results.summary(decimals=4)
            >>> print(pose_summary)
        """
        per_class={"Pose-P":self.pose.p, "Pose-R":self.pose.r, "Pose-F1":self.pose.f1}
        summary=DetMetrics.summary(self, normalize, decimals) # get box summary
        for i, s in enumerate(summary):
            s.update({**{k:round(v[i], decimals) for k, v in per_class.items()}})
        return summary
        
def smooth(y:np.ndarray, f:float=0.05)->np.ndarray:
    """Box filter of fraction f"""
    nf=round(len(y)*f*2) // 2 +1 # number of filter elements (must be odd)
    p=np.ones(nf//2) # ones padding
    yp=np.concatenate((p*y[0], y, p*y[-1]), 0) # y padded
    return np.convolve(yp, np.ones(nf)/nf, mode='valid') # y-smoothed
    
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)