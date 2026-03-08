from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

# Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/evaluation/functional/accuracy.py#L156

def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix
    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels of shape (n_clips, )
        y_real (list[int] | np.ndarray[int]): Ground truth labels of shape (n_clips,)
        normalize (str | None): Normalize confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
            If None, confusion matrix will not be normalized. Options are 'true', 'pred', 'all', None. Default to None
    Returns:
        (np.ndarray): Confusion matrix
    """
    assert normalize in ['true', 'pred', 'all', None], "Normalize must be one of ['true', 'pred', 'all', None]"

    if isinstance(y_pred, list):
        y_pred=np.array(y_pred)
        if y_pred.dtype!=np.int64: y_pred=y_pred.astype(np.int64)
    assert isinstance(y_pred, np.ndarray), f"y_pred must be list or np.ndarray, but got {type(y_pred)}"
    assert y_pred.dtype==np.int64, f"y_pred must be np.int64, but got {y_pred.dtype}"

    if isinstance(y_real, list):
        y_real=bp.array(y_real)
        if y_real.dtype!=np.int64: y_real=y_real.astype(np.int64)
    assert isinstance(y_real, np.ndarray), f"y_real must be list or np.ndarray, but got {type(y_real)}"
    assert y_real.dtype==np.int64, f"y_real dtype must be np.int64, but got {y_real.dtype}"

    label_set=np.unique(np.concatenate((y_pred, y_real))) # before concate, each is of size (n_clips,)
    num_labels=len(label_set)
    max_label=label_set[-1] # label_set is sorted in ascending order (small->large) because np.unique returns sorted unique values
    # A translation table to convert arbitrary, non-contiguous class labels into a clean sequence of indices starting from 0.
    label_map=np.zeros(max_label+1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label]=i

    y_pred_mapped=label_map[y_pred]
    y_real_mapped=label_map[y_real]

    # below, `num_labels*y_real_mapped + y_pred_mapped` convert 2D index to 1D index. For example, consider having 3 classes of (0,1,2), num_labels=3 
    # and a data point where y_real=1 and y_pred=2, this pair yield a new 1D index of 3x1+2=5
    # `np.bincount` counts the number of occurances of each 1D index, and `minlength=num_labels**2` makes sue that there are at least this number of bins
    confusion_mat=np.bincount(num_labels*y_real_mapped + y_pred_mapped, minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize=='true': confusion_mat=(confusion_mat/confusion_mat.sum(axis=1, keepdims=True))
        elif normalize=='pred': confusion_mat=(confusion_mat/confusion_mat.sum(axis=0, keepdims=True))
        elif normalize=='all': confusion_mat=(confusion_mat/confusion_mat.sum())
        confusion_mat=np.nan_to_num(confusion_mat) # replace NaN with zero and infinity with large finite numbers
    return confusion_mat
    
def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy

    Args:
        scores (list[np.ndarray]): Prediction scores for each class, with shape (n_clips,K) where n_clips is the number of clips and K is the number of classes
        labels (list[int]): Ground truth labels, with shape (n_clips,)
    Returns:
        (np.ndarray): Mean class array
    """
    pred=np.argmax(scores, axis=1) # along the K-axis, find the index to max value, returning (n_clips,)
    cf_mat=confusion_matrix(pred, labels).astype(float)

    cls_cnt=cf_mat.sum(axis=1) # sum of each row, where the row represents the total ground truth for a specific class (true positive+false negative)
    cls_hit=np.diag(cf_mat) # true positive where the prediction matches reality
    mean_class_acc=np.mean([hit/cnt if cnt else 0. for cnt, hit in zip(cls_cnt, cls_hit)]) # calculate TP/(TP+FN) which is Recall or Sensitivity

    return mean_class_acc

def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds
    Args:
        y_score (np.ndarray): Prediction score for each samples 
        y_true (np.ndarray): Ground truth binary score of 0 and 1 for each sample
    Returns:
        precision (np.ndarray): The precision of different thresholds
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and recall are tested
    """
    assert all(isinstance(x,np.ndarray) for x in (y_score, y_true))
    assert y_score.shape==y_true.shape

    # make y_true a boolean vector
    y_true=(y_true==1)
    # sort scores and corresponding truth values
    desc_score_indices=np.argsort(y_score, kind='mergesort')[::-1] # sort with [::-1] resulting descending order
    y_score=y_score[desc_score_indices] # sort from large to small
    y_true=y_true[desc_score_indices]
    # there may be tie in values, therefore find the `distinct_value_inds`
    distinct_value_inds=np.where(np.diff(y_score))[0] # array of indices where the predicted scores change value
    # identify the critical indices in a sorted array to use as potential classification thresholds, 
    # with y_true.size-1 represent the index of the last element
    threshold_inds=np.r_[distinct_value_inds, y_true.size-1] # concatenate `y_true.size-1` to distinct_value_inds
    # accumulate the true positives with decreasing threshold. Because y_true was sorted based on prediction scores in descending order. 
    # `y_true` now represents the model's 'most confident' prediction down to 'least confident'
    # np.cumsum: running count of true positive. For instance, at index i, cumsum tells us 'if threshold set to include everything from index 0 to i,
    # how many correct items we found?
    tps=np.cumsum(y_true)[threshold_inds] # calculate the number of true positives at every possible decision threshold
    fps=1+threshold_inds-tps # the number of predicted positives - tps, i.e., the number of predicted positives = 1+threshold_inds (indices count from 0)
    thresholds=y_score[threshold_inds]

    precision=tps/(tps+fps)
    precision[np.isnan(precision)]=0
    recall=tps/tps[-1] # = tp/(tp+fn) where tp+fn is the total number of actual positive cases. tps is the cumulative sum of ground truth y_real
    # stop when full recall attained and reverse the outputs so recall is decreasing
    # find the first index where the model reaches its maximum possible Recall.
    last_ind=tps.searchsorted(tps[-1]) # the first index where the number of tru positives equal the total number of positives
    sl=slice(last_ind, None, -1) # start the slice at the first index where maximum recall was achieved and move all the way to the beginning with -1 step
    # sl making recall starting from 1 to 0
    # Note: high threshold, low recall, (usually) high precision
    #       low threshold, high recall, (usually) low precision
    return np.r_[precision[sl],1], np.r_[recall[sl], 0], thresholds[sl]

def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition. Use this if you want to know how robust the model is at detecting specific categories. For example,
    is the model better at detecting 'Cat' than 'Dog'?
    Args:
        scores (list[np.ndarray]): A list of length N (samples) where each element is an np.ndarray of shape (C,) representing the confidence scores for
            each class
        labels (list[np.ndarray]): A list of lenth N (samples), where each element is a `many-hot` np.ndarray of shape (C,) where multiple classes can be
            labeled as 1
    Returns:
        (np.float64): The mean average precision
    """
    results=[]
    scores=np.stack(scores).T # from (N,C) to (C,N)
    labels=np.stack(labels).T # from (N,C) to (C,N)

    for score, label in zip(scores, labels):
        # How well did the model identify Class X across the entire dataset?
        precision, recall, _ = bynary_precision_recall_curve(score, label)
        # np.diff(recall): calculates the width along x-axis
        # np.array(precision)[:-1] provides the height of the precision curve and [:-1] ensures the dimensions match the result of np.diff
        # np.sum add all the areas
        # - is to counter the negativity of np.diff(recall) since recall is sorted in descending order (i.e., 1 to 0)
        ap = -np.sum(np.diff(recall)*np.array(precision)[:-1]) 
        results.append(ap)
    results=[x for x in results if not np.isnan(x)]
    if results==[]: return np.nan
    return np.mean(results) # The result is the average performance across all $C$ classes

def mmit_mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition. Used for reporting MMIT style map on multi-moment in times. The difference is that this method
    calculates average-precision for each sample and averages them among samples. Use this function, if you care about the model's ability to prioritize 
    the correct classes. For example, if a video has 5 labels, this measure tells whther the model put those 5 labels at the top of its prediction
    Args:
        scores (list[np.ndarray]): A list of length N (samples) where each element is an np.ndarray of shape (C,) representing the confidence scores for
            each class
        labels (list[np.ndarray]): A list of lenth N (samples), where each element is a `many-hot` np.ndarray of shape (C,) where multiple classes can be
            labeled as 1
    Returns:
        (np.float64): The MMIT style mean average precision
    """
    result=[]
    for score, label in zip(scores, labels):
        #How well did the model rank the multiple labels for this specific video/image?
        precision, recall, _=binary_precision_recall_curve(score, label) # score and label, each is of size (C,)
        ap=-np.sum(np.diff(recall)*np.array(precision)[:-1])
        results.append(ap)
    return np.mean(ap) # The result is the average performance across all $N$ samples.
        
        
def top_k_accuracy(scores, labels, topk=(1,)):
    """Calculate top k accuracy score
    Args:
        scores (list[np.ndarray]): Prediction scores for each class with shape (n_clips,K) where n_clips is the number of clips and K is the number of classes
        labels (list[int]): Ground truth labels of shape (n_clips,)
        topk (tuple[int]): K value for `top_k_accuracy`. Default to (1,)
    Returns:
        (list[float]): Top k accuracy score for each k
    """
    res=[]
    labels=np.array(labels)[:,np.newaxis] # (n_clips,1)
    for k in topk:
        # argsort sorts from small to large
        # select the biggest k
        # reorganize indices to point to large to small
        max_k_preds=np.argsort(scores, axis=1)[:,-k:][:,::-1] # (n_clips, k)
        # perform logical_or for `max_k_preds==labels` along the k-axis and reduce to 1 value of either True or False
        match_array=np.logical_or.reduce(max_k_preds==labels, axis=1) # (n_clips,)
        topk_acc_score=match_array.sum()/match_array.shape[0]
        res.append(topk_acc_score)
    return res