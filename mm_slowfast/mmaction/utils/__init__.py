from typing import List, Dict,Tuple, Union
import torch

from computer_vision.mm_slowfast.mmaction.structures.action_data_sample import ActionDataSample

SampleList=List[ActionDataSample]

ForwardResults = Union[Dict[str, torch.Tensor], List[ActionDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

