from __future__ import annotations
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import warnings
import numpy as np

class HistoryBuffer:
    """Unified storage format for different log types
    It records the history of log for further statistics
    Args:
        log_history (Sequence): History logs. Default to []
        count_history (Sequence): Counts of history logs. Default to []
        max_length (int): The max length of history logs. Default to 1000000
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/logging/history_buffer.py#L8
    """
    _statistics_methods:dict=dict()
    def __init__(self, log_history:Sequence=[], count_history:Sequence=[], max_length:int=1000000):
        
        self.max_length=max_length
        self._set_default_statistics()
        assert len(log_history)==len(count_history), f"The length of log_history {len(log_history)} must be equal to the length of count_history {len(count_history)}"

        if len(log_history)>max_length:
            warnings.warn(f"The length of history buffer ({len(log_history)}) exceeds the max_length ({max_length}), the first few elements will be ignored")
            self._log_history=np.array(log_history[-max_length:])
            self._count_history=np.array(count_history[-max_length:])
        else:
            self._log_history=np.array(log_history)
            self._count_history=np.array(count_history)
            
    def _set_default_statistics(self)->None:
        """Register default statistic methods: min, max, current, and mean"""
        self._statistics_methods.setdefault('min', HistoryBuffer.min)
        self._statistics_methods.setdefault('max', HistoryBuffer.max)
        self._statistics_methods.setdefault('current', HistoryBuffer.current)
        self._statistics_methods.setdefault('mean', HistoryBuffer.mean)

    def update(self, log_val:Union[int, float], count:int=1)->None:
        """Update the log history
        
        If the length of buffer exceeds `self._max_length`, the oldest element will be removed from the buffer
        Args:
            log_val (int | float): The value of log
            count (int): The accumulation times of log, to be used in smooth statistics. Default to 1
        """
        assert isinstance(log_val, (int, float)), f"log_val must be int or float, but got {type(log_val)}"
        assert isinstance(count, int), f"count must be int, but got {type(count)}"
        self._log_history=np.append(self._log_history, log_val)
        self._count_history=np.append(self._count_history, count)
        if len(self._log_history)>self.max_length:
            self._log_history=self._log_history[-self.max_length]
            self._count_history=self._count_history[-self.max_length]
    @property
    def data(self)->tuple[np.ndarray, np.ndarray]:
        """Get the `_log_history` and `_count_history`
        Returns:
            (tuple[np.ndarray, np.ndarray]): History logs and the counts of the history logs
        """
        return self._log_history, self._count_history
        
    @classmethod
    def register_statistics(cls, method:Callable)->Callable:
        """Register custom statistics method to `_statistics_methods`

        The registered method can be called by `history_buffer.statistics` with corresponding method name and arguments
        
        Examples:
            >>> @HistoryBuffer.register_statistics
            >>> def weighted_mean(self,window_size, weight):
            >>>     assert len(weight)==window_size
            >>>     return (self._log_history[-window_size:]*np.array(weight)).sum() / self._count_history[-window_size:].sum()
            >>> log_buffer=HistoryBuffer([1,2],[1,1])
            >>> log_buffer.statistics('weighted_mean', 2, [2,1])
            2
        Args:
            method (Callable): Custom statistics method
        Returns:
            (Callable): Original custom statistics method
        """
        method_name=method.__name__
        assert method_name not in cls._statistics_methods, "method_name cannot be registered twince"
        cls._statistics_methods[method_name]=method
        return method

    def statistics(self, method_name:str, *args, **kwargs)->Any:
        """Access statistics method by name
        Args:
            method_name (str): Name of method
        Returns:
            (Any): Depends on corresponding method
        """
        assert method_name in self._statistics_methods, f"{method_name} has not been registered in History.Buffer._statistics_methods"
        method=self._statistics_methods[method_name]
        # Provide self arguments for registered functions
        return method(self, *args, **kwargs)

    def mean(self, window_size:Optional[int]=None)->np.ndarray:
        """Return the mean of the latest `window_size` values in log histories

        If `window_size is None` or `window_size>len(self._log_history)`, return global mean value 

        Args:
            window_size (int, optional): Size of statistics window
        Returns:
            (np.ndarray): Mean value within the window (scalar value)
        """
        if window_size is not None: 
            assert isinstance(window_size, int), f"The type of window size should be int, but got {type(window_size)}"
        else: window_size=len(self._log_history)

        logs_sum=self._log_history[-window_size:].sum()
        counts_sum=self._count_history[-window_size:].sum()
        return logs_sum/counts_sum

    def max(self, window_size:Optional[int]=None)->np.ndarray:
        """Return the maximum value of the latest `window_size` values in log histories

        If `window_size is None` or `window_size>len(self._log_history)`, return global max value

        Args:
            window_size (int, optional): Size of statistics window
        Returns:
            (np.ndarray): The maximum value within the window (scalar value)
        """
        if window_size is not None: 
            assert isinstance(window_size, int), f"The type of window size should be int, but got {type(window_size)}"
        else: window_size=len(self._log_history)
        return self._log_history[-window_size:].max()

    def min(self,window_size:Optional[int]=None)->np.ndarray:
        """Return the minimum value of the latest `window_size` values in log histories
        If `window_size is None` or `window_size>len(self._log_history)`, return global min value

        Args:
            window_size (int, optional): Size of statistics window
        Returns:
            (np.ndarray): The maximum value within the window (scalar value)        
        """
        if window_size is not None:
            assert isinstance(window_size, int), f"The type of window_size should be int, but got {type(window_size)}"
        else: window_size=len(self._log_history)
        return self._log_history[-window_size:].min()

    def current(self)->np.ndarray:
        """Return the recently updated values in log histories
        Returns:
            (np.ndarray): Recently updated values in log histories
        """
        if len(self._log_history)==0:
            raise ValueError("HistoryBuffer._log_history is an empty array! Please call update first")
        return self._log_history[-1]

    def __getstate__(self)->dict:
        """Make `_statistics_methods` can be resumed
        Returns:
            (dict): Stat dict including statistics_methods
        """
        self.__dict__.update(statistics_methods=self._statistics_methods)
        return self.__dict__

    def __setstate__(self, state):
        """Try to load `_statistics_methods` from state
        Args:
            state (dict): State dict
        """
        statistics_methods=state.pop('statistics_methods', [])
        self._set_default_statistics()
        self._statistics_methods.update(statistics_methods)
        self.__dict__.update(state)
    