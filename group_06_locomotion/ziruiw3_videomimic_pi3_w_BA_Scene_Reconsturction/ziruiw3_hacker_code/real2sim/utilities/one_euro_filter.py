# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import math


class OneEuroFilter:
    """
    One Euro Filter for smoothing noisy signals
    
    Args:
        freq (float): Signal sampling frequency (Hz)
        mincutoff (float): Minimum cutoff frequency (Hz)
        beta (float): Speed coefficient
        dcutoff (float): Cutoff frequency for derivative (Hz)
    """
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Compute the smoothing factor for given parameters"""
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
    
    def exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing"""
        return a * x + (1 - a) * x_prev
    
    def __call__(self, x: float, timestamp: float = None) -> float:
        """
        Filter incoming signal value
        
        Args:
            x (float): Current signal value
            timestamp (float, optional): Current timestamp. If None, uses freq to compute timestamp
            
        Returns:
            float: Filtered signal value
        """
        # Initialize if this is the first call
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = 0 if timestamp is None else timestamp
            return x
        
        # Compute timestamp if not provided
        if timestamp is None:
            timestamp = self.t_prev + 1.0/self.freq
            
        # Time difference
        t_e = timestamp - self.t_prev
        
        # Estimate derivative
        dx = (x - self.x_prev) / t_e
        
        # Smooth derivative
        edx = self.exponential_smoothing(
            self.smoothing_factor(t_e, self.dcutoff),
            dx,
            self.dx_prev
        )
        
        # Use derivative to adjust cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(edx)
        
        # Smooth position
        filtered_x = self.exponential_smoothing(
            self.smoothing_factor(t_e, cutoff),
            x,
            self.x_prev
        )
        
        # Save values for next iteration
        self.x_prev = filtered_x
        self.dx_prev = edx
        self.t_prev = timestamp
        
        return filtered_x