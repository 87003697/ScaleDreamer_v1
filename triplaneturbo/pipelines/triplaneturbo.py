from typing import *

import torch
import torch.nn as nn

from .base import Pipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class TriplaneTurboTextTo3DPipeline(Pipeline):
    """
    A pipeline for converting text to 3D models.
    """
    def __init__(
        self,
        geometry: Optional[nn.Module] = None,
        materials: Optional[nn.Module] = None,
        background: Optional[nn.Module] = None,
        renderer: Optional[nn.Module] = None,
    ):
        super().__init__()
        if geometry:
            setattr(""