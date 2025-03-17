from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor

def create_standard_conv(input_channels: int, output_channels: int, step: int = 1, group_count: int = 1, dilate_factor: int = 1) -> nn.Conv2d:
    """Standard 3x3 convolution with padding"""
    return nn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=3,
        stride=step,
        padding=dilate_factor,
        groups=group_count,
        bias=False,
        dilation=dilate_factor,
    )

def create_pointwise_conv(input_channels: int, output_channels: int, step: int = 1) -> nn.Conv2d:
    """Pointwise 1x1 convolution"""
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=step, bias=False)

class StandardBlock(nn.Module):
    multiplier: int = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        step: int = 1,
        shortcut: Optional[nn.Module] = None,
        group_count: int = 1,
        base_channels: int = 64,
        dilate_factor: int = 1,
        norm_fn: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_fn is None:
            norm_fn = nn.BatchNorm2d
        if group_count != 1 or base_channels != 64:
            raise ValueError("StandardBlock requires group_count=1 and base_channels=64")
        if dilate_factor > 1:
            raise NotImplementedError("Dilation > 1 not implemented in StandardBlock")

        self.conv_1 = create_standard_conv(input_dim, hidden_dim, step)
        self.norm_1 = norm_fn(hidden_dim)
        self.activate = nn.ReLU(inplace=True)
        self.conv_2 = create_standard_conv(hidden_dim, hidden_dim)
        self.norm_2 = norm_fn(hidden_dim)
        self.shortcut = shortcut
        self.step = step

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        result = self.conv_1(x)
        result = self.norm_1(result)
        result = self.activate(result)

        result = self.conv_2(result)
        result = self.norm_2(result)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        result += residual
        result = self.activate(result)

        return result

class BottleneckBlock(nn.Module):
    multiplier: int = 4

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        step: int = 1,
        shortcut: Optional[nn.Module] = None,
        group_count: int = 1,
        base_channels: int = 64,
        dilate_factor: int = 1,
        norm_fn: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_fn is None:
            norm_fn = nn.BatchNorm2d
        channel_width = int(hidden_dim * (base_channels / 64.0)) * group_count

        self.conv_1 = create_pointwise_conv(input_dim, channel_width)
        self.norm_1 = norm_fn(channel_width)
        self.conv_2 = create_standard_conv(channel_width, channel_width, step, group_count, dilate_factor)
        self.norm_2 = norm_fn(channel_width)
        self.conv_3 = create_pointwise_conv(channel_width, hidden_dim * self.multiplier)
        self.norm_3 = norm_fn(hidden_dim * self.multiplier)
        self.activate = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        self.step = step

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        result = self.conv_1(x)
        result = self.norm_1(result)
        result = self.activate(result)

        result = self.conv_2(result)
        result = self.norm_2(result)
        result = self.activate(result)

        result = self.conv_3(result)
        result = self.norm_3(result)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        result += residual
        result = self.activate(result)

        return result
    
class ResNet(nn.Module):
    def __init__(
        self,
        block_type: Type[Union[StandardBlock, BottleneckBlock]],
        block_counts: List[int],
        init_zero_residual: bool = False,
        group_count: int = 1,
        channel_multiplier: int = 64,
        dilation_config: Optional[List[bool]] = None,
        norm_fn: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.norm_layer = nn.BatchNorm2d if norm_fn is None else norm_fn
        
        # Changed initial planes from 64 to 51
        self.current_channels = 51
        self.dilation_rate = 1
        
        # Configure dilation settings
        if dilation_config is None:
            dilation_config = [False, False, False]
        if len(dilation_config) != 3:
            raise ValueError(f"dilation_config must be None or 3-element tuple, got {dilation_config}")
            
        # Network parameters
        self.group_count = group_count
        self.channel_multiplier = channel_multiplier
        
        # Initial convolution block
        self.entry_conv = nn.Conv2d(
            3, self.current_channels, 
            kernel_size=7, stride=2, 
            padding=3, bias=False
        )
        self.entry_norm = self.norm_layer(self.current_channels)
        self.activate = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Main network blocks
        self.stage1 = self._build_stage(block_type, 51, block_counts[0])
        self.stage2 = self._build_stage(
            block_type, 102, block_counts[1], 
            step=2, use_dilation=dilation_config[0]
        )
        self.stage3 = self._build_stage(
            block_type, 204, block_counts[2], 
            step=2, use_dilation=dilation_config[1]
        )
        self.stage4 = self._build_stage(
            block_type, 408, block_counts[3], 
            step=2, use_dilation=dilation_config[2]
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        self._initialize_weights(init_zero_residual)

    def _initialize_weights(self, init_zero_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode="fan_out", 
                    nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if init_zero_residual:
            for module in self.modules():
                if isinstance(module, BottleneckBlock):
                    nn.init.constant_(module.norm_3.weight, 0)
                elif isinstance(module, StandardBlock):
                    nn.init.constant_(module.norm_2.weight, 0)

    def _build_stage(
        self,
        block_type: Type[Union[StandardBlock, BottleneckBlock]],
        channels: int,
        num_blocks: int,
        step: int = 1,
        use_dilation: bool = False,
    ) -> nn.Sequential:
        norm_fn = self.norm_layer
        shortcut = None
        prev_dilation = self.dilation_rate

        if use_dilation:
            self.dilation_rate *= step
            step = 1

        # Create shortcut connection if dimensions change
        if step != 1 or self.current_channels != channels * 4:
            shortcut = nn.Sequential(
                create_pointwise_conv(self.current_channels, channels * 4, step),
                norm_fn(channels * 4),
            )

        # Build layers for this stage
        stage_layers = []
        
        # First block may have different stride/shortcut
        stage_layers.append(
            block_type(
                self.current_channels, 
                channels, 
                step,
                shortcut,
                self.group_count,
                self.channel_multiplier,
                prev_dilation,
                norm_fn
            )
        )
        
        # Update channel count for subsequent blocks
        self.current_channels = channels * 4
        
        # Add remaining blocks
        for _ in range(1, num_blocks):
            stage_layers.append(
                block_type(
                    self.current_channels,
                    channels,
                    group_count=self.group_count,
                    base_channels=self.channel_multiplier,
                    dilate_factor=self.dilation_rate,
                    norm_fn=norm_fn,
                )
            )

        return nn.Sequential(*stage_layers)

    def extract_features(self, x: Tensor) -> Tensor:
        # Initial processing
        x = self.entry_conv(x)
        x = self.entry_norm(x)
        x = self.activate(x)
        x = self.pool(x)

        # Process through stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling and flatten
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.extract_features(x)