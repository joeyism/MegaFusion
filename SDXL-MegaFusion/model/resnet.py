import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.activations import get_activation

class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        norm_type: Optional[str] = None,
        eps: Optional[float] = None,
        elementwise_affine: Optional[bool] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        else:
            self.norm = None if norm_type is None else ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)

        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding: int = 1,
        norm_type: Optional[str] = None,
        eps: Optional[float] = None,
        elementwise_affine: Optional[bool] = None,
        bias: bool = True,
        interpolate: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        else:
            self.norm = None if norm_type is None else ValueError(f"unknown norm_type: {norm_type}")

        if use_conv_transpose:
            self.conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size or 4, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size or 3, padding=padding, bias=bias)
        else:
            self.conv = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if self.interpolate:
            hidden_states = F.interpolate(hidden_states, size=output_size, scale_factor=2.0, mode="nearest")

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dilation: bool = False,
    ):
        super().__init__()

        if time_embedding_norm in ["ada_group", "spatial"]:
            raise ValueError(f"This class cannot be used with `time_embedding_norm=={time_embedding_norm}`, please use `ResnetBlockCondNorm2D` instead")

        self.pre_norm = True
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        self.if_dilation = dilation

        linear_cls = nn.Linear
        conv_cls = nn.Conv2d

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        if self.if_dilation == True:
            self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        else:
            self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        
        if self.if_dilation == True:
            self.conv2 = conv_cls(out_channels, conv_2d_out_channels, kernel_size=3, stride=1,  padding=2, dilation=2)
        else:
            self.conv2 = conv_cls(out_channels, conv_2d_out_channels, kernel_size=3, stride=1,  padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = Upsample2D(in_channels, use_conv=False) if up else None
        self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op") if down else None

        self.use_in_shortcut = self.in_channels != (conv_2d_out_channels or self.out_channels) if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = nn.Conv2d(in_channels, conv_2d_out_channels or self.out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias) if self.use_in_shortcut else None


    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
                
        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}")
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor