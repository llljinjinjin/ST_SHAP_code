import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # pad是从后往前，从左往右，从上往下，原顺序是（B,C,H,W) pad顺序就是(W，H，C）
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B,  C,H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W  # 这里是经过padding的H和W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 将通道数由4倍变为2倍

    def forward(self, x, H, W):
        """
        x: B, H*W（L）, C，并不知道H和W，所以需要单独传参
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 因为是下采样两倍，如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 此时（B,H,W,C)依然是从后向前
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C],这里的-1就是在C的维度上拼接
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#  这里head_dim就是dk，并且在MSA中每个头都要有自己的相对位置表，然后生成相对位置索引表
#  基于窗口的具有相对位置偏差的多头自我注意(W-MSA)模块。它同时支持移位和非移位窗口
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        如果为True，则为query、key、value添加一个可学习的偏差。默认值:真正的
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0注意力权重的退出比例。默认值:0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0输出的退出率。默认值:0.0
        # （主要区别是在原始计算Attention的公式中的Q,K时加入了相对位置编码，Swin Transformer则将注意力的计算限制在每个窗口内）。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  #  dim (int):输入通道数。
        self.window_size = window_size  # [Mh, Mw]  window_size (tuple[int]):窗口的高度Wh  和宽度Ww
        self.num_heads = num_heads  # num_heads (int):注意头数。
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # **幂

        # 定义一个相对位置偏差的参数表
        # 每一个head都有自己的relative_position_bias_table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # 获取窗口内每个标记的成对相对位置索引
        # 首先我们利用torch.arange和torch.meshgrid函数生成对应的坐标
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # 生成对应坐标，然后堆叠起来，展开为一个二维向量
        # meshgrid其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同。
        # meshgrid生成网格，再通过stack方法拼接
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # 利用广播机制，分别在第一维，第二维，插入一个维度，进行广播相减
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(
            1)  # [2, Mh*Mw, Mh*Mw]
        # 因为采取的是相减，所以得到的索引是从负数开始的，我们加上偏移量，让其从0开始。
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]调整轴 Wh*Ww, Wh*Ww, 2 即每个里面都是二维坐标
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 后续我们需要将其展开成一维偏移量。而对于(1，2）和（2，1）这两个坐标。在二维上是不同的，但是通过将x,y坐标相加转换为一维偏移的时候，他的偏移量是相等的。
        # 所以最后我们对其中做了个乘法操作，以进行区分  这个时候索引范围为y:[0, 2*(window_size-1)*(2 * window_size - 1)], x:[0, 2*(window_size-1)]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 然后再最后一维上进行求和，展开成一个一维坐标
        # 这个时候索引范围为[0, 2 * (window_size - 1) * (2 * window_size - 1) + 2 * (window_size - 1)]化简一下[0, (2 * window_size - 1) * (2 * window_size - 1) - 1]
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # 整个训练当中，window_size大小不变，因此这个索引也不会改变
        self.register_buffer("relative_position_index", relative_position_index)  # 并注册为一个不参与网络学习的变量

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多头融合
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # 并初始化为一个截断正态分布
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C) 输入张量形状
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape  #  128，16，96
        # 经过self.qkv这个全连接层后，进行reshape，调整轴的顺序，得到形状为3
        # permute函数的作用是对tensor进行转置 3，128，3，16，96/3=32即每个头对应的通道数
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # 通过unbind分别获得qkv
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)分配给qkv 128，3，16，32

        # 根据公式，我们对q乘以一个scale缩放系数，然后与k（为了满足矩阵乘要求，需要将最后两个维度调换）进行相乘。
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale  # 形状为(numWindows*B, num_heads, window_size*window_size, window_size*window_size)的attn张量
        attn = (q @ k.transpose(-2, -1))

        # 之前我们针对位置编码设置了个形状为(2*window_size-1*2*window_size-1, numHeads)的可学习变量。
        # 再使用对应的相对位置偏置表（Relative position bias table）进行映射即可得到最终的相对位置偏置B
        # 我们用计算得到的相对编码位置索引self.relative_position_index选取，得到形状为(window_size*window_size, window_size*window_size, numHeads)的编码，加到attn张量上
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        # 通过unsqueeze加上一个batch维度 在这里进行相对位置编码
        attn = attn + relative_position_bias.unsqueeze(0)

        # 暂不考虑mask的情况，剩下就是跟transformer一样的softmax，dropout，与V矩阵乘，再经过一层全连接层和dropout
        # 当需要使用SiftedWindowAttention时，mask不为None，这需要对Attention的结果attn加上mask，并通过softmax，-100位置的值经过softmax后将会被忽略。
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#  进入SwinTransformerBlock前，x上一个经历的是create_mask，维度是[B, HW, C]，所以需要单独传入H和W
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
         # dim (int):输入通道数。
        # input_resolution (tuple[int]):输入恢复。
        # num_heads (int):注意头数。
        # window_size (int):窗口大小。
        # shift_size (int): SW-MSA的Shift大小。
        # mlp_ratio (float): mlp隐藏dim与嵌入dim的比值。
        # qkv_bias (bool，可选):如果为True，则为query、key、value添加一个可学习的偏差。默认值:真正的
        # drop (float，可选):退出率。默认值:0.0
        # attn_drop(浮动，可选):注意力下降率。默认值:0.0
        # drop_path (float，可选):随机深度速率。默认值:0.0
        # act_layer (nn。模块(可选):激活层。默认值:神经网络。GELU
        # norm_layer (nn。模块，可选):归一化层。默认值:神经网络。LayerNorm
    """

    # 与Vit的block结构是相同的
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # LN
        self.norm1 = norm_layer(dim)
        # W-MSA
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # DropPath是一种针对分支网络而提出的网络正则化方法，不通过残差结构也可以搭建深层网络。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # LN
        self.norm2 = norm_layer(dim)
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # shortcut是为了实现残差，之后将x的维度变为(B, H, W, C)。并且还要保证特征图是window_size的整数倍，所以要进行一次padding，此时维度是(B, Hp, Wp, C)。
    # 再进行mask中的移位，如果shift_size是0就说明是W-MSA，不需要移位。然后划分窗口并维度变换
    def forward(self, x, attn_mask):
        # x(B,L,C)，因此需要记录h和w
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # 残差网络
        shortcut = x   # (32,64,96)
        # 先对特征图进行LayerNorm
        x = self.norm1(x)
        x = x.view(B, H, W, C) # (32,8,8,96)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 通过self.shift_size决定是否需要对特征图进行shift
        if self.shift_size > 0:
            # 对窗口进行移位。从上向下移，从左往右移，因此是负的
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows将特征图切成一个个窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C](128,4,4,96)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C](128,16,96)

        # attn_mask在这里传给WindowAttention，之后先取消窗口划分，再将刚才的shift操作还原，再移除刚刚的padding
        # W-MSA/SW-MSA 计算Attention，通过attn_mask来区分Window Attention还是Shift Window Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # shift还原，如果没有shifted就不用还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)  #  做dropout和残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # # 再通过一层LayerNorm+全连接层，以及dropout和残差连接

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): 是否需要下采样，在最后一个stage不需要. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 移动尺寸

        # 在当前stage之中所有的block
        # depth为这个stage中的block数，downsample为patch merging 的意思
        # 注意每个block中只会有一个MSA,要么W-MSA，要么SW-MSA，所以shift_size为0代表W-MSA，不为0代表SW-MSA
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA计算SW-MSA的注意掩码
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # slice窗口切分，划分每个窗口所占的区域。再利用for循环，给每个patch打上标签，数字为同样的patch对应在同一个window里
        # 设置切片，从行上，将mask分为3部分
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 设置切片，从列上，将mask分为3部分
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 所以，mask一共有9个区域

        # 对每个区域的mask进行编号
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 划分 window
        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1] 划为窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw] 窗口展平
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw] 通过上面的减法得到每个窗口的对应区域，同一个区域为0，不同的为其他数值。attn_mask在不为0的地方填-100
        # attn_mask:用在attention上的mask。只有同一窗口的元素之间才计算attention，对于不同的窗口的直接通过标记为-100，这样就不会对原始数据产生影响了。
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # 先创建一个mask蒙版，在图像尺寸不变的情况下蒙版也不改变
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # 默认不适用checkpoint方法
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # attn_mask是直接传进SwinTransformerBlock的forward函数
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            # 防止H和W是奇数。如果是奇数，在下采样中经过一次padding就变成偶数了，但如果这里不给H和W加一的话就会导致少一个，如果是偶数，加一除二取整还是不变
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W

# 整个模型采取层次化的设计，一共包含4个Stage，每个stage都会缩小输入特征图的分辨率，像CNN一样逐层扩大感受野。
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Swin Transformer可以作为一个通用骨架，在这里将其用在分类任务中，最后分为num_classes个类. Default: 1000
        embed_dim (int): Patch embedding dimension，就是原文中的C. Default: 96
        depths (tuple(int)): 每个stage中的Swin Transformer Block数.
        num_heads (tuple(int)): 每个stage中用的multi-head数.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): mlp的隐藏层是输入层的多少倍. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True如果为True，则为查询、键、值添加一个可学习的偏差。默认值:真正的
        drop_rate (float): 在pos_drop,mlp及其他地方. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): 每一个Swin Transformer之中，注意它的dropout率是递增的. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): 如果使用可以节省内存. Default: False
    """
    # depths中的每一个值代表每一个stage中有多少个Swin Transformer Block
    # num_heads指的是在每一个stage中的所有Block中的MSA使用的头数
    def __init__(self, img_size=32,patch_size=4, in_chans=4, num_classes=3,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=2, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.17, attn_drop_rate=0.17, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        # 对应Patch partition和Linear Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 在每个block的dropout率，是一个递增序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # num_layers及stage数
            # dim为当前stage的维度，depth是当前stage堆叠多少个block，drop_patch是本层所有block的drop_patch
            # downsample是Patch merging，并且在最后一个stage为None
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 拿走这一层所有block的dpr
                                norm_layer=norm_layer,
                                #  与论文不同，代码中的stage包含的是下一层的Patch merging ，因此在最后一个stage中没有Patch merging
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 在这个分类任务中，用全局平均池化取代cls token
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # 权重初始化函数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # 依次通过每个stage
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C] 32 1 768
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1] 32 768 1
        x = torch.flatten(x, 1) # 32 768
        x = self.head(x) # 32 3
        return x

