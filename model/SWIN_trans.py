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
    The feature map is divided into Windows with no overlap according to window_size
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
    Restore each window to a feature map
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
        # padding is required if H and W of the input image are not multiples of patch_size
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # The pad is back to front, left to right, top to bottom, and the original order is (B,C,H,W). The pad order is (W,H, C).
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # Subsampling patch_size times
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B,  C,H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W  # Here's the padding H and W


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
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # Change the number of channels from 4 to 2

    def forward(self, x, H, W):
        """
        x: B, H*W（L）, C，H and W are not known, so separate parameters need to be passed
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # padding is required if H and W of the input feature map are not multiples of 2 because the downsampling is twice
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Now (B,H,W,C) is still going from back to front
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # Note that the Tensor channels here are [B, H, W, C], so they will be a little different from the official document
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C],The -1 here is the concatenation in the C dimension
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

# Here head_dim is dk, and in MSA each head must have its own relative position table, and then generate the relative position index table
# Window based multi-head Self-attention (W-MSA) module with relative position bias. It supports both shifted and unshifted Windows
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        # (The main difference is that relative position coding is added to Q and K in the original formula for calculating Attention, while Swin Transformer limits the calculation of attention to each window).
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  #  dim (int):Number of input channels.
        self.window_size = window_size  # [Mh, Mw]  window_size (tuple[int]):The height of the window is Wh and the width Ww
        self.num_heads = num_heads  # num_heads (int):Number of attention heads.
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # **power

        # Define a parameter table with relative position deviation
        # Each head has its own relative_position_bias_table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # Gets the pairwise relative position index for each tag in the window
        # First we use the torch.arange and torch.meshgrid functions to generate the corresponding coordinates
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # Generate the corresponding coordinates, then stack them and expand them into a two-dimensional vector
        # meshgrid where the first output tensor fills the elements in the first input tensor, all the same; The second output tensor fills the elements in the second input tensor in the same columns.
        # meshgrid generates a grid and then splices it using the stack method
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # The broadcast mechanism is used to insert a dimension in the first dimension and the second dimension respectively to carry out broadcast subtraction
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(
            1)  # [2, Mh*Mw, Mh*Mw]
        # Since we're subtracting, the resulting index starts at a negative number, so we add the offset to make it start at 0.
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2] Adjust the axes Wh*Ww, Wh*Ww, 2 that is, each inside is a two-dimensional coordinate
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # Then we need to expand it into a one-dimensional offset. And for the coordinates (1,2) and (2,1). It's different in two dimensions, but by adding the x and y coordinates to convert the one-dimensional offset, the offset is the same.
        # So finally we do a multiplication operation between them to distinguish this time the index range is y:[0, 2*(window_size-1)*(2 * window_size-1)], x:[0, 2*(window_size-1)]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # Then sum over the last dimension and expand to a one-dimensional coordinate
        # This time the index range is [0, 2 * (window_size-1) * (2 * window_size-1) + 2 * (window_size-1)] to simplify [0, (2 * window_size - 1) * (2 * window_size - 1) - 1]
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # The window_size remains the same throughout the training, so the index does not change
        self.register_buffer("relative_position_index", relative_position_index)  # And register as a variable that does not participate in online learning

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Multi-head fusion
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # And initialized to a truncated normal distribution
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C) 
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape  #  128，16，96
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # qkv was obtained separately by unbind
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)分配给qkv 128，3，16，32

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale  # attn tensor of shape (numWindows*B, num_heads, window_size*window_size, window_size*window_size)
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        # Relative position encoding is done here by adding a batch dimension to unsqueeze
        attn = attn + relative_position_bias.unsqueeze(0)

        # Ignoring the mask for the moment, the only thing left is softmax, dropout, and V matrix multiplication like transformer, and then go through a layer of full connection layer and dropout
        # When SiftedWindowAttention is required, mask is not None, which requires adding mask to the Attention result attn and passing softmax, the value in the -100 position will be ignored after passing softmax.
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

#  Before entering SwinTransformerBlock, the one experienced on x is create_mask with dimensions [B, HW, C], so H and W need to be passed in separately
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
         # dim (int):Number of input channels.
        # input_resolution (tuple[int]):Input recovery.
        # num_heads (int):Number of attention heads.
        # window_size (int):The window size.
        # shift_size (int): Shift size of the SW-MSA.
        # mlp_ratio (float): mlp hides the ratio of dim to embedded dim.
        # qkv_bias (bool, optional): If True, adds a learnable deviation to query, key, and value. Default value: true
        # drop (float, optional): exit rate. Default value :0.0
        # attn_drop(float, optional): Attention decline rate. Default value :0.0
        # drop_path (float, optional): random depth rate. Default value :0.0
        # act_layer (nn. Module (optional): Activation layer. Default value: neural network. GELU
        # norm_layer (nn. Module, optional): Normalized layer. Default value: neural network. LayerNorm
    """

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

        # DropPath is a network regularization method for branch networks. Deep networks can be built without residuals.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # LN
        self.norm2 = norm_layer(dim)
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # shortcut is to implement residuals and then change the dimension of x to (B, H, W, C). We also make sure that the feature map is an integer multiple of window_size, so we do a padding, where the dimensions are (B, Hp, Wp, C).
    # If shift_size is 0, it means that it is W-MSA and does not need to be shifted. Then divide the window and dimensionally transform
    def forward(self, x, attn_mask):
        # x(B,L,C)
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # Residual network
        shortcut = x   # (32,64,96)
       
        x = self.norm1(x)
        x = x.view(B, H, W, C) # (32,8,8,96)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            # Shift the window. It's moving from top to bottom, from left to right, so it's negative
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C](128,4,4,96)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C](128,16,96)

        # attn_mask is passed to WindowAttention here, which will later undo window partitioning, restore the shift, and remove the padding
        # W-MSA/SW-MSA calculates Attention and uses attn_mask to distinguish between Window Attention and Shift Window Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # Window restore
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # Remove the data from the previous pad
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)  
        x = x + self.drop_path(self.mlp(self.norm2(x)))  

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
        downsample (nn.Module | None, optional): Do you need to downsample? No need in the last stage. Default: None
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
        self.shift_size = window_size // 2  

        # All blocks in the current stage
        # depth is the number of blocks in this stage, and downsample means patch merging
        # Note that there will only be one MSA in each block, either W-MSA or SW-MSA, so shift_size is 0 for W-MSA, not 0 for SW-MSA
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
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # The channel sequence is the same as that of the feature map for subsequent window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        
        # Set the slice, dividing the mask into 3 parts from the line
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # So, mask has a total of 9 regions

        # Number the mask for each area
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition window
        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1] 
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw] 
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        
        # [nW, Mh*Mw, Mh*Mw] The corresponding region of each window is obtained by the subtraction above. The same region is 0, and the different values are other values. attn_mask specifies -100 in the field other than 0
        # attn_mask: The mask used for attention. attention is calculated only between elements of the same window, and direct passes for different Windows are marked as -100 so that the original data is not affected.
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # First create a mask that does not change while the image size remains the same
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            # Prevents H and W from being odd. If it's odd, it's even after a padding in the downsample, but if you don't add one to H and one to W, you lose one. If it's even, add one to divide two and it's still the same
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W

# The whole model adopts a hierarchical design, including 4 stages. Each stage will reduce the resolution of the input feature map and expand the receptive field layer by layer like CNN.
class SwinTransformer(nn.Module):
    r""" Swin Transformer
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Swin Transformer it can be used as a general skeleton, which is used here in the classification task, and finally divided into num_classes. Default: 1000
        embed_dim (int): Patch embedding dimension
        depths (tuple(int)): Number of Swin Transformer blocks in each stage.
        num_heads (tuple(int)): The number of multi-heads used in each stage.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): How many times the hidden layer of the mlp is the input layer? Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): In pos_drop,mlp, and elsewhere. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): For each Swin Transformer, note that the dropout rate is increasing. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """
    
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

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
                     
        # Corresponds to the Patch partition and Linear Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # The dropout rate in each block is an increasing sequence
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # num_layers and stage numbers
            # dim is the dimension of the current stage, depth is how many blocks are stacked on the current stage, and drop_patch is the drop_patch of all blocks in this layer
            # downsample is Patch merging and is None on the last stage
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  
                                norm_layer=norm_layer,
                                #  Different from the paper, the stage in the code contains the Patch merging of the next layer, so there is no Patch merging in the last stage
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # In this classification task, replace cls tokens with global averaging pooling
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # Weight initialization function
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

        # Go through each stage in turn
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C] 32 1 768
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1] 32 768 1
        x = torch.flatten(x, 1) # 32 768
        x = self.head(x) # 32 3
        return x

