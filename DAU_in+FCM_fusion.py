import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## --ECA--
class Eca_layer(nn.Module):
    def __init__(self, channel, reduction=1, bias=True):
        super(Eca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
        )
        self.conv_du2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.feature = channel

    def forward(self, x):
        y = self.avg_pool(x)  # y:[B, C, 1, 1]
        ############
        y_Transi = y.squeeze(-1)  # y_Transi:[B, C, 1]
        Ty = y_Transi.permute(0, 2, 1)  # Ty:[B, 1, C]
        Ny = torch.matmul(Ty, y_Transi)  # Ny:[B, 1, 1]
        batch = Ny.shape[0]
        v = Ny.squeeze(-1).permute(1, 0).expand((self.feature, batch))  # [B, 1, 1]-->[B, 1]-->[1, B]-->[C, B]
        v = v.unsqueeze_(-1).permute(1, 2, 0)  # [C, B]-->[C, B, 1]-->[B, 1, C]
        v_GC = v.transpose(2, 1).unsqueeze(-1)  # [B, 1, C, 1]-->[B, C, 1, 1]
        ############
        y = self.conv_du1(y) + v_GC
        y = self.conv_du2(y)
        return x * y


##########################################################################
##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        # self.CA = ca_layer(n_feat, reduction, bias=bias)
        self.CA = Eca_layer(channel=n_feat, reduction=1, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class RLN(nn.Module):
    r"""RescaleNorm替换了模型中的所有Layernorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


#########################
#  multi-layer perceptron
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):  # ？
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


########################################
# 构成划窗注意力机制
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    """Attention————W-MHSA with Parallel Conv"""

    # Reflection Padding 和 Cropping在哪？
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        # dehazeformer-m模型用的的卷积块
        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:  # use_attn = True
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        # 把Q和K放在一个张量里？
        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    # 初始化模型参数
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)  # 值Value

        if self.use_attn:
            QK = self.QK(X)  # 查询Query、键Key
            QKV = torch.cat([QK, V], dim=1)  # [B, dim*3, H, W]

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            ########################################
            # 作者提出的Parallel Conv
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        # 没用上，self.use_attn = True
        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


################################################
# DehazeFormer Block
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None,
                 last=False):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.last = last

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()  # 没用上 mlp_norm = False

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

        #################
        self.DAU = DAU(n_feat=dim, act=nn.ReLU())
        self.FCM = SFconv(features=dim)
        #################

    def forward(self, x):
        identity = x
        # 第一个RescaleNorm层
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        # 第二个W-MHSA-PC层
        x = self.attn(x)
        # 第三个Affine层
        if self.use_attn: x = x * rescale + rebias
        # 残差连接
        x = identity + x

        ####################
        x = self.DAU(x)
        if self.last:
            x = self.FCM(x)
        ####################

        identity = x

        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)  # 没用上 mlp_norm = False

        x = self.mlp(x)

        if self.use_attn and self.mlp_norm: x = x * rescale + rebias  # 没用上 mlp_norm = False

        x = identity + x
        return x


################################################
# transformer块，包含多个W-MHSA
class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        # 文中作者有提到对一部分区域不适用transformer快进行学习，这里就是对不需要学习的块打上“标签”
        # 各个函数中的 use_attn 就是根据该标签来判断要不要使用transformer
        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        # self.blocks = nn.ModuleList([TransformerBlock(use_attn=use_attns[i]) for i in range(depth)])
        # 根据depth构建相应数量的TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        # 依次通过所构建的n个TransformerBlock
        for blk in self.blocks:
            x = blk(x)
        return x


######################################
# 预处理，将输入图像的channel进行投影 3 * H * W--> C * H * W
# 将图像分为多个patch，同时也承担了上下采样工作
# 尝试改为GLASM
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


# 	我觉得并没有做到一个图片切割为多个patch
# 	self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
#					 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

#   很常规的下采样，将做到了图片的缩小
# 	self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
#					  = nn.Conv2d(24, 48, kernel_size=2, stride=2, padding=0)

######################################
# 预处理，将输出图像的channel进行反投影 C * H * W--> 3 * H * W
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


#################################################
# 选择融合机制
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        # in_feats = [x, skip_x] x为上采样后的输出，skip是对应scale层下采样的输出,两者的shape都为[B, embed_dims[i], H, W]
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)  # 将两个输入合并 in_feats.shape = [B, 2*embed_dims[i], H, W]
        in_feats = in_feats.view(B, self.height, C, H, W)  # in_feats.shape = [B, 2, embed_dims[i], H, W]

        feats_sum = torch.sum(in_feats, dim=1)  # in_feats.shape = [B, embed_dims[i], H, W]
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


##############################
# SKFF
class SKFF(nn.Module):
    def __init__(self, dim, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(dim, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, dim, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


# --------------------------------------------------------------------------------
# 隐式频率注意力机制
class SFconv(nn.Module):
    def __init__(self, features, M=4, r=2, L=32, PWC_use=False) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features
        self.DPWC_use = PWC_use

        self.PWconv = nn.Conv2d(features, features, 1, bias=False)
        self.PWconv_h = nn.Conv2d(d, features, 1, bias=False)
        self.DWPConv = nn.Sequential(
			nn.Conv2d(features, features, 1),
			nn.Conv2d(features, features, kernel_size=3, padding=3//2, groups=features, padding_mode='reflect')
		)
        self.act = nn.ReLU(inplace=True)
        self.convh = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.convm = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convll = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros((1, features, 1, 1)), requires_grad=True)

    def forward(self, x):
        # x = [浅层, 深层]
        B, C, H, W = x[0].shape
        lowlow = x[1]
        middle = x[0]
        low = self.convl(lowlow)
        high = self.convh(middle)

        emerge = low + middle + high + lowlow
        emerge = self.act(self.PWconv(emerge))
        emerge = self.gap(emerge)

        # 或许可以把三个FC层都换为一个DWConv+PWConv，DW关注通道，PW关注空间
        if not self.DPWC_use:
            fea_z = self.fc(emerge)
            high_att = self.fcs[0](fea_z)
            middle_att = self.fcs[1](fea_z)
            low_att = self.fcs[2](fea_z)
            lowlow_att = self.fcs[3](fea_z)
        else:
            fea_z = self.act(self.DWPConv_h(emerge))
            high_att = self.PWconv_h(fea_z)
            middle_att = self.PWconv_h(fea_z)
            low_att = self.PWconv_h(fea_z)
            lowlow_att = self.PWconv_h(fea_z)

        attention_vectors = torch.cat([high_att, middle_att, low_att, lowlow_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, middle_att, low_att, lowlow_att = torch.chunk(attention_vectors, 4, dim=1)

        fea_high = high * high_att
        fea_middle = middle * middle_att
        fea_low = low * low_att
        fea_lowlow = lowlow * lowlow_att
        out = self.out(fea_high + fea_middle + fea_low + fea_lowlow)
        return out * self.gamma + x


################################################
# 模型框架
class DehazeFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 refin_depth=2):
        super(DehazeFormer, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        # 第一个3*3 Conv：[B, 3, 256, 256] --> [B, embed_dims[0], 256, 256]
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        # 第一个Transformer块
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        # 下采样 [B, embed_dims[0], H, W] --> [B, embed_dims[1], H, W]
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # self.patch_merge1 = Downsample(
        #     dim=embed_dims[0], num_head=8)

        ####################
        # DAU模块，与SKFF级联使用
        self.DAU_SKFF1 = DAU(n_feat=embed_dims[0], act=nn.ReLU())
        self.DAU_SKFF2 = DAU(n_feat=embed_dims[1], act=nn.ReLU())

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        # self.patch_merge2 = Downsample(
        #     dim=embed_dims[1], num_head=8)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        # self.patch_split1 = Upsample(
        #     dim=embed_dims[2], num_head=8)

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SFconv(features=embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        # self.patch_split2 = Upsample(
        #     dim=embed_dims[3], num_head=8)

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SFconv(features=embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        #######################
        # Refin-block
        self.refine = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=refin_depth,
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    #

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)  # C:24
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)  # C:48
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)  # C:96
        x = self.layer3(x)
        x = self.patch_split1(x)  # C:48

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)  # C:24

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        # x = self.refine(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)

        # soft_refine
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]

        # # 自己写的增强模块
        # x = self.refine(feat)
        # x = self.patch_unembed(x)
        return x


def dehazeformer_t():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1 / 2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_s():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_b():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_d():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[32, 32, 32, 16, 16],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_w():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_m():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[12, 12, 12, 6, 6],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])


def dehazeformer_l():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 12, 12],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
