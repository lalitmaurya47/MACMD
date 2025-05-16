import torch
import torch.nn as nn
import torch.nn.functional as F
from ccm.ccm import CCMix
from ccm.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D


inplace = True

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)

    
class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=False, has_skip=False, exp_ratio=1.0, norm_layer='bn_2d',
                 dw_ks=3, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.):
        super().__init__()
        self.has_skip =has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='bn_2d', act_layer='relu', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(x)
        #x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



class EfficientUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(EfficientUpConvBlock, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # lightweight upsampling
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise conv
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)  # pointwise conv
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.block(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



class AttentionPoolingModulation(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(AttentionPoolingModulation, self).__init__()
        self.out_channels = out_channels
        self.in_channels_list = in_channels_list

        self.channel_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])

        # Define up-conv blocks (placeholders; real blocks are instantiated at runtime)
        self.up_blocks = nn.ModuleList([nn.Identity() for _ in in_channels_list])

        self.attention_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, 1, kernel_size=1)
        )

        self.modulation_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, *features):
        if len(features) < 2:
            raise ValueError("At least two input features are required.")

        # Compute target size
        target_h = max([feat.shape[2] for feat in features])
        target_w = max([feat.shape[3] for feat in features])
        target_size = (target_h, target_w)

        resized_features = []
        for idx, feat in enumerate(features):
            x = self.channel_proj[idx](feat)
            h, w = x.shape[2], x.shape[3]
            if (h, w) != target_size:
                scale_factor = (target_size[0] // h, target_size[1] // w)
                up_block = EfficientUpConvBlock(self.out_channels, self.out_channels, scale_factor).cuda()
                x = up_block(x)
            resized_features.append(x)

        stacked = torch.stack(resized_features, dim=1)  # (B, N, C, H, W)
        B, N, C, H, W = stacked.shape

        attn_scores = torch.stack([self.attention_proj(stacked[:, i]) for i in range(N)], dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * stacked, dim=1)

        pooled_sig = torch.sigmoid(pooled)
        modulated = []
        for i in range(N):
            x = stacked[:, i]
            x_sig = torch.sigmoid(x)
            pooled_mod = x_sig * pooled
            x_mod = pooled_sig * x
            inter = pooled_mod * x_mod
            proj = self.modulation_proj(inter)
            modulated.append(proj * x)

        out = torch.mean(torch.stack(modulated, dim=0), dim=0)
        return out
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionBlock, self).__init__()
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2 )
        self.channel_attention = ChannelAttention(in_channels * 2)
        self.spatial_attention = SpatialAttention()
        self.output_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False)

    def forward(self, f1, f2, f3):
        # Concatenate features along channel dimension: (B, 3C, H, W)
        fused = torch.cat([f1, f2, f3], dim=1)

        # Project back to original channel size
        x = self.bn(self.fusion_conv(fused))

        # Apply Channel Attention
        x = self.channel_attention(x)

        # Apply Spatial Attention
        sa = self.spatial_attention(x)
        x = x * sa

        # Final projection (optional)
        x = self.output_conv(x)

        return x

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)


class HDConv(nn.Module):
    '''
        *dilation* indicates the rate of expansion
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=(1, 2, 3, 5)):
        super(HDConv, self).__init__()
        self.dilation_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding + (dilation[0] - 1) * (kernel_size - 1) // 2,
                                    groups=4, bias=bias, dilation=dilation[0], stride=stride)

        self.dilation_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding + (dilation[1] - 1) * (kernel_size - 1) // 2,
                                    groups=4, bias=bias, dilation=dilation[1], stride=stride)

        self.dilation_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding + (dilation[2] - 1) * (kernel_size - 1) // 2,
                                    groups=4, bias=bias, dilation=dilation[2], stride=stride)

        self.dilation_4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    padding=padding + (dilation[3] - 1) * (kernel_size - 1) // 2,
                                    groups=4, bias=bias, dilation=dilation[3], stride=stride)

    def forward(self, x):
        feature1_1, feature1_2, feature1_3, feature1_4 = torch.chunk(self.dilation_1(x), 4, dim=1)
        feature2_1, feature2_2, feature2_3, feature2_4 = torch.chunk(self.dilation_2(x), 4, dim=1)
        feature3_1, feature3_2, feature3_3, feature3_4 = torch.chunk(self.dilation_3(x), 4, dim=1)
        feature4_1, feature4_2, feature4_3, feature4_4 = torch.chunk(self.dilation_4(x), 4, dim=1)

        out_1 = feature1_1 + feature2_2 + feature3_3 + feature4_4
        out_2 = feature1_2 + feature2_3 + feature3_4 + feature4_1
        out_3 = feature1_3 + feature2_4 + feature3_1 + feature4_2
        out_4 = feature1_4 + feature2_1 + feature3_2 + feature4_3
        return torch.cat((out_1, out_2, out_3, out_4), dim=1)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)




class Attention_block(nn.Module):
    def __init__(self,in_channels):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            HDConv(in_channels, in_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(in_channels)
            )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g):
        g1 = self.W_g(g)
        psi = self.relu(g1)
        psi = self.psi(psi)

        return g1*psi

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



# Existing conv_block
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            HDConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            HDConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)



# Combined Block: conv -> Channel Attention -> Spatial Attention
class AttentionConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(AttentionConvBlock, self).__init__()
        self.conv = conv_block(ch_in, ch_out)
        self.ca = ChannelAttention(ch_out)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)
        sa = self.sa(x)
        x = x * sa
        return x
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)





    


class MACMD(nn.Module):
    def __init__(self,  embed_dims = [144, 288, 576, 1152], img_size = 224) -> None:
        super(MACMD, self).__init__()    
       
        

        self.embed_dims = embed_dims
        
        self.acb = AttentionConvBlock(self.embed_dims[3], self.embed_dims[3])
        self.apm = AttentionPoolingModulation([self.embed_dims[0], self.embed_dims[1], self.embed_dims[2], self.embed_dims[3]], self.embed_dims[0]).cuda()
        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size//4)
        self.ag1 = Attention_block(self.embed_dims[0])
        self.ag2 = Attention_block(self.embed_dims[1])
        self.ag3 = Attention_block(self.embed_dims[2])
        self.ag4 = Attention_block(self.embed_dims[3])
        

        
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)             
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0], norm_in=False, has_skip=False, exp_ratio=1.0,dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 32, norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)    
        self.ffb = FeatureFusionBlock(self.embed_dims[0])
        
        
        
        #self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x1, x2, x3, x4):
        enc1,enc2,enc3,enc4 = x1, x2, x3, x4
        
        # Mulit-dilated Contextual attention Gate (MCAG)
        x1, x2, x3, x4 = self.ag1(x1), self.ag2(x2), self.ag3(x3), self.ag4(x4)   
        
        #Multi-dilated Enhanced Attention Block (MEAB)
        x4 = self.acb(x4)
        
        # Attention Pooling Modulation (APM)
        up1 = self.apm(x1,x2,x3,x4) 

        #Multi scale Cross-Channel Mix (MSCCM)
        enc3,enc2,enc1=self.ccm([enc3,enc2,enc1])

        #Segmentation Head (SegHead)
        dec3 = self.decoder1(x4)
      
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        
        ffb0 = self.ffb(dec1, enc1, up1)
    
        dec0 = self.decoder4(ffb0)
       
        
        return dec0, dec1, dec2
    



# if __name__ == "__main__":
#     with torch.no_grad():
#         model = MACMD().cuda()
#         x = torch.randn(1, 3, 224, 224).cuda()
#         # out, out1, out2 = model(x)
#         # print(out.shape, out1.shape, out2.shape)
#         out = model(x)
#         print(out.shape)