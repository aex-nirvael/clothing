'''
copyright Alex Whelan 2025
code for building model
'''

import torch

class XAttn(torch.nn.Module):
    def __init__(self, in_channels, attn_dim=256, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim

        # Project input features to attention embedding space
        self.q_proj = torch.nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.k_proj = torch.nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.v_proj = torch.nn.Conv2d(in_channels, attn_dim, kernel_size=1)

        # Cross-attention
        self.attn = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)

        # Project back to original shape
        self.out_proj = torch.nn.Conv2d(attn_dim, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # x1, x2 shape: (B, C, H, W)
        B, C, H, W = x1.shape

        # Project to attention space
        q = self.q_proj(x1)  # (B, attn_dim, H, W)
        k = self.k_proj(x2)
        v = self.v_proj(x2)

        # Flatten spatial dimensions
        q = q.flatten(2).transpose(1, 2)  # (B, HW, attn_dim)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # Apply cross-attention: x1 queries, x2 keys/values
        attn_output, _ = self.attn(q, k, v)  # (B, HW, attn_dim)

        # Reshape back to (B, attn_dim, H, W)
        attn_output = attn_output.transpose(1, 2).view(B, self.attn_dim, H, W)

        # Project back to original channel size
        out = self.out_proj(attn_output)  # (B, C, H, W)
        return out


class ClothingModel(torch.nn.Module):

  def __init__(self, chan_in, chan_out):
    super(ClothingModel, self).__init__()
    self.kernel_size = 3
    self.chan_in = chan_in
    self.chan_out = chan_out

    # reference encoder layers
    self.conv1_1 = torch.nn.Sequential(torch.nn.Conv2d(self.chan_in, 8, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(8)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv1_2 = torch.nn.Sequential(torch.nn.Conv2d(8, 32, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(32)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv1_3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(64)
                                      , torch.nn.ReLU(inplace=True)
                                      )
    
    # clothing encoder layers
    self.conv2_1 = torch.nn.Sequential(torch.nn.Conv2d(self.chan_in, 8, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(8)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv2_2 = torch.nn.Sequential(torch.nn.Conv2d(8, 32, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(32)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv2_3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(64)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    # bottleneck layer
    self.xattn = XAttn(in_channels=128)

    self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(128)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    # decoder layers
    self.conv5 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(256, 128, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(128)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv6 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(192, 96, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(96)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv7 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(112, 56, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(56)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.out = torch.nn.Sequential(torch.nn.Conv2d(62, self.chan_out, self.kernel_size, stride=1, padding=1)
                                    , torch.nn.Sigmoid()
                                      )

  def forward(self, x1, x2):

    # set up skip connections
    skips_1 = []
    skips_1.append(x1)
    skips_2 = []
    skips_2.append(x2)

    # reference encoder
    x1 = self.conv1_1(x1)
    skips_1.append(x1)
    x1 = self.conv1_2(x1)
    skips_1.append(x1)
    x1 = self.conv1_3(x1)
    skips_1.append(x1)

    # clothing encoder
    x2 = self.conv2_1(x2)
    skips_2.append(x2)
    x2 = self.conv2_2(x2)
    skips_2.append(x2)
    x2 = self.conv2_3(x2)
    skips_2.append(x2)

    # bottleneck (could be x-attn?)
    x = torch.cat((x1, x2), dim=1)
    x = self.conv4(x)

    # decoder
    x = torch.cat((x, skips_1[3], skips_2[3]), dim=1)
    x = self.conv5(x)
    x = torch.cat((x, skips_1[2], skips_2[2]), dim=1)
    x = self.conv6(x)
    x = torch.cat((x, skips_1[1], skips_2[1]), dim=1)
    x = self.conv7(x)
    x = torch.cat((x, skips_1[0], skips_2[0]), dim=1)

    return self.out(x)
  

class ClothingDiscriminator(torch.nn.Module):

  def __init__(self, chan_in):
    super(ClothingDiscriminator, self).__init__()
    self.kernel_size = 3
    self.chan_in = chan_in

    self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(self.chan_in, 8, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(8)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(8, 32, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(32)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(64)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.out = torch.nn.Sequential(torch.nn.Conv2d(64, 1, self.kernel_size, stride=1, padding=1)
                                    , torch.nn.Sigmoid()
                                      )

  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    return self.out(x)