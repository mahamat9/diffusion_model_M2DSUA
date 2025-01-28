import torch
import torch.nn as nn


def get_time_embedding(time_steps, t_emb_dim):
    factor = 10000 ** (torch.arange(start=0, end=t_emb_dim // 2, device=time_steps.device) / (t_emb_dim // 2))

    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),  # Why 8?
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),  # Why 8?
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x

        # Resnet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),  # Why 8?
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),  # Why 8?
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )
        ])


        self.t_emb_layer = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),  # Why 8?
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),  # Why 8?
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )
        ])


        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        ])

    def forward(self, x, t_emb):
        out = x

        # first Resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layer[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # second ResnetBlock
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layer[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),  # Why 8?
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),  # Why 8?
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        out = x

        # Resnet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        return out


class Unet(nn.Module):
    def __init__(self, im_channels):
        super().__init__()
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.up_channels = [128, 64, 32, 16]
        self.t_emb_dim = 128
        self.down_sample = [True, True, False]
        self.up_sample = [False, True, True]

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=(3, 3), padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, down_sample=self.down_sample[i], num_heads=4))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=4))

        self.ups = nn.ModuleList([])
        for i in range(len(self.up_channels) - 1):
            self.ups.append(UpBlock(self.up_channels[i] * 2, self.up_channels[i+1], self.t_emb_dim, up_sample=self.up_sample[i], num_heads=4))

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out