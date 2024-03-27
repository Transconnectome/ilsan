import open_clip
import torch 
import torch.nn as nn
from timm.models.layers import drop_path, to_3tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class OpenClip_ViT(nn.Module): 
    def __init__(self, 
                 model_name=None,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=1, 
                 num_heads=12, 
                 drop_rate=0., 
                 use_abs_pos_emb=True, 
                 use_shared_rel_pos_bias=False,
                 use_checkpoint=False,
                 use_lora=False):
        super().__init__()
        # load clip 
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.transformer = None
        self.num_features = self.clip.visual.conv1.out_channels
        # patch embedding
        self.patch_embed_3d = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.num_features)
        num_patches = self.patch_embed_3d.num_patches
        
        if use_abs_pos_emb:
            self.pos_embed_3d = nn.Parameter(torch.zeros(1, num_patches + 1, self.num_features))
        else:
            self.pos_embed_3d = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.use_checkpoint = use_checkpoint

        if self.pos_embed_3d is not None:
            trunc_normal_(self.pos_embed_3d, std=.02)


    def forward(self, x): 
        x = self.patch_embed_3d(x)
        batch_size, seq_len, _ = x.size() # torch.Size([16, 4096, 1408])

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # use pretrained pos embeding for rest modalities
        if self.pos_embed_3d is not None:
            x = x + self.pos_embed_3d

        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


def create_open_clip_vit_b(img_size=256,patch_size=28,drop_path_rate=0.4, use_checkpoint=False,  use_lora=False):
    model = OpenClip_ViT(
        model_name="ViT-B-16",
        img_size=img_size,
        patch_size=patch_size
    )
    return model 

def create_open_clip_vit_l(img_size=256,patch_size=28,drop_path_rate=0.4, use_checkpoint=False,  use_lora=False):
    model = OpenClip_ViT(
        model_name="ViT-L-14",
        img_size=img_size,
        patch_size=patch_size
    )
    return model