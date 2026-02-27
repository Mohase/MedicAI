# FDR block
# Patch Embedding
# Transformer encoder
# Dual decoder
# FDR-TransUnet

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================
# 1. FDR BLOCK (Feature Double Reuse)
# =============================

class FDRBlock(nn.Module):
    """
    Feature Double Reuse Block
    - Reuse 1: Dense concatination (DenseNet style) -> concatenates features to reuse info.
    - Reuse 2: Residual addition   (ResNet style) -> adds input as a skip connection. 
    """

    def __init__(self, in_channels, out_channels):
        super(FDRBlock, self).__init__()

        # First convolution: processe input
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution: processes concatenated features
        # Input channels = out_channels + in_channels (due to concatination)
        self.conv2 = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: adjusts channels if input != output channels
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x # Save for residual connection

        # First conv block 
        out = F.relu(self.bn1(self.conv1(x)))

        # Dense connection: concatenate input with conv output
        out = torch.cat([x, out], dim=1) # Reuse #1

        # Second conv block
        out = F.relu(self.bn2(self.conv2(out)))

        # Residual connection: add original input 
        out = out + self.skip(identity) # Reuse #2

        return out 

# =============================
# 2. PATCH EMBEDDING
# =============================

class PatchEmbedding(nn.Module):
    """
    Converts CNN feature maps into patch embeddings (sequence) for the transformer 
    - Splits feature map into patches
    - Projects patches to embedding dimension
    - Adds positional embeddings
    """

    def __init__(self, in_channels,embed_dim, patch_size, num_patches):
        super(PatchEmbedding, self).__init__()

        # Project patches to embedding dimensions using a conv layer
        # Conv with kernel_size=patch_size and stride=patch_size as patch extraction
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional embeddings (one for each patch)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Class token: a learnable token prepended to the sequence
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]

        # Project and reshape : (B, C , H, W) -> (B, embed_dim, H', W') -> (B, num_patches, embed_dim)
        # B = batch_size
        x = self.projection(x)  # (B, embed_dim, H', W') # turns each patch into one vector
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1,2)    # (B, num_patches, embed_dim)

        # Expand class token for batch and prepend 
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1) # (B, num_patches + 1, embed_dim)

        # Add positional embeddings (excluding class token position)
        x[:, 1:, :] = x[:, 1:, :] + self.positional_embeddings

        return x


# =============================
# 3. TRANSFORMER ENCODER
# =============================

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head self-attention: each token attends to all tokens."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)  # Q, K, V in one projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (Batch, number of tokens -> Class token + num of pacthes (each patch data = 1 token), embed_dim)
        x = self.proj(x) # Here proj is a linear layer that mixes each token's vector into a new vector of the same size
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderBlocks (as many as there are layers)."""

    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =============================
# 4. DUAL DECODER
# =============================

class DualDecoder(nn.Module):
    """
    Two decoder paths for deep supervision:
    - Path A: Patch tokens + encoder skip connections -> local / spatial detail. (patch tokens)
    - Path B: Class token expanded to spatial -> global context. (class token)
    """
    def _make_decoder_block(self, in_ch, skip_ch, dec_ch):
        """
        One double_conv block: concat input + skip, then conv -> BN -> ReLU twice.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, dec_ch, 3, padding=1),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        )

    def __init__(self, embed_dim, skip_channels, decode_channels, out_size_h, out_size_w, patch_grid_h, patch_grid_w):
        super(DualDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_grid_h = patch_grid_h
        self.patch_grid_w = patch_grid_w
        self.out_size_h = out_size_h
        self.out_size_w = out_size_w

        # Path A: patch tokens -> expand to spatial, then same structure
        # Shortcut connections are used between each block -> residual 
        self.path_a_blocks = nn.ModuleList()
        self.path_a_shortcuts = nn.ModuleList()
        in_ch = embed_dim # input channels
        for skip_ch, dec_ch in zip(skip_channels, decode_channels):
            self.path_a_blocks.append(self._make_decoder_block(in_ch, skip_ch, dec_ch))
            self.path_a_shortcuts.append(nn.Conv2d(in_ch + skip_ch, dec_ch, 1))
            in_ch = dec_ch
            
        # Path B: class token -> expand to spatial, then same structure
        self.path_b_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.path_b_blocks = nn.ModuleList()
        self.path_b_shortcuts = nn.ModuleList()
        in_ch = embed_dim
        for skip_ch, dec_ch in zip(skip_channels, decode_channels):
            self.path_b_blocks.append(self._make_decoder_block(in_ch, skip_ch, dec_ch))
            self.path_b_shortcuts.append(nn.Conv2d(in_ch + skip_ch, dec_ch, 1))
            in_ch = dec_ch

        self.head_a = nn.Conv2d(decode_channels[-1], 1, 1)
        self.head_b = nn.Conv2d(decode_channels[-1], 1, 1)

    def _upsample_and_concat(self, x, skip, block, shortcut):
        """ Upsample, concat with skip, double conv, then add residual as per paper. """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        concat = torch.cat([x, skip], dim=1)
        return block(concat) + shortcut(concat)

    def forward(self, patch_tokens, class_token, encoder_skips):
        B = patch_tokens.shape[0]
        # Path A: (B, N, D) -> (B, D, h, w)
        # B = batch size
        # N = number of tokens (num_patches)
        # D = embed_dim (length of each token vector)
        a = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.patch_grid_h, self.patch_grid_w)
        for block, shortcut, skip in zip(self.path_a_blocks, self.path_a_shortcuts, encoder_skips):
            a = self._upsample_and_concat(a, skip, block, shortcut)
        logits_a = self.head_a(a)
        logits_a = F.interpolate(logits_a, size=(self.out_size_h, self.out_size_w), mode="bilinear", align_corners=False)

        # Path B: class token (B, 1, D) -> (B, D, h, w)
        b = class_token.transpose(1, 2).reshape(B, self.embed_dim, 1, 1)
        b = b.repeat(1, 1, self.patch_grid_h, self.patch_grid_w)
        b = self.path_b_proj(b)
        for block, shortcut, skip in zip(self.path_b_blocks, self.path_b_shortcuts, encoder_skips):
            b = self._upsample_and_concat(b, skip, block, shortcut)
        logits_b = self.head_b(b)
        logits_b = F.interpolate(logits_b, size=(self.out_size_h, self.out_size_w), mode="bilinear", align_corners=False)

        return logits_a, logits_b 


# =============================
# 5. FDR-TRANSUNET (main model)
# =============================

class FDRTransUNet(nn.Module):
    """
    Full model per paper: Encoder (FDR) -> Patch Embed -> Transformer -> Dual Decoder.
    Deep supervision: Two outputs (Path A + Path B); combined mask; residual shortcuts in decoder.
    """
    def __init__(
        self,
        in_channels=1,
        encoder_channels=(32, 64, 128, 256),
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        patch_size=1,
        mlp_ratio=4.0,
        dropout=0.1,
        input_h=256,
        input_w=256,
    ):

        super(FDRTransUNet, self).__init__()
        self.input_h, self.input_w = input_h, input_w
        self.encoder_channels = list(encoder_channels)
        self.embed_dim = embed_dim

        # Stem: 256 -> 128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels[0], 7, stride=2, padding=3),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder: FDR stages + downsample -> 128 -> 64 -> 32 -> 16
        self.encoder_stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = encoder_channels[0]
        for out_ch in encoder_channels[1:]:
            self.encoder_stages.append(FDRBlock(ch, out_ch))
            self.downsample.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        self.encoder_stages.append(FDRBlock(ch,ch))
        self.bottleneck_channels = ch

        self.patch_grid_h = input_h // 16
        self.patch_grid_w = input_w // 16
        num_patches = self.patch_grid_h * self.patch_grid_w

        self.patch_embed = PatchEmbedding(self.bottleneck_channels, embed_dim, patch_size, num_patches)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)

        skip_ch = [encoder_channels[-1], encoder_channels[-2], encoder_channels[-3], encoder_channels[-4]]
        decode_ch = [256, 128, 64, 32]
        self.decoder = DualDecoder(
            embed_dim=embed_dim,
            skip_channels=skip_ch,
            decode_channels=decode_ch,
            out_size_h=input_h,
            out_size_w=input_w,
            patch_grid_h=self.patch_grid_h,
            patch_grid_w=self.patch_grid_w,
        )

    def _encoder_forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for stage, down in zip(self.encoder_stages[:-1], self.downsample):
            x = stage(x)
            skips.append(x)
            x = down(x)
        x = self.encoder_stages[-1](x)
        return x, skips
    

    def forward(self, x):
        enc_out, skips = self._encoder_forward(x)
        encoder_skips = [skips[-1], skips[-2], skips[-3], skips[-4]]

        patches = self.patch_embed(enc_out)
        tokens = self.transformer(patches)
        class_tok = tokens[:, 0:1, :]
        patch_tok = tokens[:, 1:, :]

        logits_a, logits_b = self.decoder(patch_tok, class_tok, encoder_skips)
        combined = (logits_a + logits_b) / 2.0

        return combined, logits_a, logits_b
        
