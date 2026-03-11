# FDR block
# FDR module
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
    Single FDR block with pre-activation bottleneck architecture.
    Paper: "each FDR block is the pre-activation architecture that
    sequentially consists of 1×1, 3×3, 1×1 convolution layers and
    corresponding BN and ReLU layers."

    Produces `growth_rate` output channels.
    The dense connection (concatenation) and residual addition (Eq. 3)
    are handled by the enclosing FDRModule.
    """

    def __init__(self, in_channels, growth_rate):
        super(FDRBlock, self).__init__()
        inter_channels = 4 * growth_rate

        # Pre-activation: BN → ReLU → Conv (applied in forward)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, growth_rate, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out


# =============================
# 1b. FDR MODULE (wraps multiple FDR blocks)
# =============================

class FDRModule(nn.Module):
    """
    One FDR module = multiple FDR blocks with dense + residual connections.

    Paper Eq. (3): XL = HL([X0, X1, ..., Xl-1]) + XL-1
    - Dense connection  (Reuse #1): each block receives the concatenation
      of all previous feature maps [X0, X1, ..., XL-1] as input.
    - Residual connection (Reuse #2): each block's output is added to
      the previous block's output (XL-1).

    A transition layer (BN → ReLU → 1×1 conv) compresses the final
    concatenated features to `out_channels`.

    Args:
        in_channels:  channels entering this module (X0)
        out_channels: channels leaving this module (after transition)
        growth_rate:  channels produced by each FDR block
        num_blocks:   number of FDR blocks in this module
    """

    def __init__(self, in_channels, out_channels, growth_rate, num_blocks):
        super(FDRModule, self).__init__()
        self.blocks = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        concat_ch = in_channels
        res_ch = in_channels          # first block's residual is X0

        for _ in range(num_blocks):
            self.blocks.append(FDRBlock(concat_ch, growth_rate))
            # Project XL-1 to growth_rate channels if dimensions differ
            if res_ch != growth_rate:
                self.residual_projs.append(
                    nn.Conv2d(res_ch, growth_rate, kernel_size=1, bias=False)
                )
            else:
                self.residual_projs.append(nn.Identity())
            concat_ch += growth_rate  # dense concat grows channel count
            res_ch = growth_rate      # subsequent residuals are growth_rate

        self.transition = nn.Sequential(
            nn.BatchNorm2d(concat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_ch, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        features = [x]
        prev = x
        for block, res_proj in zip(self.blocks, self.residual_projs):
            dense_in = torch.cat(features, dim=1)         # [X0, ..., XL-1]
            out = block(dense_in) + res_proj(prev)         # Eq. (3)
            features.append(out)
            prev = out
        return self.transition(torch.cat(features, dim=1))


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

    def __init__(self, in_channels, embed_dim, patch_size, num_patches):
        super(PatchEmbedding, self).__init__()

        # Project patches to embedding dimensions using a conv layer
        # Conv with kernel_size=patch_size and stride=patch_size as patch extraction
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Paper Eq. (4): Epos ∈ R^(N+1)×D — covers class token + all patches
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

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

        # Add positional embeddings to ALL tokens including class token (Eq. 4)
        x = x + self.positional_embeddings

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
        # Bias trick: start with ~0.12 probability instead of ~0.5 (sigmoid(-2) ≈ 0.12)
        with torch.no_grad():
            self.head_a.bias.fill_(-1.0)
            self.head_b.bias.fill_(-1.0)

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
        embed_dim=768,
        growth_rate=64,
        num_heads=12,
        num_layers=12,
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

        # Encoder: 4 FDR modules (3 with downsampling + 1 bottleneck)
        # Paper: "Our encoder consists of four FDR modules"
        # "the larger growth rate has fewer FDR block depths"
        self.encoder_stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = encoder_channels[0]
        for out_ch in encoder_channels[1:]:
            num_blocks = max(2, out_ch // growth_rate)
            self.encoder_stages.append(FDRModule(ch, out_ch, growth_rate, num_blocks))
            self.downsample.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch

        # 4th FDR module (bottleneck, no downsampling after)
        num_blocks = max(2, ch // growth_rate)
        self.encoder_stages.append(FDRModule(ch, ch, growth_rate, num_blocks))
        self.bottleneck_channels = ch

        self.patch_grid_h = input_h // 16
        self.patch_grid_w = input_w // 16
        num_patches = self.patch_grid_h * self.patch_grid_w

        self.patch_embed = PatchEmbedding(self.bottleneck_channels, embed_dim, patch_size, num_patches)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)

        # Skip channels from 4 FDR modules (reversed: bottleneck first, stem-adjacent last)
        # Modules output: encoder_channels[1], encoder_channels[2], ..., encoder_channels[-1], encoder_channels[-1]
        module_out_ch = list(encoder_channels[1:]) + [encoder_channels[-1]]
        skip_ch = list(reversed(module_out_ch))
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
        for stage, down in zip(self.encoder_stages[:-1], self.downsample):
            x = stage(x)
            skips.append(x)
            x = down(x)
        x = self.encoder_stages[-1](x)
        skips.append(x)  # 4th skip from bottleneck FDR module
        return x, skips
    

    def forward(self, x):
        enc_out, skips = self._encoder_forward(x)
        encoder_skips = list(reversed(skips))  # [16×16, 32×32, 64×64, 128×128]

        patches = self.patch_embed(enc_out)
        tokens = self.transformer(patches)
        class_tok = tokens[:, 0:1, :]
        patch_tok = tokens[:, 1:, :]

        logits_a, logits_b = self.decoder(patch_tok, class_tok, encoder_skips)
        combined = (logits_a + logits_b) / 2.0

        return combined, logits_a, logits_b
