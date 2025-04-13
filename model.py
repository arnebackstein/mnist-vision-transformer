import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class VisionTransformer(nn.Module):
    def __init__(self, 
                 image_size=28,  
                 patch_size=2,  
                 num_classes=10, 
                 dim=64,       
                 depth=3,      
                 heads=4,   
                 mlp_dim=128,  
                 channels=1):  
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embedding = nn.Conv2d(
            in_channels=channels,
            out_channels=dim, 
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        class CustomTransformerEncoderLayer(TransformerEncoderLayer):
            def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                batch_size = src.size(0)
                seq_len = src.size(1)
                embed_dim = self.self_attn.embed_dim
                num_heads = self.self_attn.num_heads
                head_dim = embed_dim // num_heads
                
                src_reshaped = src.reshape(-1, embed_dim)
                
                qkv = torch.matmul(src_reshaped, self.self_attn.in_proj_weight.t())
                if self.self_attn.in_proj_bias is not None:
                    qkv = qkv + self.self_attn.in_proj_bias
                
                qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
                
                q = qkv[0]  # [batch_size, num_heads, seq_len, head_dim]
                k = qkv[1]  # [batch_size, num_heads, seq_len, head_dim]
                
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                
                self.attention_patterns = attn_scores
                
                return super().forward(src, src_mask, src_key_padding_mask, is_causal=is_causal)
            
            def get_attention_patterns(self):
                return self.attention_patterns
        
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.patch_embedding.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.patch_embedding.bias)
        
        nn.init.normal_(self.mlp_head[1].weight, std=0.02)
        nn.init.zeros_(self.mlp_head[1].bias)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embedding(x)  # [batch, dim, h', w']
        x = x.flatten(2)  # [batch, dim, patches] 
        x = x.transpose(1, 2)  # [batch, patches, dim]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embedding
        
        x = self.transformer(x)
        
        x = x[:, 0] # [batch, patches, dim]
        
        x = self.mlp_head(x)
        return x