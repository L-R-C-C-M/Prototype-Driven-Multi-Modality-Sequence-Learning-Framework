import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from model.SequenceImageEnoder import SequenceImageEnoder, SequenceTransformer

class ModalityProjection(nn.Module):
    def __init__(self, dim=512,dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)  
        )

    def forward(self, x):  
        return self.proj(x)

class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DualModalityTransformerEncoder(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))

        # Self-attention modules per layer
        self.ln1_self = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ln2_self = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.self_attn1 = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout, batch_first=False) for _ in range(num_layers)])
        self.self_attn2 = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout, batch_first=False) for _ in range(num_layers)])
        self.ffn_self1 = nn.ModuleList([FeedForward(dim, dim*4, dropout) for _ in range(num_layers)])
        self.ffn_self2 = nn.ModuleList([FeedForward(dim, dim*4, dropout) for _ in range(num_layers)])

        # Cross-attention modules per layer
        self.ln1_cross = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ln2_cross = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.cross_attn1 = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout, batch_first=False) for _ in range(num_layers)])
        self.cross_attn2 = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, dropout, batch_first=False) for _ in range(num_layers)])
        self.ffn_cross1 = nn.ModuleList([FeedForward(dim, dim*4, dropout) for _ in range(num_layers)])
        self.ffn_cross2 = nn.ModuleList([FeedForward(dim, dim*4, dropout) for _ in range(num_layers)])

        self.final_output_dim = dim 
    def forward(self, x1, x2):

        x1 = x1.permute(1, 0, 2)  
        x2 = x2.permute(1, 0, 2)  
        B = x1.size(1)
        cls1 = self.cls_token1.expand(-1, B, -1) 
        cls2 = self.cls_token2.expand(-1, B, -1)
        x1 = torch.cat([cls1, x1], dim=0)  
        x2 = torch.cat([cls2, x2], dim=0)
        for i in range(self.num_layers):
            x1_ln = self.ln1_self[i](x1)
            sa1_out, sa1_weights = self.self_attn1[i](x1_ln, x1_ln, x1_ln, need_weights=True)
            x1 = x1 + sa1_out
            x1 = x1 + self.ffn_self1[i](self.ln2_self[i](x1))

            x2_ln = self.ln1_self[i](x2)
            sa2_out, sa2_weights = self.self_attn2[i](x2_ln, x2_ln, x2_ln, need_weights=True)
            x2 = x2 + sa2_out
            x2 = x2 + self.ffn_self2[i](self.ln2_self[i](x2))

            cls1 = x1[0:1]
            cls2 = x2[0:1]

            cls1_ln = self.ln1_cross[i](cls1)
            cls2_ln = self.ln1_cross[i](cls2)
            cls1_cross, cls1_weights = self.cross_attn1[i](cls1_ln, x2[:], x2[:], need_weights=True)
            cls2_cross, cls2_weights = self.cross_attn2[i](cls2_ln, x1[:], x1[:], need_weights=True)

            cls1 = cls1 + cls1_cross
            cls1 = cls1 + self.ffn_cross1[i](self.ln2_cross[i](cls1))
            cls2 = cls2 + self.ffn_cross2[i](self.ln2_cross[i](cls2))

            x1 = torch.cat([cls1, x1[1:]], dim=0)
            x2 = torch.cat([cls2, x2[1:]], dim=0)


        return x1, x2 

class CLSFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, cls1, cls2):

        weight = torch.clamp(self.alpha, 0.0, 1.0)
        fused_cls = weight * cls1 + (1.0 - weight) * cls2
        return fused_cls

class HelpClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU()
        )
        self.class_head = nn.Linear(64, num_classes)

    def forward(self, fused_cls_token):
        feat = self.classifier(fused_cls_token)  
        logits = self.class_head(feat)          
        return logits
   
class PETCTSequenceTransformer(torch.nn.Module):
    def __init__(self, arch, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super(PETCTSequenceTransformer, self).__init__()
        self.ct_sequence_image_encoder = SequenceImageEnoder(arch, d_model, dropout)
        self.pet_sequence_image_encoder = SequenceImageEnoder(arch, d_model, dropout)
        self.projct=ModalityProjection()
        self.projpet=ModalityProjection()
        self.transformer_encoder = DualModalityTransformerEncoder(d_model, nhead, num_layers, dropout=dropout)
        self.CLSFusion = CLSFusion()
        self.help_classifier=HelpClassifier()
        self.final_output_dim = 512
    def forward(self, x1, x2):
        x1 = self.ct_sequence_image_encoder(x1)
        x2 = self.pet_sequence_image_encoder(x2)
        x1, x2 = self.transformer_encoder(x1, x2)
        cls_pet = x2[0]  
        cls_ct  = x1[0]   
        fused_cls = self.CLSFusion(cls_pet, cls_ct)
        logits = self.help_classifier(fused_cls)

        fused_cls = F.normalize(fused_cls, dim=-1)

        return fused_cls,logits

if __name__ == "__main__":
    model = PETCTSequenceTransformer(arch='resnet18', d_model=512, nhead=8, num_layers=4, dropout=0.1)
    x1 = torch.randn(1, 64, 1, 224, 224)
    x2 = torch.randn(1, 64, 1, 224, 224,3)
    x,logits = model(x1, x2)
    print(x.shape)
    print(logits.shape)