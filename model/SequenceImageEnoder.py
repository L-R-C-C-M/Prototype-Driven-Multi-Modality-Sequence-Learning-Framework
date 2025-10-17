import pdb
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(torch.nn.Module):
    def __init__(self, arch):
        super(ImageEncoder, self).__init__()

        if arch == 'resnet18':
            self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'resnet50':
            self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'swin_v2_t':
            self.model = torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
            self.n_features = self.model.head.in_features
            self.model.head = torch.nn.Identity()
        elif arch == 'convnext_t':
            self.model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            self.n_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Identity()
    def forward(self, x):
        return self.model(x)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequenceImageEnoder(torch.nn.Module):
    def __init__(self, arch, d_model=512, dropout=0.1):
        super(SequenceImageEnoder, self).__init__()
        self.image_encoder = ImageEncoder(arch)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.size())==6:
            x=x.squeeze(dim=2)
            x = x.permute(0, 1, 4, 2, 3)
            x = x.view(batch_size * 64, 3, 224, 224)
        if x.size(2)==1:    
            x = x.repeat(1, 1, 3, 1, 1)  
            x = x.view(batch_size * 64, 3, 224, 224) 
        image_features = self.image_encoder(x)  
        image_features = image_features.view(batch_size, 64, -1) 

        position_embeddings = self.positional_encoding(image_features)

        return position_embeddings

class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, return_layers=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.return_layers = return_layers if return_layers is not None else [num_layers] # 默认为最后一层
       
        self.layer_mappers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(len(self.return_layers))
        ])
        self.layer_weights = nn.Parameter(torch.ones(len(self.return_layers)))

    def forward(self,x):
        x = x.permute(1, 0, 2) 
        outputs = []
        for idx, layer in enumerate(self.encoder_layers, 1):
            x = layer(x)
            if idx in self.return_layers:
                outputs.append(x)
        outputs = [out.permute(1, 0, 2) for out in outputs]
        if len(outputs)==1:
            return outputs[0]
        mapped = []
        for i, out in enumerate(outputs):
            out_mapped = self.layer_mappers[i](out)
            mapped_out = F.layer_norm(out_mapped, normalized_shape=(out_mapped.size(-1),))
            out_residual = out + mapped_out 
            mapped.append(out_residual)

        stacked = torch.stack(mapped, dim=0)  
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted_sum = torch.einsum('nblh,n->blh', stacked, weights)
        return weighted_sum 


class SequenceTransformer(torch.nn.Module):
    def __init__(self, arch, d_model=512, nhead=8, num_layers=1, dropout=0.1):
        super(SequenceTransformer, self).__init__()
        self.sequence_image_encoder = SequenceImageEnoder(arch, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dropout=dropout, return_layers=[1,2])
    def forward(self, x):
        x = self.sequence_image_encoder(x)
        x = self.transformer_encoder(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 64, 1, 224, 224)
    sequence_image_encoder = SequenceTransformer(arch='resnet18', d_model=512, dropout=0.1)
    output = sequence_image_encoder(x)
    print(output.shape)
