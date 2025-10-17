import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPrototype(nn.Module):
    def __init__(self, feat_dim=512, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("proto_0", torch.zeros(feat_dim))
        self.register_buffer("proto_1", torch.zeros(feat_dim))
        self.proto_keys = {"0": "proto_0", "1": "proto_1"}

        self._init_unit_sphere()

    @torch.no_grad()
    def _init_unit_sphere(self):
        dim = self.proto_0.shape[0]
        vec = torch.randn(dim)
        vec = F.normalize(vec, dim=0)
        self.proto_0.copy_(-vec)
        self.proto_1.copy_(vec)

    @torch.no_grad()
    def update(self, features, labels):
        for cls_str in ["0", "1"]:
            cls = int(cls_str)
            mask = (labels == cls)
            if mask.sum() > 0:
                cls_feat = features[mask].mean(dim=0).to(getattr(self, self.proto_keys[cls_str]).device)
                proto = getattr(self, self.proto_keys[cls_str])
                proto.mul_(self.momentum).add_(cls_feat * (1 - self.momentum))
                proto.copy_(F.normalize(proto, dim=0))

    def getprototypes(self):
        return {
            "0": getattr(self, self.proto_keys["0"]),
            "1": getattr(self, self.proto_keys["1"])
        }

    @torch.no_grad()
    def clone_prototypes(self):
        return {
            "0": self.proto_0.clone().detach(),
            "1": self.proto_1.clone().detach()
        }

    def forward(self, features):
        return features
