import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Proto import DynamicPrototype

class ComparisonFramework(nn.Module):
    def __init__(self, encoder_q, encoder_k, K=4096, m=0.999, T=0.07):
        super(ComparisonFramework, self).__init__()       
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        if isinstance(encoder_q.final_output_dim, tuple):
            self.register_buffer("queue", torch.randn(K, *encoder_q.final_output_dim))
        else:
            self.register_buffer("queue", torch.randn(K, encoder_q.final_output_dim))
            
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", torch.full((K,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.prototype_module = DynamicPrototype(encoder_q.final_output_dim)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.K:
            remain = self.K - ptr
            self.queue[ptr:] = keys[:remain]
            self.queue_labels[ptr:] = labels[:remain]
            self.queue[:batch_size-remain] = keys[remain:]
            self.queue_labels[:batch_size-remain] = labels[remain:]
            ptr = batch_size - remain
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_labels[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, ct_im_q, pet_im_q, labels):
        
        q,q_logits = self.encoder_q(ct_im_q, pet_im_q) 

        with torch.no_grad():
            self._momentum_update_key_encoder()
            ct_im_k = ct_im_q
            pet_im_k = pet_im_q
            k,_ = self.encoder_k(ct_im_k, pet_im_k) 

        all_features = torch.cat([q, k, self.queue.clone().detach()], dim=0)  
        
        all_labels = torch.cat([
            labels,  
            labels,  
            self.queue_labels  
        ], dim=0)

        self._dequeue_and_enqueue(k, labels)
        valid_mask = all_labels != -1
        all_features = all_features[valid_mask]
        all_labels = all_labels[valid_mask]

        return all_features, all_labels, q ,q_logits,labels 