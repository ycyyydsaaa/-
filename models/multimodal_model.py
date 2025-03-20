import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class HybridFeatureExtractor(nn.Module):
    def __init__(self, output_dim=1024):  # 添加 output_dim 参数，默认 1024
        super().__init__()
        self.densenet = densenet.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        self.densenet.classifier = nn.Identity()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Identity()
        # 添加投影层，将 2304 维投影到 output_dim
        self.proj = nn.Conv2d(2304, output_dim, kernel_size=1, bias=False)

    def forward(self, x):
        densenet_feat_map = self.densenet.features(x)  # 输出通道数 1024
        efficientnet_feat_map = self.efficientnet.extract_features(x)  # 输出通道数 1280
        if densenet_feat_map.size()[2:] != efficientnet_feat_map.size()[2:]:
            efficientnet_feat_map = F.interpolate(efficientnet_feat_map, size=densenet_feat_map.size()[2:], mode='bilinear', align_corners=False)
        fused_feat_map = torch.cat([densenet_feat_map, efficientnet_feat_map], dim=1)  # 通道数 2304
        fused_feat_map = self.proj(fused_feat_map)  # 投影到 1024
        global_feat = F.adaptive_avg_pool2d(fused_feat_map, 1).view(fused_feat_map.size(0), -1)
        return fused_feat_map, global_feat

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class TransformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TransformerAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key):
        attn_output, _ = self.attention(query, key, key)
        return self.norm(query + attn_output)

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        text_feature = torch.zeros(batch_size, 768, device=device, dtype=torch.float32)
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        logits, _, _, _, _ = self.model(x, text_feature, meta)
        return logits

class MultiModalNet(nn.Module):
    def __init__(self, disease_cols, kg_embeddings, adjacency_matrix):
        super(MultiModalNet, self).__init__()
        self.disease_cols = disease_cols
        self.register_buffer('kg_embeddings', kg_embeddings.detach().clone())
        self.register_buffer('A', adjacency_matrix.detach().clone())
        self.feature_dim = 1024  # 保持 1024
        self.spatial_size = 8
        self.num_diseases = len(disease_cols)
        self.kg_embedding_dim = kg_embeddings.size(1)

        self.feature_extractor = HybridFeatureExtractor(output_dim=self.feature_dim)  # 传入 feature_dim
        self.feature_extractor.densenet = self.feature_extractor.densenet.to(kg_embeddings.device)
        self.feature_extractor.efficientnet = self.feature_extractor.efficientnet.to(kg_embeddings.device)

        self.se_block = SEBlock(self.feature_dim)
        self.spatial_attention = SpatialAttention()
        self.img_transformer = TransformerAttention(self.feature_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.feature_dim)
        )
        self.feat_adapter = nn.Linear(self.feature_dim, self.feature_dim)
        self.cross_attention = CrossAttention(self.feature_dim)
        self.spatial_pool = nn.Conv2d(self.feature_dim, 1, kernel_size=1)

        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.img_disease_proj = nn.Linear(self.feature_dim, self.num_diseases * 128)
        self.gcn = nn.Linear(self.kg_embedding_dim, 128)

        self.fusion_dim = self.feature_dim + 128 + self.num_diseases * 128
        self.classifier = nn.Linear(self.fusion_dim, self.num_diseases)
        self.kg_projection = nn.Linear(self.kg_embedding_dim, 1)

        self.grad_cam = GradCAM(model=WrappedModel(self),
                                target_layers=[self.feature_extractor.efficientnet._conv_head])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def generate_pos_encoding(self, channels, height, width, device):
        half_channels = channels // 2
        div_term = torch.exp(
            torch.arange(0, half_channels, 2, device=device).float() * (-math.log(10000.0) / half_channels))
        pos_h = torch.arange(height, device=device).float().view(1, 1, height, 1) / height
        pos_w = torch.arange(width, device=device).float().view(1, 1, 1, width) / width
        pos_enc = torch.zeros(1, channels, height, width, device=device)
        h_term = pos_h * div_term.view(1, -1, 1, 1)
        h_term = h_term.expand(-1, -1, -1, width)
        pos_enc[:, 0:half_channels:2] = torch.sin(h_term)
        pos_enc[:, 1:half_channels:2] = torch.cos(h_term)
        w_term = pos_w * div_term.view(1, -1, 1, 1)
        w_term = w_term.expand(-1, -1, height, -1)
        pos_enc[:, half_channels::2] = torch.sin(w_term)
        pos_enc[:, half_channels + 1::2] = torch.cos(w_term)
        return pos_enc

    def forward(self, paired_img, text_feature, meta, use_text=True):
        batch_size = paired_img.size(0)
        device = paired_img.device

        fused_feat_map, global_feat = self.feature_extractor(paired_img)

        img_feat = self.se_block(fused_feat_map)
        img_feat = self.spatial_attention(img_feat)

        if img_feat.shape[2:] != (self.spatial_size, self.spatial_size):
            img_feat = F.interpolate(img_feat, size=(self.spatial_size, self.spatial_size), mode='bilinear',
                                     align_corners=False)

        pos_enc = self.generate_pos_encoding(channels=self.feature_dim, height=img_feat.size(2),
                                             width=img_feat.size(3), device=device)
        pos_enc = pos_enc.expand(batch_size, -1, -1, -1)
        img_feat = img_feat + pos_enc

        img_seq = img_feat.view(batch_size, self.feature_dim, -1).permute(2, 0, 1)
        img_seq = self.img_transformer(img_seq)

        if use_text:
            if text_feature is None:
                text_feat = torch.zeros(batch_size, 768, device=device)
            else:
                text_feat = text_feature.to(device)
            text_feat = self.text_proj(text_feat)
            text_feat = self.feat_adapter(text_feat).unsqueeze(0)
            fused_seq = self.cross_attention(img_seq, text_feat)
        else:
            fused_seq = img_seq

        fused_feat = fused_seq.permute(1, 2, 0).view(batch_size, self.feature_dim, self.spatial_size,
                                                     self.spatial_size)
        attn_map = self.spatial_pool(fused_feat)
        global_feat = (fused_feat * attn_map).sum(dim=[2, 3]) / (attn_map.sum(dim=[2, 3]) + 1e-6)

        meta_feat = self.meta_encoder(meta.to(device))

        kg_features = torch.matmul(self.A, self.kg_embeddings)
        kg_features = self.gcn(kg_features)
        kg_features = F.relu(kg_features)

        img_disease_feat = self.img_disease_proj(global_feat).view(batch_size, self.num_diseases, 128)

        kg_fused = kg_features.unsqueeze(0) + img_disease_feat
        kg_fused_flat = kg_fused.view(batch_size, -1)

        fused = torch.cat([global_feat, meta_feat, kg_fused_flat], dim=1)
        logits = self.classifier(fused)

        with torch.no_grad():
            kg_logits = self.kg_projection(self.kg_embeddings).squeeze(-1)
        kg_logits = kg_logits.unsqueeze(0).expand(batch_size, -1)

        return logits, None, kg_logits, fused_feat, attn_map

    def generate_diagnostic_path(self, logits):
        if logits.size(0) != 1:
            raise ValueError(f"预期批次大小为 1，但得到 {logits.size(0)}")
        probs = torch.sigmoid(logits)
        diagnostic_path = {}
        for i, d in enumerate(self.disease_cols):
            diagnostic_path[d] = "exists" if probs[0][i].item() > 0.5 else "does not exist"
        return diagnostic_path