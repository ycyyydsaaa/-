import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import BertTokenizer, BertModel
from models.fpn import FPN

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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class MultiModalNet(nn.Module):
    def __init__(self, disease_cols, kg_embeddings):
        super(MultiModalNet, self).__init__()
        self.disease_cols = disease_cols
        self.kg_embeddings = kg_embeddings.detach().requires_grad_(False)
        self.text_dim = 768  # 定义文本特征的维度

        # 图像编码器
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b3')
        self.fpn = FPN(in_channels_list=[40, 48, 1536], out_channels=256)
        self.se_block = SEBlock(256)
        self.spatial_attention = SpatialAttention()

        # 元数据编码器
        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # 知识图谱投影
        kg_embedding_dim = self.kg_embeddings.size(1)
        self.kg_projection = nn.Linear(kg_embedding_dim, 1)

        # 可学习权重
        self.img_weight = nn.Parameter(torch.ones(1) * 2.0)
        self.text_weight = nn.Parameter(torch.ones(1) * 1.0)  # 新增：文本特征的可学习权重

        # 融合维度
        self.fusion_dim = 256 + self.text_dim + 32  # 使用 self.text_dim 更新融合维度
        self.classifiers = nn.ModuleList([nn.Linear(self.fusion_dim, 1) for _ in range(len(disease_cols))])
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)

        self.hook_handle = None
        self.gradients = None

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

    def activations_hook(self, grad):
        self.gradients = grad

    def extract_multi_features(self, x):
        endpoints = {}
        x = self.image_encoder._swish(self.image_encoder._bn0(self.image_encoder._conv_stem(x)))
        endpoints['C3'] = x
        prev_x = x
        for idx, block in enumerate(self.image_encoder._blocks):
            x = block(x)
            if idx == 7:
                endpoints['C4'] = x
            prev_x = x
        x = self.image_encoder._swish(self.image_encoder._bn1(self.image_encoder._conv_head(prev_x)))
        endpoints['C5'] = x
        return [endpoints['C3'], endpoints['C4'], endpoints['C5']]

    def forward(self, paired_img, text_feature, meta, seg_target=None):
        # 提取图像特征
        features = self.extract_multi_features(paired_img)
        fpn_feats = self.fpn(features)
        img_feat = fpn_feats[-1]
        img_feat = self.se_block(img_feat)
        img_feat = self.spatial_attention(img_feat)
        feature_maps = img_feat

        # 注册梯度钩子
        if img_feat.requires_grad:
            self.hook_handle = img_feat.register_hook(self.activations_hook)

        img_feat = self.img_weight * img_feat.mean([2, 3])

        # 处理文本特征并应用权重
        if text_feature is None:
            text_feat = torch.zeros(paired_img.size(0), self.text_dim).to(paired_img.device)
        else:
            text_feat = text_feature * self.text_weight  # 应用可学习权重

        # 元数据特征
        meta_feat = self.meta_encoder(meta)

        # 特征融合
        fused = torch.cat([img_feat, text_feat, meta_feat], dim=1)
        logits = torch.cat([classifier(fused) for classifier in self.classifiers], dim=1)

        # 知识图谱投影
        with torch.no_grad():
            kg_logits = self.kg_projection(self.kg_embeddings).squeeze(-1)
        kg_logits = kg_logits.unsqueeze(0).expand(logits.size(0), -1)

        # 分割输出（可选）
        seg_output = self.segmentation_head(fpn_feats[-1]) if seg_target is not None else None

        return logits, seg_output, kg_logits, feature_maps, self.gradients

    def generate_diagnostic_path(self, logits):
        if logits.size(0) != 1:
            raise ValueError(f"预期批次大小为 1，但得到 {logits.size(0)}")
        probs = torch.sigmoid(logits)
        diagnostic_path = {}
        for i, d in enumerate(self.disease_cols):
            diagnostic_path[d] = "exists" if probs[0][i].item() > 0.5 else "does not exist"
        return diagnostic_path