import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import BertTokenizer, BertModel
from models.fpn import FPN

class MultiModalNet(nn.Module):
    def __init__(self, disease_cols, kg_embeddings):
        super(MultiModalNet, self).__init__()
        self.disease_cols = disease_cols
        self.kg_embeddings = kg_embeddings

        # 图像编码器：使用 EfficientNet-b3
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b3')
        print("Loaded pretrained weights for efficientnet-b3")
        # 验证 conv_stem 的输出通道数
        print(f"Conv stem output channels: {self.image_encoder._conv_stem.out_channels}")

        # FPN：根据实际输出的多尺度特征调整输入通道数 [40, 48, 1536]
        self.fpn = FPN(in_channels_list=[40, 48, 1536], out_channels=256)

        # 文本编码器：BERT
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # meta 编码器：将 2 维元信息编码为 32 维
        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # 融合后的特征维度：图像 256 + 文本 768 + meta 32 = 1056
        self.fusion_dim = 1056
        self.classifiers = nn.ModuleList([nn.Linear(self.fusion_dim, 1) for _ in range(len(disease_cols))])
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)

    def extract_multi_features(self, x):
        """从 EfficientNet-b3 中提取多尺度特征"""
        endpoints = {}
        # 前向传播提取特征
        x = self.image_encoder._swish(self.image_encoder._bn0(self.image_encoder._conv_stem(x)))  # [B, 40, 128, 256]
        endpoints['C3'] = x  # 第一个特征点，通道数 40
        prev_x = x
        for idx, block in enumerate(self.image_encoder._blocks):
            x = block(x)
            if idx == 7:  # C4: 中层特征，通道数 48
                endpoints['C4'] = x  # [B, 48, 32, 64]
            prev_x = x
        # 最后一层（conv_head）
        x = self.image_encoder._swish(self.image_encoder._bn1(self.image_encoder._conv_head(prev_x)))  # [B, 1536, 8, 16]
        endpoints['C5'] = x
        return [endpoints['C3'], endpoints['C4'], endpoints['C5']]

    def forward(self, paired_img, text, meta, seg_target=None):
        # 提取多尺度图像特征
        features = self.extract_multi_features(paired_img)
        print("Multi-scale features shapes:", [f.shape for f in features])  # 调试输出
        fpn_feats = self.fpn(features)  # FPN 融合多尺度特征
        img_feat = fpn_feats[-1].mean([2, 3])  # 空间均值池化得到 [batch, 256]

        # 文本编码（BERT）
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        if inputs['input_ids'].size(1) == 0:
            text_feat = torch.zeros((paired_img.size(0), 768), device=paired_img.device)
        else:
            inputs = {k: v.to(paired_img.device) for k, v in inputs.items()}
            text_feat = self.text_encoder(**inputs).last_hidden_state.mean(1)  # [batch, 768]

        # meta 编码
        meta_feat = self.meta_encoder(meta)  # [batch, 32]

        # 融合所有模态特征
        fused = torch.cat([img_feat, text_feat, meta_feat], dim=1)  # [batch, 1056]
        logits = torch.cat([classifier(fused) for classifier in self.classifiers], dim=1)  # [batch, num_diseases]

        seg_output = self.segmentation_head(fpn_feats[-1]) if seg_target is not None else None
        return logits, seg_output, None

    def generate_diagnostic_path(self, logits):
        if logits.size(0) != 1:
            raise ValueError(f"预期批次大小为 1，但得到 {logits.size(0)}")
        probs = torch.sigmoid(logits)
        diagnostic_path = {}
        for i, d in enumerate(self.disease_cols):
            diagnostic_path[d] = "exists" if probs[0][i].item() > 0.5 else "does not exist"
        return diagnostic_path