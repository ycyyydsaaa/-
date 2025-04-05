import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    import logging
    logging.error("torch_geometric 未安装，请运行 'pip install torch-geometric'")
    raise
import gc
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_tensor_info(tensor, name):
    if tensor is not None:
        memory_mb = tensor.element_size() * tensor.nelement() / 1024 ** 2
        ref_count = sys.getrefcount(tensor)
        print(f"Tensor {name}: Memory = {memory_mb:.2f} MB, Ref Count = {ref_count}")
    else:
        print(f"Tensor {name}: None")

class HybridFeatureExtractor(nn.Module):
    def __init__(self, output_dim=2048):
        super().__init__()
        # 初始化 EfficientNet-B5，所有层默认可训练
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        self.efficientnet._fc = nn.Identity()  # 替换全连接层为 Identity

        # 投影层，输入通道数为左右眼特征拼接后的通道数（2048 * 2 = 4096）
        self.proj = nn.Conv2d(4096, output_dim, kernel_size=1, bias=False)

    def forward(self, x):
        # 分割双目图像
        b, c, h, w = x.shape
        left_img = x[:, :, :, :w//2]  # 左眼图像 [batch_size, 3, 256, 256]
        right_img = x[:, :, :, w//2:]  # 右眼图像 [batch_size, 3, 256, 256]

        # 分别提取左右眼特征
        left_feat = self.efficientnet.extract_features(left_img)  # [batch_size, 2048, H, W]
        right_feat = self.efficientnet.extract_features(right_img)  # [batch_size, 2048, H, W]

        # 确保左右眼特征图空间尺寸一致
        if left_feat.size()[2:] != right_feat.size()[2:]:
            target_size = (min(left_feat.size(2), right_feat.size(2)),
                           min(left_feat.size(3), right_feat.size(3)))
            left_feat = F.interpolate(left_feat, size=target_size, mode='bilinear', align_corners=False)
            right_feat = F.interpolate(right_feat, size=target_size, mode='bilinear', align_corners=False)

        # 在通道维度上拼接左右眼特征
        fused_feat_map = torch.cat([left_feat, right_feat], dim=1)  # [batch_size, 4096, H, W]

        # 通过1x1卷积投影到指定维度
        fused_feat_map = self.proj(fused_feat_map)  # [batch_size, output_dim, H, W]

        # 全局池化生成全局特征
        global_feat = F.adaptive_avg_pool2d(fused_feat_map, 1).view(fused_feat_map.size(0), -1)  # [batch_size, output_dim]

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
    def __init__(self, dim, num_heads=4, num_layers=2):
        super(TransformerAttention, self).__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, num_heads) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_layers)
        ])
        self.norm2s = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

    def forward(self, x, batch_idx=0):
        for attn, norm, ffn, norm2 in zip(self.layers, self.norms, self.ffns, self.norm2s):
            attn_output, _ = attn(x, x, x)
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                logger.warning(f"Batch {batch_idx}: TransformerAttention contains NaN or Inf")
                attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
            x = norm(x + attn_output)
            ffn_output = ffn(x)
            if torch.isnan(ffn_output).any() or torch.isinf(ffn_output).any():
                logger.warning(f"Batch {batch_idx}: TransformerAttention ffn_output contains NaN or Inf")
                ffn_output = torch.nan_to_num(ffn_output, nan=0.0, posinf=1.0, neginf=-1.0)
            x = norm2(x + ffn_output)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, query, key, batch_idx=0):
        attn_output, _ = self.attention(query, key, key)
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            logger.warning(f"Batch {batch_idx}: CrossAttention contains NaN or Inf")
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.norm(query + self.scale * attn_output)

class MetaFusion(nn.Module):
    def __init__(self, feat_dim, meta_dim):
        super().__init__()
        self.meta_proj = nn.Linear(meta_dim, feat_dim)
        self.attn = nn.Linear(feat_dim * 2, 1)

    def forward(self, img_feat, meta_feat):
        meta_feat = self.meta_proj(meta_feat)
        combined = torch.cat([img_feat, meta_feat], dim=-1)
        attn_weights = torch.sigmoid(self.attn(combined))
        return img_feat * attn_weights + meta_feat * (1 - attn_weights)

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        logits, _, _, _, _ = self.model(x, None, meta, use_text=False)
        return logits

class MultiModalNet(nn.Module):
    def __init__(self, disease_cols, kg_embeddings, adjacency_matrix):
        super(MultiModalNet, self).__init__()
        self.disease_cols = disease_cols
        self.register_buffer('kg_embeddings', kg_embeddings.detach().clone())
        self.register_buffer('A', adjacency_matrix.detach().clone())
        self.feature_dim = 2048
        self.spatial_height = 8  # 修改为 [8, 8]
        self.spatial_width = 8   # 修改为 [8, 8]
        self.num_diseases = len(disease_cols)
        self.kg_embedding_dim = kg_embeddings.size(1)  # 256

        self.feature_extractor = HybridFeatureExtractor(output_dim=self.feature_dim)
        self.feature_extractor.efficientnet = self.feature_extractor.efficientnet.to(kg_embeddings.device)

        self.se_block = SEBlock(self.feature_dim)
        self.spatial_attention = SpatialAttention()
        self.img_transformer = TransformerAttention(self.feature_dim, num_heads=4, num_layers=2)

        self.text_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.feature_dim)
        )
        self.feat_adapter = nn.Linear(self.feature_dim, self.feature_dim)
        self.cross_attention = CrossAttention(self.feature_dim, num_heads=4, dropout=0.1)
        self.spatial_pool = nn.Conv2d(self.feature_dim, 1, kernel_size=1)

        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.meta_fusion = MetaFusion(self.feature_dim, 128)

        self.img_disease_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, self.num_diseases * 256)
        )
        self.gcn1 = GCNConv(self.kg_embedding_dim, 512)
        self.gcn2 = GCNConv(512, 256)
        self.img_kg_attention = nn.MultiheadAttention(256, num_heads=4)

        self.fusion_dim = self.feature_dim + (self.num_diseases * self.kg_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fusion_dim // 2, self.num_diseases)
        )
        self.kg_projection = nn.Linear(self.kg_embedding_dim, 1)

        self.grad_cam = GradCAM(model=WrappedModel(self),
                                target_layers=[self.feature_extractor.efficientnet._bn1])
        self._initialize_weights()

    def initialize_kg_logits(self):
        if hasattr(self, 'kg_logits') and self.kg_logits is not None:
            logger.info("kg_logits 已存在，跳过初始化")
            device = next(self.kg_projection.parameters()).device
            self.kg_logits = self.kg_logits.to(device)
            return

        device = next(self.kg_projection.parameters()).device
        self.kg_embeddings = self.kg_embeddings.to(device)
        with torch.no_grad():
            kg_logits = self.kg_projection(self.kg_embeddings).squeeze(-1)
            self.register_buffer('kg_logits', kg_logits)

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
        pos_enc[:, 0:half_channels:2] = torch.sin(h_term.expand(-1, -1, -1, width))
        pos_enc[:, 1:half_channels:2] = torch.cos(h_term.expand(-1, -1, -1, width))
        w_term = pos_w * div_term.view(1, -1, 1, 1)
        pos_enc[:, half_channels::2] = torch.sin(w_term.expand(-1, -1, height, -1))
        pos_enc[:, half_channels + 1::2] = torch.cos(w_term.expand(-1, -1, height, -1))
        return pos_enc

    def forward(self, paired_img, text_feature, meta, use_text=True, batch_idx=0):
        batch_size = paired_img.size(0)
        device = paired_img.device

        fused_feat_map, global_feat = self.feature_extractor(paired_img)
        img_feat = self.se_block(fused_feat_map)
        img_feat = self.spatial_attention(img_feat)

        del fused_feat_map
        torch.cuda.empty_cache()
        gc.collect()

        # 调整为 [8, 8]
        if img_feat.shape[2:] != (self.spatial_height, self.spatial_width):
            img_feat = F.interpolate(img_feat, size=(self.spatial_height, self.spatial_width),
                                     mode='bilinear', align_corners=False)

        # 更新位置编码为 [8, 8]
        pos_enc = self.generate_pos_encoding(channels=self.feature_dim, height=self.spatial_height,
                                             width=self.spatial_width, device=device)
        pos_enc = pos_enc.expand(batch_size, -1, -1, -1)
        img_feat = img_feat + pos_enc

        # 序列长度变为 8*8=64
        img_seq = img_feat.view(batch_size, self.feature_dim, -1).permute(2, 0, 1)  # [64, batch_size, 2048]
        img_seq = self.img_transformer(img_seq, batch_idx=batch_idx)

        del img_feat, pos_enc
        torch.cuda.empty_cache()
        gc.collect()

        if use_text and text_feature is not None:
            text_feat = self.text_proj(text_feature.to(device))
            text_feat = self.feat_adapter(text_feat)
            text_feat = text_feat.unsqueeze(0).expand(img_seq.size(0), -1, -1)
            fused_seq = self.cross_attention(img_seq, text_feat, batch_idx=batch_idx)
        else:
            fused_seq = img_seq

        # 调整回 [batch_size, feature_dim, 8, 8]
        fused_feat = fused_seq.permute(1, 2, 0).view(batch_size, self.feature_dim, self.spatial_height, self.spatial_width)
        attn_map = self.spatial_pool(fused_feat)
        attn_sum = attn_map.sum(dim=[2, 3])
        global_feat_weighted = (fused_feat * attn_map).sum(dim=[2, 3]) / (attn_sum + 1e-4)
        if torch.isnan(global_feat_weighted).any() or torch.isinf(global_feat_weighted).any():
            logger.warning(f"Batch {batch_idx}: global_feat_weighted contains NaN or Inf")
            global_feat_weighted = torch.nan_to_num(global_feat_weighted, nan=0.0, posinf=1.0, neginf=-1.0)

        del fused_seq, fused_feat
        torch.cuda.empty_cache()
        gc.collect()

        meta_feat = self.meta_encoder(meta.to(device))
        fused_img_meta = self.meta_fusion(global_feat_weighted, meta_feat)

        kg_features = self.gcn1(self.kg_embeddings, self.A.indices())
        kg_features = F.relu(kg_features)
        kg_features = self.gcn2(kg_features, self.A.indices())
        kg_features = F.relu(kg_features)

        img_disease_feat = self.img_disease_proj(fused_img_meta).view(batch_size, self.num_diseases, 256)
        img_disease_seq = img_disease_feat.permute(1, 0, 2)

        kg_seq = kg_features.unsqueeze(1).expand(-1, batch_size, -1).permute(0, 1, 2)
        kg_fused, _ = self.img_kg_attention(img_disease_seq, kg_seq, kg_seq)
        kg_fused = kg_fused.permute(1, 0, 2)
        kg_fused_flat = kg_fused.contiguous().view(batch_size, -1)

        fused = torch.cat([fused_img_meta, kg_fused_flat], dim=1)
        fused = F.normalize(fused, p=2, dim=1)
        logits = self.classifier(fused)

        kg_logits = self.kg_logits.unsqueeze(0).expand(batch_size, -1)

        del global_feat, meta_feat, kg_features, img_disease_feat, kg_fused, kg_fused_flat, fused, attn_map
        torch.cuda.empty_cache()
        gc.collect()

        return logits, global_feat_weighted, kg_logits, None, None

    def generate_diagnostic_path(self, logits):
        if logits.size(0) != 1:
            raise ValueError(f"预期批次大小为 1，但得到 {logits.size(0)}")
        probs = torch.sigmoid(logits)
        diagnostic_path = {}
        for i, d in enumerate(self.disease_cols):
            diagnostic_path[d] = "exists" if probs[0][i].item() > 0.5 else "does not exist"
        return diagnostic_path

    def clear_resources(self):
        print("Before clearing resources:")
        print_tensor_info(self.kg_embeddings, "kg_embeddings")
        print_tensor_info(self.A, "A")
        print_tensor_info(self.kg_logits, "kg_logits")

        self.kg_embeddings = None
        self.A = None
        self.kg_logits = None
        self.grad_cam = None
        torch.cuda.empty_cache()
        gc.collect()

        print("After clearing resources:")
        print_tensor_info(self.kg_embeddings, "kg_embeddings")
        print_tensor_info(self.A, "A")
        print_tensor_info(self.kg_logits, "kg_logits")
        logger.info("MultiModalNet 资源已清空")