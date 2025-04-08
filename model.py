import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained DenseNet with regularization
class DenseNetTokenizer(nn.Module):
    def __init__(self, model_name="densenet121", out_dim=256, dropout_rate=0.3):
        super().__init__()
        # Load pretrained DenseNet with frozen parameters
        self.model = models.densenet121(pretrained=True).features

        # Freeze early layers to prevent overfitting
        for param in list(self.model.parameters())[:-20]:  # Keep last few layers trainable
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),  # Add LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        x = self.model(x)  # Extract features
        x = torch.flatten(x, start_dim=2).mean(dim=2)  # Global average pooling
        x = self.dropout(x)  # Apply dropout before projection
        x = self.projection(x)
        return x

# Improved CCT with stronger regularization
class CCT(nn.Module):
    def __init__(self, embedding_dim=256, num_layers=4, num_heads=4, mlp_ratio=2.0, 
                 num_classes=8, dropout=0.4, attn_dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        return x

# Regularized Hybrid CCT model
class HybridCCT(nn.Module):
    def __init__(self, num_classes=8, embedding_dim=256):
        super().__init__()
        self.tokenizer = DenseNetTokenizer(out_dim=embedding_dim, dropout_rate=0.3)

        self.transformer = CCT(
            embedding_dim=embedding_dim,
            num_layers=4,
            num_heads=4,
            mlp_ratio=2.0,
            num_classes=num_classes,
            dropout=0.4
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformer(x)
        return x
