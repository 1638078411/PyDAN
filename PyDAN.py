import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from collections import OrderedDict

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class RadixSoftmax1D(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class SE1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE1D, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DAN(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, 
                 rd_divisor=8, se_reduction=16, act_layer=nn.ReLU, norm_layer=None, 
                 drop_block=None, **kwargs):
        super(DAN, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix

        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding

        self.conv = nn.Conv1d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)

        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
 
        self.fc1 = nn.Conv1d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv1d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax1D(radix, groups)

        self.se = SE1D(out_channels, reduction_ratio=se_reduction)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, L = x.shape
        if self.radix > 1:
            x = x.view(B, self.radix, RC // self.radix, L)
            x_gap = x.sum(dim=1)
        else:
            x_gap = x

        x_gap = F.adaptive_avg_pool1d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1)

        if self.radix > 1:
            out = (x * x_attn.view(B, self.radix, RC // self.radix, 1)).sum(dim=1)
        else:
            out = x * x_attn

        out = self.se(out)
        
        return out



class FeaturePyramid1D(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(FeaturePyramid1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.fuse1 = nn.Conv1d(2*out_channels, out_channels, kernel_size=1)
        self.fuse2 = nn.Conv1d(2*out_channels, out_channels, kernel_size=1)

        self.output_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = F.relu(self.conv1(x), inplace=False)
        c2 = F.relu(self.conv2(c1), inplace=False)
        c3 = F.relu(self.conv3(c2), inplace=False)

        p3 = c3
        p2 = self.up1(p3)
 
        p2 = F.interpolate(p2, size=c2.size(2), mode='linear', align_corners=False)
        
        p2 = torch.cat([p2, c2], dim=1)
        p2 = F.relu(self.fuse2(p2), inplace=False)
        
        p1 = self.up2(p2)

        p1 = F.interpolate(p1, size=c1.size(2), mode='linear', align_corners=False)
        
        p1 = torch.cat([p1, c1], dim=1)
        p1 = F.relu(self.fuse1(p1), inplace=False)

        out = self.output_conv(p1)
        return out

class PyDAN(nn.Module):
    def __init__(self, input_dim, num_classes, radix=2, se_reduction=16, 
                 min_depth=3, max_depth=5, depth_factor=0.5):
        super(PyDAN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
     
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False)
        )

        depth = min_depth + int((max_depth - min_depth) * depth_factor * min(1.0, input_dim / 1000))
        print(f"Using dynamic depth: {depth} layers")

        self.mid_blocks = nn.ModuleList()
        in_channels = 32
        out_channels = 64
        
        for i in range(depth):
            if i > 0 and i % 2 == 0:
                in_channels = out_channels
                out_channels = min(256, out_channels * 2)
                
            block = DAN(
                in_channels, 
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                radix=radix,
                se_reduction=se_reduction,
                norm_layer=nn.BatchNorm1d
            )
            self.mid_blocks.append(block)
            in_channels = out_channels
        
        self.pyramid = FeaturePyramid1D(out_channels)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(out_channels, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.initial_conv(x)
        
        for block in self.mid_blocks:
            x = block(x)
        
        x = self.pyramid(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.forward_features(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

def train_PyDAN(X_train, y_train, X_test, y_test, 
                            epochs=100, batch_size=32, lr=0.001, 
                            radix=2, se_reduction=16, model_dir='models',
                            min_depth=3, max_depth=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    feature_dim = X_train.shape[1]
    depth_factor = min(1.0, feature_dim / 1000)  
    

    num_classes = len(torch.unique(y_train))
    model = PyDAN(
        input_dim=feature_dim,
        num_classes=num_classes,
        radix=radix,
        se_reduction=se_reduction,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_factor=depth_factor
    ).to(device)
    
    print("\nModel Architecture:")
    print(model)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           
        factor=0.5,           
        patience=5,          
        min_lr=1e-6,          
        threshold=0.001,     
        threshold_mode='abs' 
    )

    model_path = os.path.join(model_dir, f'best_model_radix{radix}_se{se_reduction}_classes{num_classes}.pth')
    
    if os.path.exists(model_path):
        try:
            print(f"Loading existing model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            print("Successfully loaded existing model!")
            
            model.eval()
            with torch.no_grad():
                test_inputs = X_test.to(device)
                outputs = model(test_inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_true = y_test.numpy()
                
                if num_classes == 2:
                    existing_auc = roc_auc_score(y_true, probs[:, 1])
                else:
                    existing_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
                print(f"Existing model AUC: {existing_auc:.4f}")
                
        except RuntimeError as e:
            os.remove(model_path)
    else:
        print("No existing model found. Starting training from scratch...")
    

    best_val_auc = 0.0
    train_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            test_inputs = X_test.to(device)
            outputs = model(test_inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true = y_test.numpy()
            
            if num_classes == 2:
                auc_score = roc_auc_score(y_true, probs[:, 1])
            else:
                auc_score = roc_auc_score(
                    y_true, probs, 
                    multi_class='ovr', 
                    average='macro'
                )
        
        if epoch >= 5:
            scheduler.step(auc_score)
        
        if auc_score > best_val_auc:
            best_val_auc = auc_score
            model_path = os.path.join(model_dir, f'best_model_radix{radix}_se{se_reduction}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model at epoch {epoch+1} with AUC: {auc_score:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_aucs.append(auc_score)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val AUC: {auc_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    model.eval()
    with torch.no_grad():
        test_inputs = X_test.to(device)
        outputs = model(test_inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_test.numpy()
    
    return model

model = train_PyDAN(
    X_train, Y_train, X_test, Y_test,
    epochs=30,
    batch_size=64,
    lr=0.001,
    radix=2,
    se_reduction=16,
    min_depth=3,
    max_depth=6
)
