import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        self.feature_extract1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.features_extract2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )



    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # pass
        points = points.permute(0, 2, 1)
        features = self.feature_extract1(points)
        features = self.features_extract2(features)
        features = torch.max(features, 2, keepdim=True)[0]
        features = features.view(-1, 1024)
        features = self.classifier(features)
        return features




# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, args, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.feature_extract1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.features_extract2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.maxpooling = torch.nn.MaxPool1d(args.num_points)
        self.combo_seg_extract = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.seg_layer = nn.Sequential(
            nn.Conv1d(128, num_seg_classes, 1),
        )
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # pass
        points = points.permute(0, 2, 1)
        encoded_features = self.feature_extract1(points)
        features = self.features_extract2(encoded_features)
        pooled_features = self.maxpooling(features)
        # pooled_features = torch.max(features, 2, keepdim=True)[0]
        pooled_features = pooled_features.expand(points.shape[0], 1024, points.shape[2])
        seg_input_feats = torch.cat((encoded_features, pooled_features), dim=1)
        out = self.combo_seg_extract(seg_input_feats)
        out = self.seg_layer(out).permute(0, 2, 1)
        return out



