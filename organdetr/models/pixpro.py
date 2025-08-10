import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP3d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP3d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm3d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5, step=0):
    """ q, k: N * C * D * H * W
        coord_q, coord_k: N * 6 (x_upper_left, y_upper_left, z_upper_left, x_lower_right, y_lower_right, z_lower_right)
    """
    N, C, D, H, W = q.shape
    # [bs, feat_dim, D*H*W]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate center_coord, width, height, depth
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, 1, -1).repeat(1, D, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1, 1).repeat(1, D, 1, W)
    z_array = torch.arange(0., float(D), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1, 1).repeat(1, 1, H, W)
    
    q_bin_width = ((coord_q[:, 3] - coord_q[:, 0]) / W).view(-1, 1, 1, 1)
    q_bin_height = ((coord_q[:, 4] - coord_q[:, 1]) / H).view(-1, 1, 1, 1)
    q_bin_depth = ((coord_q[:, 5] - coord_q[:, 2]) / D).view(-1, 1, 1, 1)
    k_bin_width = ((coord_k[:, 3] - coord_k[:, 0]) / W).view(-1, 1, 1, 1)
    k_bin_height = ((coord_k[:, 4] - coord_k[:, 1]) / H).view(-1, 1, 1, 1)
    k_bin_depth = ((coord_k[:, 5] - coord_k[:, 2]) / D).view(-1, 1, 1, 1)
    
    q_start_x = coord_q[:, 0].view(-1, 1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1, 1)
    q_start_z = coord_q[:, 2].view(-1, 1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1, 1)
    k_start_z = coord_k[:, 2].view(-1, 1, 1, 1)

    q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2 + q_bin_depth ** 2)
    k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2 + k_bin_depth ** 2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_q_z = (z_array + 0.5) * q_bin_depth + q_start_z
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y
    center_k_z = (z_array + 0.5) * k_bin_depth + k_start_z

    dist_center = torch.sqrt(
        (center_q_x.view(-1, D * H * W, 1) - center_k_x.view(-1, 1, D * H * W)) ** 2
        + (center_q_y.view(-1, D * H * W, 1) - center_k_y.view(-1, 1, D * H * W)) ** 2
        + (center_q_z.view(-1, D * H * W, 1) - center_k_z.view(-1, 1, D * H * W)) ** 2
    ) / max_bin_diag.view(N, 1, 1)
    pos_mask = (dist_center < pos_ratio).float().detach()

    '''
    # Count elements where dist_center < pos_ratio (where pos_mask == 1)
    count_pos = pos_mask.sum().item()

    # Count all elements in the tensor
    total_elements = pos_mask.numel()

    print("****************************Number of elements satisfying dist_center < pos_ratio:", int(count_pos))
    print("****************************Total number of elements in the tensor:", total_elements)

    '''

    logit = torch.bmm(q.transpose(1, 2), k)
    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()

# def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
def Proj_Head(in_dim=384, inner_dim=4096, out_dim=256):
    return MLP3d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP3d(in_dim, inner_dim, out_dim)


class PixPro(nn.Module):
    def __init__(self, encoder, encoder_k, config, update_encoder=True):
        super().__init__()

        # parse arguments
        self.pixpro_p               = config['pixpro']['pixpro_p']
        self.pixpro_momentum        = config['pixpro']['pixpro_momentum']
        self.pixpro_pos_ratio       = config['pixpro']['pixpro_pos_ratio']
        self.pixpro_clamp_value     = config['pixpro']['pixpro_clamp_value']
        self.pixpro_transform_layer = config['pixpro']['pixpro_transform_layer']
        self.pixpro_ins_loss_weight = config['pixpro']['pixpro_ins_loss_weight']
        self.same_mlp = config['pixpro']['same_mlp']
        self.num_feat = config['pixpro']['num_feat']
        self.propagate = 'propagate' in config['pixpro'] and config['pixpro']['propagate']
        print(f"***********************self.propagate={self.propagate}")

        # create the encoder
        self.encoder = encoder
        self.encoder_k = encoder_k

        # can be already updated by outer teacher student model, no need to update here.
        self.update_encoder = update_encoder

        if self.same_mlp: 
            self.projector = Proj_Head(in_dim=config['backbone']['fpn_channels'])
            self.predictor = Proj_Head(256, 256, 256)
            self.projector_k = Proj_Head(in_dim=config['backbone']['fpn_channels'])
        else:
            self.projectors = nn.ModuleList([Proj_Head(in_dim=config['backbone']['fpn_channels']) for _ in range(self.num_feat)])
            self.predictors = nn.ModuleList([Proj_Head(256, 256, 256) for _ in range(self.num_feat)])
            self.projectors_k = nn.ModuleList([Proj_Head(in_dim=config['backbone']['fpn_channels']) for _ in range(self.num_feat)])

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.same_mlp:
            for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        else:
            for i in range (self.num_feat):
                for param_q, param_k in zip(self.projectors[i].parameters(), self.projectors_k[i].parameters()):
                    param_k.data.copy_(param_q.data)
                    param_k.requires_grad = False

        '''
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        if self.same_mlp:
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        else:
            for i in range (self.num_feat):
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projectors[i])
                nn.SyncBatchNorm.convert_sync_batchnorm(self.predictors[i])
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projectors_k[i])
        '''

        self.K = int(config['pixpro']['num_instances'] * 1. / config['pixpro']['batch_size'] * config['pixpro']['epochs'])
        self.k = 0 

        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1(in_planes=256, out_planes=256)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP3d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError

        if self.pixpro_ins_loss_weight > 0.:
            self.projector_instance = Proj_Head(in_dim=config['backbone']['fpn_channels'])
            self.projector_instance_k = Proj_Head(in_dim=config['backbone']['fpn_channels'])
            self.predictor = Pred_Head(in_dim=config['backbone']['fpn_channels'])

            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            '''
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
            '''

            self.avgpool = nn.AvgPool3d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.pixpro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        # update only if it is not updated outside pixpro
        if self.update_encoder:
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        if self.same_mlp:
            for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        else:
            for i in range(self.num_feat):
                for param_q, param_k in zip(self.projectors[i].parameters(), self.projectors_k[i].parameters()):
                    param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        if self.pixpro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def featprop(self, feat):
        N, C, D, H, W = feat.shape

        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)

        # [N, C, D * H * W]
        feat = feat.view(N, C, -1)

        # [N, D * H * W, D * H * W]
        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.pixpro_p

        # [N, C, D * H * W]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, D, H, W)

    def regression_loss(self, x, y):
        return -2. * torch.einsum('nc, nc->n', [x, y]).mean()

    def forward(self, im_1, im_2, coord1, coord2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        feat_1 = self.encoder(im_1)  # queries: NxC
        feat_2 = self.encoder(im_2)
        # todo: use multi-level feature map
        preds_1 = []
        preds_2 = []
        for i in range(self.num_feat):
            # print(f"*******************i={i}, P={'P' + str(4 - i)}, P.shape={feat_1['P' + str(4 - i)].shape}", flush=True)
            if self.same_mlp:
                pred_1 = self.predictor(self.projector(feat_1['P' + str(4 - i)]))
                pred_2 = self.predictor(self.projector(feat_2['P' + str(4 - i)]))
            else:
                pred_1 = self.predictors[i](self.projectors[i](feat_1['P' + str(4 - i)]))
                pred_2 = self.predictors[i](self.projectors[i](feat_2['P' + str(4 - i)]))

            if self.propagate:
                pred_1 = self.featprop(pred_1)
                pred_2 = self.featprop(pred_2)

            pred_1 = F.normalize(pred_1, dim=1)
            pred_2 = F.normalize(pred_2, dim=1)
            preds_1.append(pred_1)
            preds_2.append(pred_2)


        if self.pixpro_ins_loss_weight > 0.:
            proj_instance_1 = self.projector_instance(feat_1)
            pred_instacne_1 = self.predictor(proj_instance_1)
            pred_instance_1 = F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0), -1), dim=1)

            proj_instance_2 = self.projector_instance(feat_2)
            pred_instance_2 = self.predictor(proj_instance_2)
            pred_instance_2 = F.normalize(self.avgpool(pred_instance_2).view(pred_instance_2.size(0), -1), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_1)  # keys: NxC
            feat_2_ng = self.encoder_k(im_2)
            projs_ng_1 = []
            projs_ng_2 = []
            for i in range(self.num_feat):
                if self.same_mlp:
                    proj_1 = self.projector_k(feat_1_ng['P' + str(4 - i)])
                    proj_2 = self.projector_k(feat_2_ng['P' + str(4 - i)])
                else:
                    proj_1 = self.projectors_k[i](feat_1_ng['P' + str(4 - i)])
                    proj_2 = self.projectors_k[i](feat_2_ng['P' + str(4 - i)])

                proj_1 = F.normalize(proj_1, dim=1)
                proj_2 = F.normalize(proj_2, dim=1)
                projs_ng_1.append(proj_1)
                projs_ng_2.append(proj_2)

            if self.pixpro_ins_loss_weight > 0.:
                proj_instance_1_ng = self.projector_instance_k(feat_1_ng)
                proj_instance_1_ng = F.normalize(self.avgpool(proj_instance_1_ng).view(proj_instance_1_ng.size(0), -1),
                                                 dim=1)

                proj_instance_2_ng = self.projector_instance_k(feat_2_ng)
                proj_instance_2_ng = F.normalize(self.avgpool(proj_instance_2_ng).view(proj_instance_2_ng.size(0), -1),
                                                 dim=1)

        # compute loss
        losses = []
        for i in range(self.num_feat):
            loss_i = regression_loss(preds_1[i], projs_ng_2[i], coord1, coord2, self.pixpro_pos_ratio, self.k) \
                + regression_loss(preds_2[i], projs_ng_1[i], coord2, coord1, self.pixpro_pos_ratio, self.k)

            losses.append(loss_i)
            if i == 0:
                loss = loss_i
            else:
                loss = loss + loss_i

        loss = loss / self.num_feat

        if self.pixpro_ins_loss_weight > 0.:
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) + \
                         self.regression_loss(pred_instance_2, proj_instance_1_ng)
            loss = loss + self.pixpro_ins_loss_weight * loss_instance

        return loss, losses 
