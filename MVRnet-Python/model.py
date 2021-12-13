import torch.nn as nn
import torch
from torchvision.models import resnet50


class PRNet(nn.Module):
    def __init__(self, num_classes):
        super(PRNet, self).__init__()
        self.backbone = list(resnet50(pretrained=True).children())[:-3]  # Get the model without the last three layers.
        self.backbone[6] = list(self.backbone[6].children())[0]  # Set the conv4 to conv4_1.
        self.backbone = nn.Sequential(*self.backbone)

        self.branch_bb0 = self._get_shared_branch_backbone()
        self.branch_bb1 = self._get_shared_branch_backbone()
        self.branch_bb2 = self._get_shared_branch_backbone()

        self.fcs = nn.ModuleList()
        [self.fcs.append(nn.Linear(256, num_classes)) for _ in range(16)]

        self.conv2048s = nn.ModuleList()
        [self.conv2048s.append(nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                             nn.BatchNorm2d(256))) for _ in range(12)]

        self.conv512s = nn.ModuleList()
        [self.conv512s.append(nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                             nn.BatchNorm2d(256))) for _ in range(4)]

        self.gmp_horizontal_B = nn.AdaptiveMaxPool2d((2, 1))
        self.gmp_vertical_C = nn.AdaptiveMaxPool2d((1, 2))
        self.gmp_horizontal_D = nn.AdaptiveMaxPool2d((3, 1))
        self.gmp_vertical_E = nn.AdaptiveMaxPool2d((1, 3))

        self.gmp_full1 = nn.AdaptiveMaxPool2d(1)
        self.gmp_full2 = nn.AdaptiveMaxPool2d(1)
        self.gmp_ch0 = nn.AdaptiveMaxPool3d((4, 1, 1))
        self.gmp_ch1 = nn.AdaptiveMaxPool3d((4, 1, 1))

    @staticmethod
    def _get_shared_branch_backbone():
        net = list(resnet50(pretrained=True).children())[6:8]  # Gets conv4 and conv5 layers.
        net[0] = net[0][1:]  # We already have conv4_1 from backbone. So, omit it here.
        """
        TODO: This part is not clear in MGN paper. Also, disabling down sample code is suspicious.
        #net[1] = net[1][0]  # We only want conv5_1 not all conv5.
        """
        net[1][0].downsample[0].stride = (1, 1)
        net[1][0].conv2.stride = (1, 1)
        net[1][0].stride = 1
        return nn.Sequential(*net)

    @staticmethod
    def _get_branch():
        net = list(resnet50(pretrained=True).children())[6:8]  # Gets conv4 and conv5 layers.
        net[0] = net[0][1:]  # We already have conv4_1 from backbone. So, omit it here.
        """
        TODO: This part is not clear in MGN paper. Also, disabling down sample code is suspicious.
        #net[1] = net[1][0]  # We only want conv5_1 not all conv5.
        """
        net[1][0].downsample[0].stride = (1, 1)
        net[1][0].conv2.stride = (1, 1)
        net[1][0].stride = 1
        return nn.Sequential(*net)

    def forward(self, images):

        bb_out = self.backbone(images)

        b0_features = self.branch_bb0(bb_out)
        b1_features = self.branch_bb1(bb_out)
        b2_features = self.branch_bb2(bb_out)

        to_conv2048 = []
        to_conv512 = []

        to_conv2048.append(self.gmp_full1(b1_features))      # branch B
        to_conv2048.append(self.gmp_full2(b2_features))      # branch C
        # branch B -----------------------------------------------------------------------------------------------------
        to_conv2048.extend(torch.split(self.gmp_horizontal_B(b1_features), split_size_or_sections=1, dim=2))
        to_conv2048.extend(torch.split(self.gmp_horizontal_D(b1_features), split_size_or_sections=1, dim=2))
        # branch C -----------------------------------------------------------------------------------------------------
        to_conv2048.extend(torch.split(self.gmp_vertical_C(b2_features), split_size_or_sections=1, dim=3))
        to_conv2048.extend(torch.split(self.gmp_vertical_E(b2_features), split_size_or_sections=1, dim=3))
        # branch A -----------------------------------------------------------------------------------------------------
        to_conv512.extend([item.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                           for item in torch.split(b0_features, split_size_or_sections=512, dim=1)])
        unified_conv_out = [operator(operand).squeeze() for operator, operand in zip(self.conv2048s, to_conv2048)] + \
                           [operator(operand).squeeze() for operator, operand in zip(self.conv512s, to_conv512)]

        if not self.training:
            return torch.cat(unified_conv_out, dim=1)

        triplet_losses = unified_conv_out[:2]
        sm_losses = [operator(operand) for operator, operand in zip(self.fcs, unified_conv_out)]
        return triplet_losses + sm_losses


if __name__ == '__main__':
    model = PRNet(123).cuda()
    from data import Data
    dataset_object = Data('C:/datasets/vreid/veri')
    for batch, (images, labels) in enumerate(dataset_object.train_loader):
        images = images.cuda()
        outs = model(images)
