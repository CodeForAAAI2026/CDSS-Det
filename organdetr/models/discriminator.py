import torch.nn as nn
import torch

class FCDiscriminator_img_3D(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img_3D, self).__init__()

        '''
        self.bn1 = nn.BatchNorm3d(ndf1)
        self.bn2 = nn.BatchNorm3d(ndf2)
        '''

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(num_classes, ndf1, kernel_size=3, padding=1)
        self.insnorm1 = nn.InstanceNorm3d(num_features=ndf1, affine=True, eps=1e-5)
        self.conv2 = nn.Conv3d(ndf1, ndf2, kernel_size=3, padding=1)
        self.insnorm2 = nn.InstanceNorm3d(num_features=ndf2, affine=True, eps=1e-5)
        self.conv3 = nn.Conv3d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv3d(ndf2, 1, kernel_size=3, padding=1)

        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        # Apply convolutional layers with Leaky ReLU activation
        # print(f"Before conv1 and norm: {torch.norm(x)}")
        x = self.conv1(x)
        x = self.insnorm1(x)
        x = self.leaky_relu(x)
        # print(f"After conv1 and norm: {torch.norm(x)}")

        x = self.conv2(x)
        x = self.insnorm2(x)
        x = self.leaky_relu(x)
        # print(f"After conv2 and norm: {torch.norm(x)}")

        x = self.conv3(x)
        x = self.leaky_relu(x)
        # print(f"After conv3: {torch.norm(x)}")

        # Apply the classifier to get the final output
        x = self.classifier(x)
        # print(f"Logits: {torch.norm(x)}")  # Final logits before applying sigmoid

        return x

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)
