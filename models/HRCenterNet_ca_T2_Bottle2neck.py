"""
    @author mhuazheng
    @create 2021-10-09 21:08
"""
import torch
from torch import nn
from models.modules import BasicBlock, Bottleneck, Bottle2neck
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            # nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.Conv2d(channel, 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Conv2d(8, channel, 1, padding=0, bias=True),
            # nn.Sigmoid()
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        z = self.max_pool(x)
        y = self.ca(y)
        z = self.ca(z)
        channe_sum = y + z
        result = F.sigmoid(channe_sum)
        return x * result


"""
当只包含一个分支的时候，便提取该分支的特征，并且最后没有融合的模块，直接返回提取结果；
当分支的数目大于1时，先对每一个分支分别进行特征提取，然后对特征提取结果进行特征融合，最后再返回融合结果
"""


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        """
        ModuleList 可以存储多个 model，传统的方法，一个model就要写一个 forward ，但是如果将它们存到一个 ModuleList 的话，就可以使用一个 forward。
        ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。
        当添加 nn.ModuleList 作为 nn.Module 对象的一个成员时（即当我们添加模块到我们的网络时），所有 nn.ModuleList 内部的 nn.Module 的 parameter 也被添加作为 我们的网络的 parameter。
        使用 ModuleList 也可以使得网络的结构具有灵活性，比如我需要将网络的层数设置为变量，传统的方法要借助 list 实现，并且不方便，而使用 ModuleList就可以简化这个操作
        """

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:  # 自身与自身之间不需要融合
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:

                    """
                    此时的目标是将所有分支上采样到和i分支相同的分辨率并融合，也就是说j所代表的分支分辨率比i分支低，
                    2**(j-i)表示j分支上采样这么多倍才能和i分支分辨率相同。
                    先使用1x1卷积将j分支的通道数变得和i分支一致，进而跟着BN，
                    然后依据上采样因子将j分支分辨率上采样到和i分支分辨率相同，此处使用最近邻插值, 并将二者进行求和
                    当j>i条件的循环被全部执行完毕之后，所有的比i分辨率低的分支都与i进行了融合
                    """

                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:

                    """
                    此时j所代表的分支分辨率比i分支高，正好和上面相反。
                    此时再次内嵌了一个循环，这层循环的作用是当i-j > 1时，也就是说两个分支的分辨率差了不止二倍，此时还是两倍两倍往下采样，
                    例如i-j = 2时，j分支的分辨率比i分支大4倍，就需要下采样两次，循环次数就是2
                    """

                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)  # assert 如果表达式为假，触发异常；如果表达式为真，不执行任何操作。

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class _HRCenterNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(_HRCenterNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            # Bottleneck(64, 64, downsample=downsample),  # 这里用downsample是因为上一层的输出是64和下一层的输出256维度不匹配
            # Bottleneck(256, 64),
            # Bottleneck(256, 64),
            # Bottleneck(256, 64),
            Bottle2neck(64, 64, downsample=downsample),  # 这里用downsample是因为上一层的输出是64和下一层的输出256维度不匹配
            Bottle2neck(256, 64),
            Bottle2neck(256, 64),
            Bottle2neck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)  low + high
        self.low_resolution1 = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, nof_joints, kernel_size=(1, 1), stride=(1, 1)),

            # nn.Upsample(scale_factor=(2.0 ** 1), mode='nearest'),

            nn.Sigmoid()
        )

        self.low_resolution2 = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, nof_joints, kernel_size=(1, 1), stride=(1, 1)),

            nn.Upsample(scale_factor=(2.0 ** 1), mode='nearest'),

            nn.Sigmoid()
        )

        self.high_resolution = nn.Sequential(
            nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(5, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(5, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(5, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            BasicBlock(5, 5),
            BasicBlock(5, 5),
            BasicBlock(5, 5),
            BasicBlock(5, 5),

            nn.Conv2d(5, nof_joints, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

        # 注意力机制ca
        self.calayer = CALayer(5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        # x = self.final_layer(x[0])

        x_low1 = self.low_resolution1(x[0])
        x_low2 = self.low_resolution2(x[0])

        x_high = self.high_resolution(x_low1)

        x_fusion = (x_high + x_low2) / 2

        # 注意力机制ca
        # x_high = self.calayer(x_high)
        # x_low = self.calayer(x_low2)

        x_fusion = self.calayer(x_fusion)

        return x_fusion


def HRCenterNet():
    model = _HRCenterNet(32, 5, 0.1)

    return model
