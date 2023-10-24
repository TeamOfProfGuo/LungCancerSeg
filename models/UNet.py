import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D UNet
class UNet3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet3D, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
        
        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


# 2D UNet
class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2)


        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2)


        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.maxpool3 = nn.MaxPool2d(2)


        self.conv7 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.maxpool4 = nn.MaxPool2d(2)


        self.conv9 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU())
        self.dropout2 = nn.Dropout(0.5)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv11 = nn.Sequential(nn.Conv2d(1024, 512, 2, padding='same'), nn.ReLU())
        # concatnate
        self.conv12 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU())
        self.conv13 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
        self.upsampling2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv14 = nn.Sequential(nn.Conv2d(512, 256, 2, padding='same'), nn.ReLU())
        # concatnate
        self.conv15 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),  nn.ReLU())
        self.conv16 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.upsampling3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv17 = nn.Sequential(nn.Conv2d(256, 128, 2, padding='same'), nn.ReLU())
        # concatnate
        self.conv18 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.conv19 = nn.Sequential((nn.Conv2d(128, 128, 3, padding=1)), nn.ReLU())
        self.upsampling4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv20 = nn.Sequential(nn.Conv2d(128, 64, 2, padding='same'), nn.ReLU())
        # concatnate
        self.conv21 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.conv22 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        # classifier
        self.conv23 = nn.Sequential(nn.Conv2d(64, 2, 3, padding=1), nn.ReLU())
        self.conv24 = nn.Sequential(nn.Conv2d(2, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        conv2 = self.conv2(x)
        x = self.maxpool1(conv2)   # [B, 64, 256, 256]

        x = self.conv3(x)
        conv4 = self.conv4(x)
        x = self.maxpool2(conv4) # [B, 128, 128, 128]

        x = self.conv5(x)
        conv6 = self.conv6(x)
        x = self.maxpool3(conv6)  # [B, 256, 64, 64]

        x = self.conv7(x)
        x = self.conv8(x)
        dp1 = self.dropout1(x)    # [B, 512, 64, 64]
        x = self.maxpool4(dp1)    # [B, 512, 32, 32]

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.dropout2(x)
        x = self.upsampling1(x)  # [B, 1024, 64, 64]

        x = self.conv11(x)
        # concatnate
        x = torch.cat((dp1, x), dim=1)  # [B, 1024, 64, 64]
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.upsampling2(x)

        x = self.conv14(x)
        # concatnate
        x = torch.cat((conv6, x), dim=1)   # [B, 512, 128, 128]
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.upsampling3(x)

        x = self.conv17(x)
        # concatnate
        x = torch.cat((conv4, x), dim=1)   # [B, 256, 256, 256]
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.upsampling4(x)

        x = self.conv20(x)
        # concatnate
        x = torch.cat((conv2, x), dim=1)  # [B, 128, 512, 512]
        x = self.conv21(x)      # [B, 64, 512, 512]
        x = self.conv22(x)

        x = self.conv23(x)   # [B, 2, 512, 512]
        x = self.conv24(x)   # [B, 1, 512, 512]

        return x
