import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from MACMD import MACMD





class PVT_B2_MACMD(nn.Module):
    def __init__(self, n_classes=9, n_channels =  1, img_size = 224,  encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_path'):
        super(PVT_B2_MACMD, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        
        #backbone network initialization with pretrained weight
        if encoder == 'maxvit':
            self.backbone = MaxViT4Out_Small(n_class=n_classes, img_size=224)
            # path = pretrained_dir + '/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth'
            channels = [96, 192, 384, 768]
        elif encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = pretrained_dir + '/pvt/pvt_v2_b0.pth'
            channels= [32, 64, 160, 256] 
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = pretrained_dir + '/pvt/pvt_v2_b1.pth'
            channels= [64, 128, 320, 512] #[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = pretrained_dir + '/pvt/pvt_v2_b2.pth'
            channels= [64, 128, 320, 512] #[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = pretrained_dir + '/pvt/pvt_v2_b3.pth'
            channels=[64, 128, 320, 512] #[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = pretrained_dir + '/pvt/pvt_v2_b4.pth'
            channels=[64, 128, 320, 512] #[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = pretrained_dir + '/pvt/pvt_v2_b5.pth'
            channels=[64, 128, 320, 512] #[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[64, 128, 320, 512] #[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[64, 128, 320, 512] #[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[256, 512, 1024, 2048] # [2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[256, 512, 1024, 2048] #[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[256, 512, 1024, 2048] #[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = pretrained_dir + '/pvt/pvt_v2_b2.pth'
            channels= [64, 128, 320, 512] #[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2_b2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            print('Pretrain weights loaded.')
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = MACMD(embed_dims=channels, img_size = img_size)
        
        print('Model %s created, param count: %d' %
                     ('MACMD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        # self.out_head4 = nn.Conv2d(channels[0], n_classes, 1)
        self.out_head3 = nn.Conv2d(channels[0], n_classes, 1)
        self.out_head2 = nn.Conv2d(channels[1], n_classes, 1)
        #self.out_head1 = nn.Conv2d(channels[3], n_classes, 1)
        self.out_head = nn.Conv2d(32, n_classes, 1)
        
    def forward(self, x, mode='test'):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        #print(x.shape)
        # encoder
        x1, x2, x3, x4 = self.backbone(x)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
       # decoder
        dec_outs, dec_outs1, dec_outs2 = self.decoder(x1, x2, x3, x4)
        # print(dec_outs.shape)
        # print(dec_outs1.shape)
        # print(dec_outs2.shape)
        # prediction heads  
        # p4 = self.out_head4(dec_outs[0])
        p_3 = self.out_head3(dec_outs1)
        p_2 = self.out_head2(dec_outs2)
        p_1 = self.out_head(dec_outs)
        #print(p1.shape)


        # p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p_2, scale_factor=8, mode='bilinear')
        p2 = F.interpolate(p_3, scale_factor=4, mode='bilinear')
        p1 = F.interpolate(p_1, scale_factor=2, mode='bilinear')

        # if mode == 'test':
        #     return [p4, p3, p2, p1]
        
        #return [p4, p3, p2, p1]
        #return p1 , p2, p3
        return torch.sigmoid(p1) , torch.sigmoid(p2), torch.sigmoid(p3)
               

        
if __name__ == '__main__':
    model = PVT_B2_MACMD().cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    P1, P2, P3 = model(input_tensor)
    print(P1.size(), P2.size(), P3.size())