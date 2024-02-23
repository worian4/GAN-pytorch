import torch.nn as nn


class _netG_1(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf, n_extra_layers_g):
        super(_netG_1, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Extra layers
        for t in range(n_extra_layers_g):
            self.main.add_module('extra-layers-{0}{1}conv'.format(t, ngf),
                            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            self.main.add_module('extra-layers-{0}{1}batchnorm'.format(t, ngf),
                            nn.BatchNorm2d(ngf))
            self.main.add_module('extra-layers-{0}{1}relu'.format(t, ngf),
                            nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('final_layerdeconv', 
        	             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.main.add_module('final_layertanh', 
        	             nn.Tanh())

        self.main = self.main


    def forward(self, input):
        return self.main(input)

class _netD_1(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf,  n_extra_layers_d):
        super(_netD_1, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Extra layers
        for t in range(n_extra_layers_d):
            self.main.add_module('extra-layers-{0}{1}conv'.format(t, ndf * 8),
                            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))
            self.main.add_module('extra-layers-{0}{1}batchnorm'.format(t, ndf * 8),
                            nn.BatchNorm2d(ndf * 8))
            self.main.add_module('extra-layers-{0}{1}relu'.format(t, ndf * 8),
                            nn.LeakyReLU(0.2, inplace=True))


        self.main.add_module('final_layersconv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        self.main.add_module('final_layerssigmoid', nn.Sigmoid())
        self.main = self.main

    def forward(self, input):
        return self.main(input)




