import torch
import torch.nn as nn
import torchaudio


class res_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(res_block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in != self.dim_out:
            self.s_conv = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_out, (1, ), bias=False))
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_in, (3, ), padding=(1, )))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(self.dim_in, self.dim_out, (1, )))
        self.avgpool = nn.AvgPool1d(kernel_size=(2,))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        sc = x.clone()
        if self.dim_in != self.dim_out:
            sc = self.s_conv(sc)
            sc = self.avgpool(sc)
        else:
            sc = self.avgpool(sc)

        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.lrelu(x)
        x = self.conv2(x)

        return x + sc

class ResnetBlock(nn.Module):
    def __init__(self, dim, spk_emb, dilation):
        super(ResnetBlock, self).__init__()

        self.dim = dim

        self.s_conv = nn.utils.weight_norm(nn.Conv1d(dim, dim, kernel_size=(1,)))

        self.b_conv = nn.utils.weight_norm(nn.Conv1d(dim, 2*dim, kernel_size=(3,), dilation=(dilation, ), padding=(dilation, ), padding_mode='reflect'))
        if spk_emb is not None:
            self.b_conv2 = nn.utils.weight_norm(nn.Conv1d(spk_emb, 2*dim, kernel_size=(1,)))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.b_conv3 = nn.utils.weight_norm(nn.Conv1d(dim, dim, kernel_size=(1,), dilation=(dilation, )))

    def forward(self, x, spk_emb):
        s = self.s_conv(x)
        b = self.b_conv(x)
        if spk_emb is not None:
            # spk_emb = torch.unsqueeze(spk_emb, 2)
            b = b + self.b_conv2(spk_emb)
        b = self.tanh(b[:, :self.dim, :]) * self.sigmoid(b[:, self.dim:, :])
        b = self.b_conv3(b)
        return s + b

class DownBlock(nn.Module):
    def __init__(self, x, r, mult):
        super(DownBlock, self).__init__()
        self.ResnetBlock1 = ResnetBlock(x, None, 1)
        self.ResnetBlock2 = ResnetBlock(x, None, 3)
        self.ResnetBlock3 = ResnetBlock(x, None, 9)
        self.ResnetBlock4 = ResnetBlock(x, None, 27)
        self.gelu = nn.GELU()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(x, mult * 32, kernel_size=(r*2, ), stride=(r, ), padding=(r // 2 + r % 2, )))

    def forward(self, x):
        x = self.ResnetBlock1(x, None)
        x = self.ResnetBlock2(x, None)
        x = self.ResnetBlock3(x, None)
        x = self.ResnetBlock4(x, None)
        x = self.gelu(x)
        x = self.conv1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, x, r, mult):
        super(UpBlock, self).__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.utils.weight_norm(nn.ConvTranspose1d(x, mult * 32 // 2, kernel_size=(r * 2,), stride=(r,), padding=(r // 2 + r % 2, )))
        self.ResnetBlock1 = ResnetBlock(mult * 32 // 2, 128, 1)
        self.ResnetBlock2 = ResnetBlock(mult * 32 // 2, 128, 3)
        self.ResnetBlock3 = ResnetBlock(mult * 32 // 2, 128, 9)
        self.ResnetBlock4 = ResnetBlock(mult * 32 // 2, 128, 27)

    def forward(self, x, spk_emb):
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.ResnetBlock1(x, spk_emb)
        x = self.ResnetBlock2(x, spk_emb)
        x = self.ResnetBlock3(x, spk_emb)
        x = self.ResnetBlock4(x, spk_emb)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(1, 32, kernel_size=(7,), padding=(3, ), padding_mode='reflect'))
        self.DownBlock1 = DownBlock(32, 2, 2)
        self.DownBlock2 = DownBlock(64, 2, 4)
        self.DownBlock3 = DownBlock(128, 8, 8)
        self.DownBlock4 = DownBlock(256, 8, 16)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=(7,), padding=(3, ), padding_mode='reflect'))
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(512, 4, kernel_size=(7,), padding=(3, ), padding_mode='reflect', bias=False))
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.DownBlock4(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = x / torch.sum(x**2 + 1e-12, dim=1, keepdim=True)**0.5
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.hop_length = 256
        mult = 16
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(4, mult * 32, kernel_size=(7,), padding=(3,), padding_mode='reflect'))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(mult * 32, mult * 32, kernel_size=(7,), padding=(3,), padding_mode='reflect'))
        self.UpBlock1 = UpBlock(mult * 32, 8, 16)
        self.UpBlock2 = UpBlock(256, 8, 8)
        self.UpBlock3 = UpBlock(128, 2, 4)
        self.UpBlock4 = UpBlock(64, 2, 2)
        self.gelu = nn.GELU()
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=(7,), padding=(3, ), padding_mode='reflect'))
        self.tanh = nn.Tanh()

    def forward(self, x, spk_emb):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.UpBlock1(x, spk_emb)
        x = self.UpBlock2(x, spk_emb)
        x = self.UpBlock3(x, spk_emb)
        x = self.UpBlock4(x, spk_emb)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x

class Speaker(nn.Module):
    def __init__(self):
        super(Speaker, self).__init__()
        self.dim = 128
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, f_min=80.0, f_max=7600.0)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(80, 32, kernel_size=(3,), padding=(1, )))
        self.res_block1 = res_block(32, 64)
        self.res_block2 = res_block(64, 128)
        self.res_block3 = res_block(128, 256)
        self.res_block4 = res_block(256, 512)
        self.res_block5 = res_block(512, 512)
        self.avgpool = nn.AdaptiveAvgPool1d((1, ))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.m_conv = nn.utils.weight_norm(nn.Conv1d(512, self.dim, kernel_size=(1,)))
        self.v_conv = nn.utils.weight_norm(nn.Conv1d(512, self.dim, kernel_size=(1,)))

    def forward(self, x):
        out = self.mel_spec(x)
        out = torch.squeeze(out, 1)
        # out = out * 1e4 + 1
        out = torch.log(out + 1e-6)
        out = self.conv1(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.avgpool(out)
        # out = torch.squeeze(out, -1)
        out = self.lrelu(out)
        mu = self.m_conv(out)
        logvar = self.v_conv(out)
        return mu, logvar

class NVCNet(nn.Module):
    def __init__(self):
        super(NVCNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.speaker = Speaker()

    def forward(self, x, y):
        style = self.embed(y)[0]
        content = self.encode(x)
        # style = torch.unsqueeze(style, 2)
        out = self.decode(content, style)
        return out
        # return style, content

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, content, style):
        x = self.decoder(content, style)
        return x

    def embed(self, x):
        mu, logvar = self.speaker(x)
        spk_emb = self.sample(mu, logvar)
        # spk_emb = torch.squeeze(spk_emb, -1)
        return spk_emb, mu, logvar

    def sample(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu

    def kl_loss(self, mu, logvar):
        return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar, dim=1))

class NDiscrminator(nn.Module):
    def __init__(self):
        super(NDiscrminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(1, 16, kernel_size=(15,), padding=(7, ), padding_mode='reflect'))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4))
        self.conv3 = nn.utils.weight_norm(
            nn.Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16))
        self.conv4 = nn.utils.weight_norm(
            nn.Conv1d(256, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=64))
        self.conv5 = nn.utils.weight_norm(
            nn.Conv1d(1024, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=256))
        self.conv6 = nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=(5,), padding=(2, )))
        self.conv7 = nn.utils.weight_norm(nn.Conv1d(1024, 103, kernel_size=(3,), padding=(1, )))

    def forward(self, x, y, batch):
        results = []

        output = self.conv1(x)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv2(output)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv3(output)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv4(output)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv5(output)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv6(output)
        output = self.lrelu(output)
        results.append(output)

        output = self.conv7(output)
        if y is not None:
            idx = torch.stack([torch.arange(batch).cuda(), torch.reshape(y, (batch,))], dim=0)
            # output = torch.index_select(output, 1, torch.reshape(y, (batch, )))
            output = output[idx.tolist()]
        results.append(output)

        return results

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis1 = NDiscrminator()
        self.dis2 = NDiscrminator()
        self.dis3 = NDiscrminator()
        self.avgpool = nn.AvgPool1d(kernel_size=(4,), stride=(2,), padding=(1,), count_include_pad=False)

        self.mel1 = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2048, win_length=2048, n_mels=80,
                                                         hop_length=512)
        self.mel2 = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024, n_mels=80,
                                                         hop_length=256)
        self.mel3 = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=512, win_length=512, n_mels=80,
                                                         hop_length=128)

        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.celoss = nn.BCEWithLogitsLoss()
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x, y, batch):
        results = []
        results.append(self.dis1(x, y, batch))
        x = self.avgpool(x)
        results.append(self.dis2(x, y, batch))
        x = self.avgpool(x)
        results.append(self.dis3(x, y, batch))
        return results

    def spectral_loss(self, x, target):
        loss = []
        sx = self.mel1(x)
        # sx = sx * 1e4 + 0.001
        sx = torch.log(sx + 1e-6)
        st = self.mel1(target)
        # st = st * 1e4 + 0.001
        st = torch.log(st + 1e-6)
        st.required_grad = False
        # st = st.detach()
        loss.append(self.mseloss(sx, st))

        sx = self.mel2(x)
        # sx = sx * 1e4 + 0.001
        sx = torch.log(sx + 1e-6)
        st = self.mel2(target)
        # st = st * 1e4 + 0.001
        st = torch.log(st + 1e-6)
        st.required_grad = False
        # st = st.detach()
        loss.append(self.mseloss(sx, st))

        sx = self.mel3(x)
        # sx = sx * 1e4 + 0.001
        sx = torch.log(sx + 1e-6)
        st = self.mel3(target)
        # st = st * 1e4 + 0.001
        st = torch.log(st + 1e-6)
        st.required_grad = False
        # st = st.detach()
        loss.append(self.mseloss(sx, st))

        return sum(loss)

    def preservation_loss(self, x, target):
        return self.mseloss(x, target)

    def perceptual_loss(self, x, target, batch):
        loss = []
        out_x, out_t = self(x, None, batch), self(target, None, batch)
        for (a, t) in zip(out_x, out_t):
            for la, lt in zip(a[:-1], t[:-1]):
                lt.required_grad = False
                loss.append(self.l1loss(la, lt))
        return sum(loss) / 3

    def adverarial_loss(self, results, v):
        loss = []
        for out in results:
            t = torch.FloatTensor([v]).cuda()
            # r = self.celoss(self.sigmoid(out[-1]), t.repeat(out[-1].shape))
            r = self.celoss(out[-1], t.repeat(out[-1].shape))
            # loss.append(torch.mean(r))
            loss.append(r)
        return sum(loss)
