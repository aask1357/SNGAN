import torch.nn as nn
from torchvision.models.inception import inception_v3
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy


def to_var(x):
    """
        x: torch Tensor
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    """
        x: torch Variable
    """
    return x.data.cpu().numpy()


def init_params(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class Inception:
    def __init__(self, cuda=True):
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.inception_model = inception_v3(pretrained=True, transform_input=False).type(self.dtype)
        self.inception_model.eval()
        
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(self.dtype)
        

    def inception_score(self, imgs, batch_size=32, resize=False, splits=10):
        """Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [0, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits

        code from - https://github.com/sbarratt/inception-score-pytorch
        """
        N = len(imgs)

        assert batch_size > 0
        assert N > batch_size, "{} <= {}".format(N, batch_size)

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # Load inception model
        def get_pred(x):
            if resize:
                x = self.up(x)
            x = self.inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Get predictions
        preds = np.zeros((N, 1000))
        with torch.no_grad():
            for i, batch in enumerate(dataloader, 0):
                batch = batch.type(self.dtype)
                batchv = to_var(batch)
                batch_size_i = batch.size()[0]

                preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
        del batch, batchv, batch_size_i
        torch.cuda.empty_cache()

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
        transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))
