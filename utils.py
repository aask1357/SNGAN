import torch.nn as nn
from torchvision.models.inception import inception_v3
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def dataloader(dataset, root, batch_size, num_workers, train):
    data = getattr(datasets, dataset)(
        root=root, train=train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    )

    return torch.utils.data.DataLoader(data,
        shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)


class GenLoader:
    def __init__(self, G, n_data, batch_size, device):
        assert batch_size > 0
        assert n_data >= batch_size, f"n_data({n_data}) < batch_size({batch_size})"
        self.G = G
        self.z_dim = G.z_dim
        self.device = device
        self.n_data = n_data
        self.batch_size = batch_size

    def __iter__(self):
        self.index = 0
        return self
 
    def __next__(self):
        if self.index >= self.n_data:
            raise StopIteration
        
        if self.index + self.batch_size > self.n_data:
            bs = self.n_data - self.index
        else:
            bs = self.batch_size
        self.index += self.batch_size
        
        with torch.no_grad():
            z = torch.randn(bs, self.z_dim, device=self.device)
            x = self.G(z).clamp(-1, 1)#.permute(0, 2, 3, 1)
        return x
    
    def __len__(self):
        return (self.n_data - 1) // self.batch_size + 1


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
    def __init__(self, dataloader, device):
        self.device = device
        self.dataloader = dataloader
        self.n_data = dataloader.n_data
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(self.device)
        

    def inception_score(self, resize=True, splits=10):
        # code from - https://github.com/sbarratt/inception-score-pytorch
        # Get predictions
        preds = np.zeros((self.n_data, 1000))
        with torch.no_grad():
            i = 0
            for x in self.dataloader:
                batch_size = x.shape[0]
                if resize:
                    x = self.up(x)
                x = self.inception_model(x)
                x = F.softmax(x, dim=1).data.cpu().numpy()

                preds[i:i + batch_size] = x
                i += batch_size
        del x
        torch.cuda.empty_cache()
        
        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (self.n_data // splits): (k+1) * (self.n_data // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    
    class IgnoreLabelLoader():
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.n_data = len(dataloader.dataset)

        def __iter__(self):
            self.iterator = iter(self.dataloader)
            return self
        
        def __next__(self):
            try:
                x, _ = next(self.iterator)
                return x.to(device)
            except StopIteration:
                raise StopIteration
    
    dataloader = dataloader(
        dataset='CIFAR10',
        root='/home/ash-arch/Documents/datasets/cifar10',
        batch_size=64,num_workers=4,train=True
    )
    dataloader = IgnoreLabelLoader(dataloader)

    model = Inception(dataloader=dataloader, device=device)

    print ("Calculating Inception Score...")
    print (model.inception_score(resize=True, splits=10))
