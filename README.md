# Pytorch Implementation of SN-GAN with CIFAR10

## Acknowledgement
clone, edit from https://github.com/isr-wang/SNGAN

## Requirements
no nsml

## Paper
[Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-)

## Run Example
```{python}
python main.py --sn
```

## Architecture

GAN Architecture is adopted from the papers' Appendix B.4 for CIFAR10
<p align="center">
  <img src="./assets/architecture.png">
</p>

## Results

*Generated images*
<p align="center">
  <img src="./assets/gen_example.png">
</p>

*Loss*
<p align="center">
  <img src="./assets/g_loss.png">
</p>
<p align="center">
  <img src="./assets/d_loss.png">
</p>
<p align="center">
  <img src="./assets/d_loss_fake.png">
</p>
<p align="center">
  <img src="./assets/d_loss_real.png">
</p>


