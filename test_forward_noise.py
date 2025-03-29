import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from models.diffusion import GaussianDiffusion

if __name__=="__main__":
    device = 'cpu'
    image_size = 128
    transform2Tensor = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transform2Image = transforms.ToPILImage()
    img = Image.open('datasets/data/cat.jpg')
    x0 = transform2Tensor(img).unsqueeze(0)
    diffusion = GaussianDiffusion(device=device, time_steps=500)

    plt.figure(figsize=(16, 8))
    for index, time in enumerate([10, 20, 100, 200, 300, 500]):
        noise = torch.randn(x0.shape)
        xt = diffusion.q_xt_x0(x0, torch.tensor([time]), noise)
        xt = (xt*0.5)+0.5
        xt_image = transform2Image(xt.squeeze(0))
        plt.subplot(1, 6, index+1)
        plt.imshow(xt_image)
        plt.axis('off')
        plt.title(f"t={time}")
    plt.show()


