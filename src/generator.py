import torch
from model import Generator
from torchvision.utils import save_image
import os

os.makedirs("final_generated", exist_ok=True)

generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

z = torch.randn(25, 100)
gen_imgs = generator(z)

save_image(gen_imgs, "final_generated/final.png", nrow=5, normalize=True)