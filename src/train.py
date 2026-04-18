import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image

from data import get_dataloader
from model import Generator, Discriminator
import matplotlib.pyplot as plt

# Path
data_dir = r"D:\GAN-IMG\Global-Challenge\Brain_Tumor_Dataset"

os.makedirs("generated_images", exist_ok=True)

# Load data
loader = get_dataloader(data_dir)

# Models
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 5
g_losses = []
d_losses = []

# 🔥 LOG FILE
log_file = open("training_log.txt", "w")

print("Training started...")

for epoch in range(epochs):

    epoch_g_loss = 0
    epoch_d_loss = 0

    for i, (imgs, _) in enumerate(loader):

        batch_size = imgs.size(0)

        real = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        z = torch.randn(batch_size, 100, 1, 1)

        gen_imgs = generator(z)

        g_loss = criterion(discriminator(gen_imgs), real)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        real_loss = criterion(discriminator(imgs), real)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch+1} Batch {i}")

    # Average loss
    avg_g = epoch_g_loss / len(loader)
    avg_d = epoch_d_loss / len(loader)

    g_losses.append(avg_g)
    d_losses.append(avg_d)

    # 🔥 LOGGING
    log = f"Epoch {epoch+1} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}"
    print(log)
    log_file.write(log + "\n")

    # Save generated images
    save_image(gen_imgs[:25], f"generated_images/epoch_{epoch+1}.png", nrow=5, normalize=True)
    print(f"Saved images for epoch {epoch+1}")

# Close log file
log_file.close()

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# 🔥 GRAPH (IMPROVED)
plt.figure(figsize=(8,5))

plt.plot(g_losses, label="Generator Loss", color='blue')
plt.plot(d_losses, label="Discriminator Loss", color='red')

plt.title("GAN Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("loss_graph.png")

plt.show()