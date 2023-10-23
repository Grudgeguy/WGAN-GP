import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os 
from tqdm import tqdm
from model import Generator,Discriminator
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from utils import initialize_weights,show_batch,gradient_penalty,denorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 64
features_gen = 16
features_critic = 16
batch_size = 64
z_dim = 100
epochs = 3
channels = 3
lr = 1e-4
lambda_gp = 10
critic_iter = 5

transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size), 
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels)], [0.5 for _ in range(channels)]
        ),
    ]
)

dataset = ImageFolder(r"C:\Users\dhruv\Untitled Folder\Face Generation\celeb_dataset", transform=transform)

dataloader = DataLoader(dataset, batch_size, shuffle=True)
gen = Generator(z_dim, channels, features_gen).to(device)
critic = Discriminator(channels, features_critic).to(device)
initialize_weights(gen)
initialize_weights(critic)

show_batch(dataloader)

opt_gen = optim.Adam(gen.parameters(), lr,betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr,betas=(0.0,0.9))

# Replace 'output_directory' with your desired directory path
output_directory = 'WGAN_GP_Celeb'
os.makedirs(output_directory, exist_ok=True)

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
step = 0

gen.train()
critic.train()
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        
        for _ in range(critic_iter):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic,real,fake,device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) +lambda_gp*gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
        #Train Generator
        output = critic(fake).reshape(-1)
        loss_gen = - torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and save a grid of generated fake images
        if batch_idx % 300 == 0 and batch_idx != 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)

                # Save a grid of generated fake images
                fake_grid = vutils.make_grid(fake, normalize=True, padding=2, nrow=8)  # Adjust nrow as needed

                image_filename = os.path.join(output_directory, f'fake_images_epoch{epoch}_batch{batch_idx}.png')
                vutils.save_image(fake_grid, image_filename)

                # Display real and fake images in TensorBoard
                img_grid_real = vutils.make_grid(real[:32], normalize=True, padding=2, nrow=8)
                img_grid_fake = vutils.make_grid(fake[:32], normalize=True, padding=2, nrow=8)

            step += 1
