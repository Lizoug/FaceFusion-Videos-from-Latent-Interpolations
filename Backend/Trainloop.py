import torch
from torch.utils.data import DataLoader
from Dataset import CelebADataset, transform
from Critic import Critic
from Generator import Generator
from tqdm import tqdm
from Checkpoints import Checkpoint


class Trainer:
    def __init__(self, chkpt_path="model.pt", save_interval=10):
        # Hyperparameters
        self.lr = 0.0002
        self.beta1 = 0.5
        self.num_epochs = 10
        self.batch_size = 32
        self.nz = 100
        self.ngf = 64
        self.ndf = 64

        self.DATASET_DIR = r"C:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Project_Data\img_align_celeba\img_align_celeba"


        # Set device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up DataLoader
        self.data_loader = DataLoader(CelebADataset(self.DATASET_DIR, 
                                                    transform=transform(64)), 
                                                    batch_size=self.batch_size, 
                                                    shuffle=True)
        
        # Initialize networks
        self.critic = Critic(nc=3, ndf=self.ndf).to(self.device)
        self.generator = Generator(nz=self.nz, ngf=self.ngf, nc=3).to(self.device)

        # Set up Optimizers
        # By setting this value close to 1, the optimizer gives more 
        # weight to past squared gradients, making the estimate more stable over time.
        self.optimizerD = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # Initialize the checkpoint handler
        self.checkpoint_handler = Checkpoint(chkpt_path)
        self.save_interval = save_interval
        self.batch_count = 0

        # Load checkpoint if available
        generator_state, critic_state, optimG_state, optimD_state, batch_count = self.checkpoint_handler.load()
        
        if generator_state is not None:
            self.generator.load_state_dict(generator_state)
            self.critic.load_state_dict(critic_state)
            self.optimizerG.load_state_dict(optimG_state)
            self.optimizerD.load_state_dict(optimD_state)
            self.batch_count = batch_count
        else:
            print("Could not find checkpoint, starting from scratch")

    # Set up Loss function (Wasserstein loss)
    def critic_loss(self, real_output, fake_output):
        """Calculate the critic's loss based on the Wasserstein distance."""
        return torch.mean(real_output) - torch.mean(fake_output) #maximieren

    # Set up the generator loss
    def generator_loss(self, fake_output):
        """Calculate the generator's loss based on the Wasserstein distance."""
        return -torch.mean(fake_output) # min wegen -


    def train(self):
        # Training loop
        for epoch in range(self.num_epochs):
                # Wrap the data loader with tqdm for a progress bar
                bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
                criticloss = 0
                cnt = 0
                for i, real_images in enumerate(bar):
                    
                    # Transfer real images to the device
                    real_images = real_images.to(self.device)
                    batch_size = real_images.size(0)
                    
                    # -----------------------------------------
                    # Train Critic
                    # -----------------------------------------
                    
                    # reseting the gradients of the optimizer's managed parameters to zero, 
                    # preventing accumulation of gradients across multiple training iterations.
                    self.optimizerD.zero_grad()
                    
                    # Compute loss with real images
                    outputs_real = self.critic(real_images)
                    
                    # Generate fake images
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    fake_images = self.generator(z)
                    
                    # Compute loss with fake images
                    # prevent the generator from updating, ensuring only the discriminator learns from the data
                    outputs_fake = self.critic(fake_images.detach())
                    
                    # Get critic loss and perform backward propagation
                    c_loss = self.critic_loss(outputs_real, outputs_fake)
                    c_loss.backward()
                    self.optimizerD.step()

                    criticloss += c_loss.item()
                    cnt += batch_size

                    # Clip the critic's weights to ensure 1-Lipschitz condition
                    for p in self.critic.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    
                    # -----------------------------------------
                    # Train Generator
                    # -----------------------------------------
                    
                    self.optimizerG.zero_grad()
                    
                    # Generate fake images
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    fake_images = self.generator(z)
                    
                    # Compute loss with fake images
                    outputs = self.critic(fake_images)
                    
                    # Get generator loss and perform backward propagation
                    g_loss = self.generator_loss(outputs)
                    g_loss.backward()
                    # Updates the weights
                    self.optimizerG.step()

                    # Checkpointing
                    self.batch_count += 1
                    if self.batch_count % self.save_interval == 0:
                        self.checkpoint_handler.save(self.generator, self.critic, self.optimizerG, self.optimizerD, self.batch_count)

                    bar.set_description(f"critic loss: {criticloss/cnt}")

                # Print epoch results
                print(f"Epoch [{epoch+1}/{self.num_epochs}] | Critic Loss: {c_loss.item()} | Generator Loss: {g_loss.item()}")

if __name__ == "__main__":
    wgan = Trainer()
    wgan.train()