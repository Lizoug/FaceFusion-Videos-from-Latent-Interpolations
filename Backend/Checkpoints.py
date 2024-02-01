import torch


class Checkpoint:
    def __init__(self, path):
        self.path = path

    def save(self, generator, critic, optimizerG, optimizerD, batch_count):
        """Save the model's state."""
        torch.save({
            "generator_state": generator.state_dict(),
            "critic_state": critic.state_dict(),
            "optimG_state": optimizerG.state_dict(),
            "optimD_state": optimizerD.state_dict(),
            "batch_count": batch_count
        }, self.path)

    def load(self):
        """Load the model's state if exists and return it."""
        try:
            chkpt = torch.load(self.path)
            return chkpt["generator_state"], chkpt["critic_state"], chkpt["optimG_state"], chkpt["optimD_state"], chkpt["batch_count"]
        except FileNotFoundError:
            print("Could not find checkpoint, starting from scratch")
            return None, None, None, None, 0
