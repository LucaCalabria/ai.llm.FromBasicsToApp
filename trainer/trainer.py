import torch

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, xb, yb):
        # Set the model to training mode
        self.model.train()
        
        # Forward pass
        logits, loss = self.model(xb, yb)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, xb, yb):
        # Set the model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            logits, loss = self.model(xb, yb)
        
        return loss.item()