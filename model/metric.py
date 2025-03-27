import torch

class Metrics:
    @staticmethod
    def calculate_loss(predictions, targets):
        loss = torch.mean((predictions - targets) ** 2)
        return loss

    @staticmethod
    def calculate_accuracy(predictions, targets):
        correct = torch.abs(predictions - targets) < 0.05  
        accuracy = torch.mean(correct.float()) 
        return accuracy

    @staticmethod
    def calculate_validation_loss(predictions, targets):
        return Metrics.calculate_loss(predictions, targets)

    @staticmethod
    def log_metrics(epoch, train_loss, val_loss, train_accuracy=None, val_accuracy=None):
        if train_accuracy is not None and val_accuracy is not None:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, "
                  f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}")
        else:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
