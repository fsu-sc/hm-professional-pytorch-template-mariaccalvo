#  5. TensorBoard Analysis

import time
import json
import torch
from model.metric import Metrics
from model.dynamic_model import DenseModel
from torch.utils.tensorboard import SummaryWriter
from data_loader.function_dataset import FunctionsDataset

# function to load configurations
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# 4 different configurations for different experiments
basic_config = load_config("configs/config.json")
optimal_config = load_config("configs/optimal.json")
overfit_config = load_config("configs/overfit.json")
underfit_config = load_config("configs/underfit.json")

# dictionary of configurations
configs = {
    "Basic Configuration": basic_config,
    "Optimal Configuration": optimal_config,
    "Overfit Configuration": overfit_config,
    "Underfit Configuration": underfit_config
}

# run the experiments
for config_name, config in configs.items():
    print(f"Running experiment: {config_name}")

    # initialize writer
    log_dir = f'runs/{config_name.replace(" ", "_")}'  
    writer = SummaryWriter(log_dir)

    # initialize model
    model = DenseModel(
        input_size=1,  
        output_size=1,
        hidden_layers=config["hidden_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        activation_hidden=config["activation_hidden"],
        activation_output=config["activation_output"],
    )

    # initialize dataset and dataLoader
    train_dataset = FunctionsDataset(n_samples=1000, function="linear")
    val_dataset = FunctionsDataset(n_samples=200, function="linear")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # log model 
    dummy_input = torch.ones(1, 1)  
    writer.add_graph(model, dummy_input)

    # hyperparameters
    num_epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    metrics = Metrics()

    # training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # training phase
        model.train()
        train_loss, train_accuracy = 0.0, 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            loss = metrics.calculate_loss(predictions, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += metrics.calculate_accuracy(predictions, y).item()
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        # log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # validation phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                loss = metrics.calculate_validation_loss(predictions, y)
                val_loss += loss.item()
                val_accuracy += metrics.calculate_accuracy(predictions, y).item()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        # log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Time/epoch', time.time() - start_time, epoch)

        # print log
        metrics.log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy)

    # close TensorBoard writer
    writer.close()