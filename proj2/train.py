from argparse import ArgumentParser
import data_helper
import model_helper
from torchvision import transforms, datasets, models
import torch
from torch import nn, optim
import os

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="directory containing training, validation, and test data; directories and data must be organized in a structure compatible with torchvision models")
    parser.add_argument("--save_dir", help="directory to save checkpoint in")
    parser.add_argument("-g", "--gpu", help="flag to toggle GPU usage for model training", action="store_true", default=False)
    parser.add_argument("-a", "--arch", help="torchvision pretrained model architecture to use, defaults to vgg19", default="vgg19")
    parser.add_argument("-l", "--learning_rate", help="learning rate to use for model training", default=0.06)
    parser.add_argument("-u", "--hidden_units", help="hidden units to use in model classifier" , nargs="+", type=int, default=[4096, 1024, 512])
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10, type=int)
    
    sysargs = parser.parse_args()
    print(sysargs)

    # Make save directory if it doesn't already exist
    if not os.path.isdir(sysargs.save_dir):
        os.mkdir(sysargs.save_dir)
    
    # Create image dataset and dataloaders
    image_datasets, dataloaders = data_helper.create_dataloaders(sysargs.data_dir)
        
    # Load specified model
    output_size = 102
    model = model_helper.load_pretrained_model(sysargs.arch, sysargs.hidden_units, output_size)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=sysargs.learning_rate)
    
    # Enable CUDA if available
    device = torch.device("cuda:0" if (sysargs.gpu and torch.cuda.is_available()) else "cpu")
        
    print_every = 40
    step = 0
    train_error = 0
    
    model.to(device)
    model.train()

    for e in range(sysargs.epochs):
        for images, labels in dataloaders[0]:
            step += 1
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Compute loss and error
            loss = criterion(outputs, labels)
            train_error += loss.item()
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            if step % print_every == 0:
                with torch.no_grad():
                    validation_loss, acc = model_helper.validation(model, criterion, dataloaders[1], device)
                    print("Epoch {}/{}:".format(e+1, sysargs.epochs),
                      "Training error: {0:.4f}".format(train_error / print_every),
                      "Validation error: {0:.4f}".format(validation_loss),
                      "Validation accuracy: {0:.4f}".format(acc))
            
                train_error = 0
    
    print("------ MODEL TRAINING COMPLETE ------")
    print("Testing network on test set...")
    
    # Test model on test set
    loss, acc = model_helper.validation(model, criterion, dataloaders[2], device)
    print("Loss: {0:.4f} Accuracy: {1:.2f}%".format(loss, acc * 100))
    print("Saving checkpoint...")
    
    checkpoint = {
        'model_arch': sysargs.arch,
        'input_size': model_helper.load_model_helper(sysargs.arch)[1],
        'hidden_sizes': sysargs.hidden_units,
        'output_size': output_size,
        'state_dict': model.state_dict(),
        'num_epochs': sysargs.epochs,
        'optimizer_state': optimizer.state_dict(),
        'mapping': image_datasets[0].class_to_idx}

    torch.save(checkpoint, sysargs.save_dir + "/checkpoint.pth")