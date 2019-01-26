from argparse import ArgumentParser
import model_helper, data_helper
import torch
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image_path", help="path to image to use as prediction input")
    parser.add_argument("checkpoint", help="path to predictive model checkpoint file (.pth) to use")
    parser.add_argument("--top_k", help="return top K most likely classes", type=int, default=5)
    parser.add_argument("--category_names", help="use a mapping of categories to real names: argument takes path to JSON file containing a dict mapping")
    parser.add_argument("-g", "--gpu", help="flag to toggle GPU usage for model training", action="store_true", default=False)
    
    sysargs = parser.parse_args()
    print(sysargs)
    
    # Load model checkpoint
    checkpoint = torch.load(sysargs.checkpoint)
    
    # Use checkpoint to load model
    num_epochs, model, optimizer = model_helper.load_checkpoint(checkpoint)
    
    # Enable CUDA if available
    device = torch.device("cuda:0" if (sysargs.gpu and torch.cuda.is_available()) else "cpu")
    
    # Make predictions
    probs, classes = model_helper.predict(sysargs.image_path, model, device, sysargs.top_k)
    
    # Load category names if provided, print top K predictions
    print("Top {} predictions:".format(sysargs.top_k))
    print("----------")
    
    if sysargs.category_names:
        with open(sysargs.category_names, 'r') as f:
            cat_to_name = json.load(f)
        flower_names = data_helper.get_flower_names(classes, cat_to_name)
        to_print = [ pred for pred in zip(probs, classes, flower_names) ]
        for k in range(sysargs.top_k):
            print("{0}. Class: {1:4} Name: {2:20} Probability: {3:.2f}%".format(k+1, to_print[k][1], to_print[k][2], to_print[k][0] * 100))
    else:
        to_print = [ pred for pred in zip(probs, classes)]
        for k in range(sysargs.top_k):
            print("{0}. Class: {1:4} Probability: {2:.2f}%".format(k+1, to_print[k][1], to_print[k][0] * 100))
    


