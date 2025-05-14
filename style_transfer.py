# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and preprocess an image
def image_loader(image_path, max_size=400):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Resize the image while keeping the aspect ratio
    size = max_size if max(image.size) > max_size else max(image.size)
    
    # Define transformation: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformation and add a batch dimension
    image = transform(image).unsqueeze(0)
    
    # Send the image to GPU or CPU
    return image.to(device)

# Function to calculate the Gram matrix (used in style comparison)
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)  # Flatten the spatial dimensions
    return torch.mm(tensor, tensor.t())  # Matrix multiplication to compute Gram matrix

# Function to extract features from specific layers of the model
def get_features(image, model, layers):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
            if name in layers:
                features[name] = x
    return features

# Function to calculate total loss (content + style loss)
def total_loss(content_weight, style_weight, content_features, style_features, generated_features, style_layers):
    # Content loss compares the content of generated and content images
    content_loss = torch.mean((generated_features['conv_4'] - content_features['conv_4']) ** 2)
    
    # Style loss compares the style using Gram matrices
    style_loss = 0
    for layer in style_layers:
        G = gram_matrix(generated_features[layer])
        A = gram_matrix(style_features[layer])
        style_loss += torch.mean((G - A) ** 2)
    
    # Combine both losses
    return content_weight * content_loss + style_weight * style_loss

# Function to convert tensor to image format for display/save
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)  # Remove batch dimension
    image = image.permute(1, 2, 0)  # Rearrange dimensions for display
    image = image.numpy()
    image = image.clip(0, 1)  # Clamp values between 0 and 1
    return image

# Main function that performs neural style transfer
def run_style_transfer(content_path, style_path):
    # Load content and style images
    content = image_loader(content_path)
    style = image_loader(style_path)

    # Clone the content image to start generating from it
    generated = content.clone().requires_grad_(True)

    # Load the pretrained VGG19 model for feature extraction
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    # Specify which layers to use for content and style features
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Get features from the content and style images
    content_features = get_features(content, vgg, content_layers)
    style_features = get_features(style, vgg, style_layers)

    # Set up optimizer to update the generated image
    optimizer = optim.Adam([generated], lr=0.003)

    # Run the optimization for 300 steps
    for step in range(300):
        optimizer.zero_grad()
        # Get features from the current generated image
        generated_features = get_features(generated, vgg, style_layers + content_layers)
        # Calculate loss
        loss = total_loss(1, 1e6, content_features, style_features, generated_features, style_layers)
        # Backpropagation and update
        loss.backward(retain_graph=True)
        optimizer.step()
        # Print loss every 50 steps
        if step % 50 == 0:
            print(f"Step {step}, Loss {loss.item()}")

    # Convert final tensor to image
    final_img = im_convert(generated)

    # Save the image
    output_path = "static/stylized_output.jpg"
    plt.imsave(output_path, final_img)
    return output_path
