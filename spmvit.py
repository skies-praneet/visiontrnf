import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import SimpleViT
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

# Define a class for Vision Transformer Interpretability
class ViTInterpretability:
    def __init__(self, image_size, patch_size, num_classes):
        self.model = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024
        )

    def load_image(self, image_path):
        """Load and preprocess an image."""
        img = Image.open(image_path).convert('RGB')
        transform = ToTensor()
        return transform(img).unsqueeze(0)  # Add batch dimension

    def visualize_attention_maps(self, image_tensor, layer_idxs=[1, 3, 5], head_idxs=[0, 2, 4, 6]):
        """Visualize attention maps across specified layers and heads."""
        self.model.eval()
        with torch.no_grad():
            attention_maps = self.model.transformer.get_attention_map(image_tensor)

        fig, axes = plt.subplots(len(layer_idxs), len(head_idxs), figsize=(15, 10))
        fig.suptitle("Attention Maps Across Layers and Heads", fontsize=16)

        for i, layer_idx in enumerate(layer_idxs):
            for j, head_idx in enumerate(head_idxs):
                attention = attention_maps[layer_idx][0, head_idx].cpu().numpy()
                axes[i, j].imshow(attention, cmap='viridis')
                axes[i, j].set_title(f'Layer {layer_idx}, Head {head_idx}')
                axes[i, j].axis('off')
        plt.show()

    def integrated_gradients(self, image_tensor, target_class, steps=50):
        """Compute Integrated Gradients for feature attribution."""
        baseline = torch.zeros_like(image_tensor)
        scaled_images = [baseline + float(i) / steps * (image_tensor - baseline) for i in range(steps + 1)]
        scaled_images = torch.cat(scaled_images, dim=0)

        scaled_images.requires_grad = True
        outputs = self.model(scaled_images)
        target_scores = outputs[:, target_class].sum()
        grads = grad(target_scores, scaled_images)[0]
        avg_grads = grads.mean(dim=0)

        integrated_grad = (image_tensor - baseline) * avg_grads
        integrated_grad = integrated_grad.squeeze().sum(dim=0)
        return integrated_grad.detach().cpu().numpy()

    def visualize_integrated_gradients(self, image_tensor, integrated_grad):
        """Overlay Integrated Gradients on the original image."""
        plt.figure(figsize=(8, 8))
        plt.imshow(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.imshow(integrated_grad, cmap='hot', alpha=0.6)
        plt.title("Integrated Gradients")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def analyze_neuron_activations(self, image_tensor):
        """Analyze and visualize neuron activation distributions across layers."""
        self.model.eval()
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        hooks = []
        for idx, layer in enumerate(self.model.transformer.layers):
            hook = layer.register_forward_hook(get_activation(f"Layer_{idx}"))
            hooks.append(hook)

        self.model(image_tensor)

        fig, axes = plt.subplots(len(activations), 2, figsize=(12, 12))
        fig.suptitle("Neuron Activation Distributions Across Layers", fontsize=16)

        for idx, layer_name in enumerate(activations.keys()):
            activation = activations[layer_name].cpu().numpy().flatten()
            ax = axes[idx // 2, idx % 2]
            ax.hist(activation, bins=50, color='purple', alpha=0.7)
            ax.set_title(layer_name)
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Frequency")

        for hook in hooks:
            hook.remove()

        plt.tight_layout()
        plt.show()

    def generate_saliency_map(self, image_tensor, target_class):
        """Generate saliency map to show pixel sensitivity."""
        image_tensor.requires_grad_()
        self.model.zero_grad()
        output = self.model(image_tensor)
        output[0, target_class].backward()
        saliency = image_tensor.grad.abs().squeeze().sum(dim=0).cpu().numpy()
        return saliency

    def visualize_saliency_map(self, image_tensor, saliency):
        """Visualize the saliency map as a heatmap overlay."""
        plt.figure(figsize=(8, 8))
        plt.imshow(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.imshow(saliency, cmap='hot', alpha=0.6)
        plt.title("Saliency Map")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def analyze_layer_gradients(self, image_tensor, target_class):
        """Analyze layer-wise influence using gradient flow."""
        gradients = {}

        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                gradients[name] = grad_output[0].detach().cpu().mean().item()
            return hook

        hooks = []
        for idx, layer in enumerate(self.model.transformer.layers):
            hook = layer.register_backward_hook(get_gradient(f"Layer_{idx}"))
            hooks.append(hook)

        self.model.zero_grad()
        output = self.model(image_tensor)
        output[0, target_class].backward()

        plt.figure(figsize=(10, 5))
        layer_grad_values = [gradients[f"Layer_{i}"] for i in range(len(self.model.transformer.layers))]
        plt.plot(layer_grad_values, marker='o', color='blue')
        plt.xlabel("Layer")
        plt.ylabel("Average Gradient Magnitude")
        plt.title("Layer-wise Influence by Gradient Flow")
        plt.grid()
        plt.show()

        for hook in hooks:
            hook.remove()

# Main analysis workflow
if __name__ == "__main__":
    # Configuration
    IMAGE_SIZE = 256
    PATCH_SIZE = 32
    NUM_CLASSES = 1000
    IMAGE_PATH = 'sample_image.jpg'  # Path to your image file
    TARGET_CLASS = 0  # Specify the target class for analysis

    # Initialize the interpretability analysis
    vit_interpreter = ViTInterpretability(IMAGE_SIZE, PATCH_SIZE, NUM_CLASSES)
    
    # Load and preprocess the image
    image_tensor = vit_interpreter.load_image(IMAGE_PATH)
    
    # 1. Visualize Attention Maps Across Heads and Layers
    vit_interpreter.visualize_attention_maps(image_tensor)

    # 2. Calculate and visualize Integrated Gradients
    integrated_grad = vit_interpreter.integrated_gradients(image_tensor, TARGET_CLASS)
    vit_interpreter.visualize_integrated_gradients(image_tensor, integrated_grad)

    # 3. Analyze Neuron Activation Distributions Across Layers
    vit_interpreter.analyze_neuron_activations(image_tensor)

    # 4. Generate and visualize the Saliency Map
    saliency = vit_interpreter.generate_saliency_map(image_tensor, TARGET_CLASS)
    vit_interpreter.visualize_saliency_map(image_tensor, saliency)

    # 5. Analyze Layer-wise Influence via Gradient Flow
    vit_interpreter.analyze_layer_gradients(image_tensor, TARGET_CLASS)

    # Additional Analysis: Layer-wise Attention Patterns
    def visualize_layer_attention_patterns(model, image_tensor, num_layers=6):
        """Visualizes attention patterns across specified number of layers."""
        model.eval()
        with torch.no_grad():
            attention_patterns = model.transformer.get_attention_map(image_tensor)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(20, 5))
        for i in range(num_layers):
            attention = attention_patterns[i][0, 0].cpu().numpy()  # First head
            axes[i].imshow(attention, cmap='viridis')
            axes[i].set_title(f'Layer {i+1}')
            axes[i].axis('off')
        plt.suptitle('Attention Patterns Across Layers', fontsize=20)
        plt.show()

    visualize_layer_attention_patterns(vit_interpreter.model, image_tensor)

    # Additional Functionality: Track Weight Histograms
    def visualize_weight_histograms(model):
        """Visualizes the histograms of weights for each layer."""
        plt.figure(figsize=(15, 10))
        for i, layer in enumerate(model.transformer.layers):
            weights = layer.to_q.weight.data.cpu().numpy()  # Weights of the query layer
            plt.subplot(2, 3, i + 1)
            plt.hist(weights.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title(f'Layer {i+1} Weights Histogram')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    visualize_weight_histograms(vit_interpreter.model)

    # Additional Analysis: Monitor Layer Outputs
    def visualize_layer_outputs(model, image_tensor):
        """Visualizes the output of each layer for a given image."""
        model.eval()
        outputs = []

        def get_output(name):
            def hook(model, input, output):
                outputs.append(output.detach())
            return hook

        hooks = []
        for idx, layer in enumerate(model.transformer.layers):
            hook = layer.register_forward_hook(get_output(f"Layer_{idx}"))
            hooks.append(hook)

        model(image_tensor)

        fig, axes = plt.subplots(len(outputs), 1, figsize=(10, len(outputs) * 2))
        for i, output in enumerate(outputs):
            axes[i].imshow(output[0].cpu().numpy(), cmap='viridis')
            axes[i].set_title(f'Layer {i + 1} Output')
            axes[i].axis('off')

        for hook in hooks:
            hook.remove()

        plt.tight_layout()
        plt.show()

    visualize_layer_outputs(vit_interpreter.model, image_tensor)
