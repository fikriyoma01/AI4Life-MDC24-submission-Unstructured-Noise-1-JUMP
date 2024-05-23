from pathlib import Path
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x - self.dncnn(x)
        return y

def create_model(model_path: Path):
    """Create and save an example DnCNN model"""
    model = DnCNN()
    example_input = torch.rand(1, 1, 540, 540)
    jit_model = torch.jit.trace(model, example_inputs=example_input)
    print(f'Saving model to: {model_path.absolute()}')
    torch.jit.save(jit_model, model_path)

if __name__ == "__main__":
    model_path = Path(__file__).parent / "resources/model.pth"
    create_model(model_path)

