from datasets import load_dataset
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# loading the dataset
imagenet = load_dataset(
  'Maysee/tiny-imagenet', 
  split='valid',
)

# preprocessing the dataset (converting to RGB and to tensors)
images = []
labels = []

for img in imagenet:
  rbgImg = img['image'].convert('RGB')
  resizedImg = rbgImg.resize((227, 227))
  pixels = np.array(resizedImg).astype(np.float32) / 255.0
  permuted = np.transpose(pixels, (2, 0, 1))
  images.append(permuted)
  labels.append(img['label'])

# creating the nn API
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
g = torch.Generator(device=device).manual_seed(42)

class Conv:
  def __init__(self, num_of_kernels, channels_in, kernel_size, stride, padding):
    self.weight = torch.randn((num_of_kernels, channels_in, kernel_size, kernel_size), generator=g, device=device) * 0.1
    self.stride = stride
    self.padding = padding
  
  def __call__(self, x):
    self.out = torch.nn.functional.conv2d(input=x, weight=self.weight, stride=self.stride, padding=self.padding)
    return self.out
  
  def parameters(self):
    return [self.weight]
  
class Pooling:
  def __init__(self, kernel_size, stride):
    self.kernel_size = kernel_size
    self.stride = stride
  
  def __call__(self, x):
    self.out = torch.nn.functional.max_pool2d(x, self.kernel_size, self.stride, 0)
    return self.out
  
  def parameters(self):
    return []
  
class Flatten:
  def __call__(self, x):
    self.out = x.reshape(x.shape[0], -1)
    return self.out
  
  def parameters(self):
    return []

class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g, device=device) / fan_in**0.5
    self.bias = torch.zeros(fan_out, device=device) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class ReLU:
  def __call__(self, x):
    self.out = torch.relu(x)
    return self.out
  def parameters(self):
    return []
  
# initializing the NN
batch_size = 16
learning_rate = 0.0001
num_of_epochs = 10

layers = [
  Conv(num_of_kernels=96, channels_in=3, kernel_size=11, stride=4, padding=0), ReLU(), Pooling(kernel_size=3, stride=2),
  Conv(num_of_kernels=256, channels_in=96, kernel_size=5, stride=1, padding=2), ReLU(), Pooling(kernel_size=3, stride=2),
  Conv(num_of_kernels=384, channels_in=256, kernel_size=3, stride=1, padding=1), ReLU(),
  Conv(num_of_kernels=384, channels_in=384, kernel_size=3, stride=1, padding=1), ReLU(),
  Conv(num_of_kernels=256, channels_in=384, kernel_size=3, stride=1, padding=1), ReLU(), Pooling(kernel_size=3, stride=2),
  Flatten(),
  Linear(fan_in=9216, fan_out=4096), ReLU(),
  Linear(fan_in=4096, fan_out=4096), ReLU(),
  Linear(fan_in=4096, fan_out=1000), ReLU(),
  Linear(fan_in=1000, fan_out=200),
]

# requiring gradient on all layer's parameters
params = []
for layer in layers:
  params += layer.parameters()

for p in params:
  p.requires_grad = True

def activation_plot(tensor):
  # black = false; white = true
  h_detached = tensor.cpu().detach().view(132, -1)
  plt.imshow(h_detached <= 0, cmap="gray", vmin=0, vmax=1) 
  plt.show()

# training loop
for epoch in range(num_of_epochs):
  # sampling a random batch
  randomIndecies = [random.randint(0, 10000-1) for _ in range(batch_size)]
  image_batch = []
  ans = []
  for i in randomIndecies:
    image_batch.append(images[i])
    ans.append(labels[i])

  image_batch = torch.tensor(image_batch).to(torch.float32).to(device)
  ans = torch.tensor(ans).to(device)

  # forward pass
  def forward(image_batch):
    for layer in layers:
      image_batch = layer(image_batch)

    return image_batch

  # backward pass
  def backward(logits, ans):
    global learning_rate
    loss = torch.nn.functional.cross_entropy(logits, ans)
    print(f"epoch {epoch}, loss = {loss}")

    for p in params:
      p.grad = None

    loss.backward()

    for p in params:
      p.data -= p.grad * learning_rate
  
  logits = forward(image_batch)
  backward(logits, ans)

def save_params():
  for i in range(len(layers)):
    layer = layers[i]
    layer_params = layer.parameters()
    layer_params = [p.detach().cpu() for p in layer_params]
    if layer_params:
      torch.save(layer_params, f"params/layer_{i}.pth")

save_params()