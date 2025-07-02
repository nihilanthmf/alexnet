import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
from datasets import load_dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# listing the paths for all images
# all_files = [f for f in os.listdir("images")][:1000]
def sort_key(filename):
    name = filename.rsplit('.', 1)[0]
    cls, idx = map(int, name.split('.'))
    return (cls, idx)

all_files = sorted(os.listdir("images"), key=sort_key)[:100000]

# extracting indices from filenames
index_to_file = {}

def process_image(img):
  img = img.convert('RGB').resize((227, 227))
  pixels = np.array(img).astype(np.float32) / 255.0
  image = np.transpose(pixels, (2, 0, 1))

  return image

# function to load the images on the fly (convert to RGB and to tensors)
def load_image(index):
  img_path = all_files[index]
  label, index_str = img_path.split('.', 1)

  img = Image.open(f"images/{img_path}")
  image = process_image(img)

  return image, int(label)

def load_eval_images():
  test_split = load_dataset(
    'Maysee/tiny-imagenet', 
    split="valid"
  )

  return test_split

# creating the nn API
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
g = torch.Generator(device=device).manual_seed(42)
evaluation_mode = False

class Conv:
  def __init__(self, num_of_kernels, channels_in, kernel_size, stride, padding, weights_file_name):
    self.weight = torch.load(f"inference_params/{weights_file_name}.pth")[0].to(device) if evaluation_mode else torch.randn((num_of_kernels, channels_in, kernel_size, kernel_size), generator=g, device=device) * (1.0 / (channels_in * kernel_size ** 2) ** 0.5)
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
  def __init__(self, fan_in, fan_out, weights_file_name, bias=True):
    self.weight =  torch.load(f"inference_params/{weights_file_name}.pth")[0].to(device) if evaluation_mode else torch.randn((fan_in, fan_out), generator=g, device=device) / fan_in**0.5
    self.bias = torch.load(f"inference_params/{weights_file_name}.pth")[1].to(device) if evaluation_mode else torch.zeros(fan_out, device=device) if bias else None
  
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
batch_size = 2
learning_rate = 1e-2
learning_rate_decay = 1e-3
learning_rate_decay_2 = 1e-4
num_of_epochs = 100_000

layers = [
  Conv(num_of_kernels=96, channels_in=3, kernel_size=11, stride=4, padding=0, weights_file_name="layer_0"), ReLU(), Pooling(kernel_size=3, stride=2),
  Conv(num_of_kernels=256, channels_in=96, kernel_size=5, stride=1, padding=2, weights_file_name="layer_3"), ReLU(), Pooling(kernel_size=3, stride=2),
  Conv(num_of_kernels=384, channels_in=256, kernel_size=3, stride=1, padding=1, weights_file_name="layer_6"), ReLU(),
  Conv(num_of_kernels=384, channels_in=384, kernel_size=3, stride=1, padding=1, weights_file_name="layer_8"), ReLU(),
  Conv(num_of_kernels=256, channels_in=384, kernel_size=3, stride=1, padding=1, weights_file_name="layer_10"), ReLU(), Pooling(kernel_size=3, stride=2),
  Flatten(),
  Linear(fan_in=9216, fan_out=4096, weights_file_name="layer_14"), ReLU(),
  Linear(fan_in=4096, fan_out=4096, weights_file_name="layer_16"), ReLU(),
  Linear(fan_in=4096, fan_out=1000, weights_file_name="layer_18"), ReLU(),
  Linear(fan_in=1000, fan_out=200, weights_file_name="layer_20"),
]

# requiring gradient on all layer's parameters
params = []
for layer in layers:
  params += layer.parameters()

for p in params:
  p.requires_grad = True

# forward pass
def forward(image_batch):
  for layer in layers:
    image_batch = layer(image_batch)

  return image_batch

training_loss_valuse = []
eval_loss_values = []
eval_images = load_eval_images()[:5000]

current_training_loss_values=[]
current_eval_loss_values=[]

def get_eval_result(indecies, imgs):
  image_batch = []
  ans = []
  for i in indecies:
    image = imgs['image'][i]
    label = imgs['label'][i]

    image = process_image(image)

    image_batch.append(image)
    ans.append(label)

  image_batch = torch.tensor(np.array(image_batch)).to(device)
  ans = torch.tensor(np.array(ans)).to(device)

  logits = forward(image_batch)

  return logits, image_batch, ans

if (evaluation_mode):
  test_images = load_eval_images()[:1000]

  logits, image_batch, ans = get_eval_result(range(len(test_images['image'])), test_images)

  loss = torch.nn.functional.cross_entropy(logits, ans)
  print("eval loss =", loss)

  preds = torch.argmax(logits, dim=1)
  correct = (preds == ans).sum()
  accuracy = correct.item() / len(ans) * 100
  print("eval accuracy =", accuracy)

else:
  # training loop
  for epoch in range(num_of_epochs):
    # sampling a random batch
    randomIndecies = [random.randint(0, len(all_files) - 1) for _ in range(batch_size)]
    image_batch = []
    ans = []
    for i in randomIndecies:
      image, label = load_image(i)
      image_batch.append(image)
      ans.append(label)

    image_batch = torch.tensor(image_batch).to(device)
    ans = torch.tensor(ans).to(device)

    # backward pass
    def backward(loss):
      global learning_rate

      for p in params:
        p.grad = None

      loss.backward()

      if epoch > 20_000:
        learning_rate = learning_rate_decay
      if epoch > 70_000:
        learning_rate = learning_rate_decay_2

      for p in params:
        p.data -= p.grad * learning_rate

    logits = forward(image_batch)

    loss = torch.nn.functional.cross_entropy(logits, ans)
    print(f"epoch {epoch}, loss = {loss}")

    if epoch % 200 == 0:
      training_loss_valuse.append(np.mean(current_training_loss_values))
      eval_loss_values.append(np.mean(current_eval_loss_values))
    else:
      current_training_loss_values.append(loss.item())

      randomIndecies = [random.randint(0, len(eval_images["image"]) - 1) for _ in range(batch_size)]
      logits, _, ans = get_eval_result(randomIndecies, eval_images)

      eval_loss = torch.nn.functional.cross_entropy(logits, ans)

      current_eval_loss_values.append(eval_loss.item())

    backward(loss)

  # Create plot
  plt.plot(list(range(len(training_loss_valuse))), training_loss_valuse, label='Training loss', marker='o')
  plt.plot(list(range(len(eval_loss_values))), eval_loss_values, label='Eval loss', marker='x')

  # Add labels and title
  plt.xlabel('Time')
  plt.ylabel('Value')
  plt.title('Sensor Data Over Time')
  plt.legend()
  plt.grid(True)

  # Show plot
  plt.show()

  def save_params():
    for i in range(len(layers)):
      layer = layers[i]
      layer_params = layer.parameters()
      layer_params = [p.detach().cpu() for p in layer_params]
      if layer_params:
        torch.save(layer_params, f"params/layer_{i}.pth")

  save_params()