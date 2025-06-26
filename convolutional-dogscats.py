import torch
import random
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import json
from PIL import Image
import numpy as np

# setting training/inference mode
toTrain = False
inferenceImages = ["./cnn-dogs-cats/parrot.jpg", "./cnn-dogs-cats/prnk.jpeg", "./cnn-dogs-cats/prianik.jpeg", "./cnn-dogs-cats/kitty.jpg"]

# inferenceImages = []
# for i in range(1, 64):
#     inferenceImages.append(f"./dogs-vs-cats-redux-kernels-edition/train/cat.{12000+i}.jpg")
#     inferenceImages.append(f"./dogs-vs-cats-redux-kernels-edition/train/dog.{12000+i}.jpg")

# model setup
dir = "./dogs-vs-cats-redux-kernels-edition/train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing the network's params
g = torch.Generator(device=device).manual_seed(42)

batch_size = 16 # will be 2x that cause both dogs and cats
convo_w_kernels = 128

weights_scale = 0.05

convo_w = torch.randn((convo_w_kernels, 3, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w2 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w3 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w4 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device)* weights_scale

convo_w5 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w6 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w7 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale
convo_w8 = torch.randn((convo_w_kernels, convo_w_kernels, 3, 3), generator=g, dtype=torch.float32, device=device) * weights_scale

wh = torch.randn((8192, 10), generator=g, dtype=torch.float32, device=device) * weights_scale
bh = torch.zeros((10,), device=device)

wo = torch.randn((10, 2), generator=g, dtype=torch.float32, device=device) * weights_scale
bo = torch.zeros((2,), device=device)

convo_w = convo_w.to(device)
wh = wh.to(device)
bh = bh.to(device)
wo = wo.to(device)
bo = bo.to(device)

params = [convo_w, convo_w2, convo_w3, convo_w4, convo_w5, convo_w6, convo_w7, convo_w8, wh, bh, wo, bo]

for p in params:
    p.requires_grad=True

loss_values=[]

def saveParams():
    with open(f"./weights/convo_w.txt", "w") as f:
        json.dump(convo_w.tolist(), f)
    with open(f"./weights/convo_w2.txt", "w") as f:
        json.dump(convo_w2.tolist(), f)
    with open(f"./weights/convo_w3.txt", "w") as f:
        json.dump(convo_w3.tolist(), f)
    with open(f"./weights/convo_w4.txt", "w") as f:
        json.dump(convo_w4.tolist(), f)
    with open(f"./weights/convo_w5.txt", "w") as f:
        json.dump(convo_w5.tolist(), f)
    with open(f"./weights/convo_w6.txt", "w") as f:
        json.dump(convo_w6.tolist(), f)
    with open(f"./weights/convo_w7.txt", "w") as f:
        json.dump(convo_w7.tolist(), f)
    with open(f"./weights/convo_w8.txt", "w") as f:
        json.dump(convo_w8.tolist(), f)
    with open(f"./weights/wh.txt", "w") as f:
        json.dump(wh.tolist(), f)
    with open(f"./weights/bh.txt", "w") as f:
        json.dump(bh.tolist(), f)
    with open(f"./weights/wo.txt", "w") as f:
        json.dump(wo.tolist(), f)
    with open(f"./weights/bo.txt", "w") as f:
        json.dump(bo.tolist(), f)

def convolute(img_tensor:torch.tensor, kernel):
    output = F.conv2d(input=img_tensor, weight=kernel, padding=1)
    
    output = output.squeeze(1)
    
    return output

def model(img):
    img_convo = convolute(img, convo_w)
    img_activations = F.relu(img_convo)
    img_convo_2 = convolute(img_activations, convo_w2)
    img_activations_2 = F.relu(img_convo_2)

    img_pooling = F.max_pool2d(img_activations_2, kernel_size=2, stride=2)

    img_convo_3 = convolute(img_pooling, convo_w3)
    img_activations_3 = F.relu(img_convo_3)
    img_convo_4 = convolute(img_activations_3, convo_w4)
    img_activations_4 = F.relu(img_convo_4)

    img_pooling_2 = F.max_pool2d(img_activations_4, kernel_size=2, stride=2)

    img_convo_5 = convolute(img_pooling_2, convo_w5)
    img_activations_5 = F.relu(img_convo_5)
    img_convo_6 = convolute(img_activations_5, convo_w6)
    img_activations_6 = F.relu(img_convo_6)

    img_pooling_3 = F.max_pool2d(img_activations_6, kernel_size=2, stride=2)

    img_convo_7 = convolute(img_pooling_3, convo_w7)
    img_activations_7 = F.relu(img_convo_7)
    img_convo_8 = convolute(img_activations_7, convo_w8)
    img_activations_8 = F.relu(img_convo_8)

    img_pooling_3 = F.max_pool2d(img_activations_8, kernel_size=2, stride=2)

    flat = img_pooling_3.reshape(batch_size*2, -1)
    h = torch.tanh(flat @ wh + bh)
    
    logits = h @ wo + bo

    return logits

def readImage(imgPath):
    img = Image.open(imgPath)
    img = img.resize((128, 128))
        
    pixels = np.array(img)
    return pixels

def readImageBatch(indecies):
    images = []
    ans = []

    # 0 = cat; 1 = dog
    def readImg(cat, i):
        pixels = readImage(os.path.join(dir, f"cat.{i}.jpg" if cat else f"dog.{i}.jpg"))
        
        images.append(pixels)
        ans.append(0 if cat else 1)

    for index in indecies:
        readImg(True, index)
        readImg(False, index)

    return [images, ans]

def train():
    global batch_size
    for i in range(50000):
        randomIndecies = [random.randint(1, 11500) for _ in range(batch_size)]

        readImagesRaw = readImageBatch(randomIndecies)
        # img = torch.tensor(readImagesRaw[0], device=device).float().permute(0, 3, 1, 2) # that shit was 10x slower for some reason
        img = torch.from_numpy(np.array(readImagesRaw[0])).float().permute(0, 3, 1, 2).to(device) / 255.0 # that shit 10x speed for some fucking reason
        ans = torch.tensor(readImagesRaw[1], device=device).to(device)

        logits = model(img)

        # calc loss
        loss = F.cross_entropy(logits, ans)

        # applying L2 regularization
        loss += 0.00003 * sum(p.pow(2).sum() for p in params)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == ans).float().mean().item()
        print(f"{i} epoch loss: {loss.item()}, acc: {acc}")

        for p in params:
            p.grad = None

        loss.backward()

        print(convo_w.grad.norm())
        print(wh.grad.norm())
        
        learningAlpha = 0.025

        if i > 5000:
            learningAlpha = 0.01

        for p in params:
            p.data -= learningAlpha * p.grad
    saveParams()
    
    batch_size = 100
    getTestLoss()

def getTestLoss():
    global batch_size
    batch_size = 64
    # indecies = range(11500, 12500)
    
    for i in range(11516, 12500, 64):
        readImagesRaw = readImageBatch(range(i-64, i))

        # img = torch.tensor(readImagesRaw[0], device=device).float().permute(0, 3, 1, 2) # that shit was 10x slower for some reason
        img = torch.from_numpy(np.array(readImagesRaw[0])).float().permute(0, 3, 1, 2).to(device) / 255.0 # that shit 10x speed for some fucking reason
        ans = torch.tensor(readImagesRaw[1], device=device).to(device)

        logits = model(img)

        # calc loss
        loss = F.cross_entropy(logits, ans)

        # applying L2 regularization
        loss += 0.000015 * sum(p.pow(2).sum() for p in params)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == ans).float().mean().item()
        print(f"Test loss: {loss.item()}, test acc: {acc}")

def inference(imagesPaths):
    images = []
    for path in imagesPaths:
        images.append(readImage(path))
    
    img = torch.from_numpy(np.array(images)).float().permute(0, 3, 1, 2).to(device) / 255.0

    logits = model(img)

    def softmax(logits):
        counts = logits.exp()
        prob = ((counts / counts.sum(1, keepdim=True)) * 1000).round() / 1000

        return prob

    probs = softmax(logits).tolist()

    for i in range(len(probs)):
        p = probs[i]
        print("cat" if p[0]>p[1] else "dog;", f"confidence: {max(p)};", imagesPaths[i])

# training loop
if (toTrain):
    train()
else:
    batch_size = len(inferenceImages)//2 # this is very questionable

    with open(f"./weights/convo_w.txt", "r") as f:
        convo_w = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w2.txt", "r") as f:
        convo_w2 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w3.txt", "r") as f:
        convo_w3 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w4.txt", "r") as f:
        convo_w4 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w5.txt", "r") as f:
        convo_w5 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w6.txt", "r") as f:
        convo_w6 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w7.txt", "r") as f:
        convo_w7 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/convo_w8.txt", "r") as f:
        convo_w8 = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/wh.txt", "r") as f:
        wh = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/bh.txt", "r") as f:
        bh = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/wo.txt", "r") as f:
        wo = torch.tensor(json.loads(f.read()), requires_grad=True)
    with open(f"./weights/bo.txt", "r") as f:
        bo = torch.tensor(json.loads(f.read()), requires_grad=True)

    inference(inferenceImages)
