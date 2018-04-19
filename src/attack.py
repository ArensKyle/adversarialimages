import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.models
import json

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import *


def image_loader(image_name):
    # Size the pretrained network needs
    image_size = (224, 224)
    # Scale image
    loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Open image and prepare it
    img = Image.open(image_name)
    img = img.convert("RGB")  # Auto remove alpha channel
    img = loader(img).float()
    img = normalize(img).float()
    img = img.unsqueeze(0)  # this is needed for VGG16
    return img


def tensor_to_image(tens):
    img = tens[0]
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = np.moveaxis(img, 0, 2)
    return img


def get_imagenet_mapping():
    with open("../data/imagenet_labels.json", "r") as f:
        return json.loads(f.read())


def get_model():
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.eval()
    return vgg16


cross_entr_loss = nn.CrossEntropyLoss()
# Get the image
image_location = "../data/images/test.jpg"
image = Variable(image_loader(image_location), requires_grad=True)
orig_image = Variable(image_loader(image_location), requires_grad=True)
# Get the pretrained model for imagenet
model = get_model()
# Our predictions.
out = model.forward(image)
orig_pred = Variable(torch.LongTensor(np.array([out.data.numpy().argmax()])), requires_grad=False)

# This is a frilled lizard, Chlamydosaurus kingi now.
targeted_class = 43

for _ in tqdm(range(10)):
    prediction = model.forward(image)
    loss = cross_entr_loss(prediction, Variable(torch.LongTensor(np.array([targeted_class]))))
    loss.backward()

    # Add perturbation
    eta = 0.02
    x_grad = torch.sign(image.grad.data)
    adv_x = eta * x_grad
    image.data = image.data - adv_x

# Check adversarilized output
mapping = get_imagenet_mapping()

pred_adversarial = mapping[str(np.argmax(model.forward(image).data.numpy()))]
original_prediction = mapping[str(int(orig_pred.data.numpy()))]

print("Original Prediction of test.jpg: ")
print("\tPred Label:", original_prediction)
print("\tConfidence:", np.max(F.softmax(out, dim=-1).data.numpy()))
print("Adversarial Prediction: ")
print("\tPred Label:", pred_adversarial)
print("\tConfidence:", np.max(F.softmax(model.forward(image), dim=-1).data.numpy()))

# Save our images so we can compare!
torchvision.utils.save_image(image.data, filename="attack.jpg")
torchvision.utils.save_image(orig_image.data, filename="orig.jpg")
