import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'your-bucket-name'
folder_name = 'your-image-folder/'

# Create a directory to store images locally
os.makedirs('./images', exist_ok=True)

# Download images from S3
def download_images_from_s3(bucket, folder):
  response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
  for obj in response['Contents']:
    file_name = obj['Key']
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
      s3.download_file(bucket, file_name, os.path.join('./images', os.path.basename(file_name)))
      print(f'Downloaded {file_name}')

download_images_from_s3(bucket_name, folder_name)

from torchvision import transforms
from PIL import Image
import os

# Define transformations
transform = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
  image = Image.open(image_path)
  return transform(image)

# Apply preprocessing to all images in the directory
image_paths = ['./images/' + img for img in os.listdir('./images')]
preprocessed_images = [load_and_preprocess_image(img) for img in image_paths]

import torch
import torch.nn as nn

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1):
    super(UNet, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.decoder = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid()  # Use Sigmoid for binary segmentation
    )

  def forward(self, x):
    x = self.encoder(x)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = self.decoder(x)
    return x

model = UNet()

import torch.optim as optim

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of a training loop
num_epochs = 10

for epoch in range(num_epochs):
  for images, masks in train_loader:  # Assuming `masks` are ground truth masks
    outputs = model(images)
    loss = criterion(outputs, masks)

        # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Save the trained model locally
model_path = 'unet_model.pth'
torch.save(model.state_dict(), model_path)

# Upload the model to S3
s3.upload_file(model_path, bucket_name, 'models/unet_model.pth')
print(f"Model saved and uploaded to S3: {bucket_name}/models/unet_model.pth")

