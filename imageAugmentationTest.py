# Image augmentation test and download image to local folder
import torch
import torchvision
import torchvision.transforms as transforms
# import PIL.Image
import os

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  transforms.Resize((244,244)),
  transforms.ColorJitter(brightness=(0.2,1.5),contrast=(0.1,2.5),hue=.05, saturation=(.0,.15)),

  # transforms.RandomAffine(0, translate=(0.0,0.0), scale=None, shear=(0.0,0.0), interpolation=PIL.Image.NEAREST, fill=(0,0,0)),
  transforms.RandomAffine(0, translate=(0.0,0.0), scale=None, shear=(0.0,0.0), fill=(0,0,0)),
  transforms.ToTensor()
])

Images = torchvision.datasets.ImageFolder(root='./testImage', transform=transform)

# dataloader = torch.utils.data.DataLoader(Images, batch_size=4, shuffle=True)
dataloader = torch.utils.data.DataLoader(Images, batch_size=4, shuffle=False)
# Data loader의 parameter로 num_workers=2를 주면, 에러가 뜬다. (아마 cpu 밖에 없는 컴퓨터에 2를 줘서 그런 듯)
# Shuffle = True로 주면 순서가 계속 바뀌니까 보기 쉽게 할 때는 snuffle=False로 조정

# Define a directory to save the augmented images
save_dir = 'augmented_images'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)


for imgNum in range(5):
  # Iterate over the dataloader
  for i, data in enumerate(dataloader):
    print(len(data))
    # Get the images and labels from the batch
    images, labels = data
    print('images')

    # Save the images to disk
    for j in range(images.size(0)):
        image = images[j]
        filename = f'{save_dir}/image_{i * 5 + j}_{imgNum}.jpg'
        print(filename)
        # print(j)
        # print(type(image));
        torchvision.utils.save_image(image, filename)