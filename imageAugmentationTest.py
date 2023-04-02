# Image augmentation test and download image to local folder
import torch
import torchvision
import torchvision.transforms as transforms
# import PIL.Image
import os

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  transforms.Resize((502, 322)),
  transforms.ColorJitter(brightness=(0.6,1.3),contrast=(0.9,2.1),hue=.05, saturation=(.0,.15)),
  # transforms.RandomAffine(0, translate=(0.0,0.0), scale=None, shear=(0.0,0.0), interpolation=PIL.Image.NEAREST, fill=(0,0,0)),
  transforms.RandomAffine(0, translate=(0.0,0.0), scale=None, shear=(0.0,0.0), fill=(0,0,0)),
  transforms.ToTensor()
])

Images = torchvision.datasets.ImageFolder(root='./originalImage', transform=transform)

# dataloader = torch.utils.data.DataLoader(Images, batch_size=4, shuffle=True)
dataloader = torch.utils.data.DataLoader(Images, batch_size=1, shuffle=False)
# Data loader의 parameter로 num_workers=2를 주면, 에러가 뜬다. (아마 cpu 밖에 없는 컴퓨터에 2를 줘서 그런 듯)
# Shuffle = True로 주면 순서가 계속 바뀌니까 보기 쉽게 할 때는 snuffle=False로 조정

# Define a directory to save the augmented images
save_dir = 'augmented_images'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)


for imgNum in range(3):
  # Iterate over the dataloader

  for i, data in enumerate(dataloader):
    file_paths = Images.imgs[i][0].split('\\')[-1].split('.')[0]
    print(file_paths)

    # Get the images and labels from the batch
    image, label = data
    # print('images')

    # Save the images to disk
    # if imgNum < 3:
    #   filename = f'{save_dir}/trainImage/{file_paths}_{imgNum}.jpg'
    if imgNum == 0:
      filename = f'{save_dir}/trainImage_1/{file_paths}/{imgNum}.jpg'
    elif imgNum == 1:
      filename = f'{save_dir}/trainImage_2/{file_paths}/{imgNum}.jpg'
    elif imgNum == 2:
      filename = f'{save_dir}/trainImage_3/{file_paths}/{imgNum}.jpg'
    else:
      filename = f'{save_dir}/testImage/{file_paths}/{imgNum}.jpg'
    print(filename)
    # print(j)
    # print(type(image));
    pathName_1 = f'{save_dir}/trainImage_1/{file_paths}'
    pathName_2 = f'{save_dir}/trainImage_2/{file_paths}'
    pathName_3 = f'{save_dir}/trainImage_3/{file_paths}'
    if not os.path.exists(pathName_1):
      os.makedirs(pathName_1)
    if not os.path.exists(pathName_2):
      os.makedirs(pathName_2)
    if not os.path.exists(pathName_3):
      os.makedirs(pathName_3)

    torchvision.utils.save_image(image, filename)