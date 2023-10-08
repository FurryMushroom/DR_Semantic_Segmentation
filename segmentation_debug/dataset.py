import torchvision
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
SEGLABE_PATH='C:/Users/1/Desktop/DRseg/dataset/DDR-dataset/lesion-segmentation/label/HE'
IMG_PATH='C:/Users/1/Desktop/DRseg/dataset/DDR-dataset/lesion-segmentation/image'
INPUT_SIZE=512
# this is running on server
class SEGData(Dataset):  # 继承Dataset这个类
    def __init__(self):
        self.img_path = IMG_PATH
        self.label_path = SEGLABE_PATH
        self.label_data = os.listdir(self.label_path)  # os.listdir(path) path:需要列出的目录路径 返回指定路径下的文件和文件夹list。
        self.totensor=torchvision.transforms.ToTensor()
        self.resizer=torchvision.transforms.Resize((INPUT_SIZE,INPUT_SIZE))
        self.resizer1 = torchvision.transforms.Resize((1536, 1536))
        self.tenCropper = torchvision.transforms.TenCrop(1024, vertical_flip=True)
        # 调整PILImage对象的尺寸。PILImage对象size属性返回的是w, h，而resize的参数顺序是h, w。
        # parameter (h,w)，height and width，both int，resize input image to (h,w) size，equals force。
        # if use one int number，resize the shorter edge to the int number，longer edge gets adjusted proportionately。
    def __len__(self):
        return len(self.label_data)
    def __getitem__(self, item):
        # split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
        img_name = os.path.join(self.img_path, self.label_data[item])
        img_name = os.path.split(img_name) # 以 "PATH" 中最后一个 '/' 作为分隔符，分隔后，将索引为0的视为目录（路径），将索引为1的视为文件名
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0] + '.jpg' # label is .tif format,while image is .jpg
        img_data = os.path.join(self.img_path, img_name)
        label_data = os.path.join(self.label_path, self.label_data[item])
        # 将图片和标签都转为正方形
        img = Image.open(img_data) # fp -文件名 (字符串)，pathlib.Path对象或文件对象
        label = Image.open(label_data)
        w, h = img.size
        slide=min(h,w)
        center_cropper=torchvision.transforms.CenterCrop(slide)
        cropped_img = center_cropper(img)
        cropped_label = center_cropper(label)
        res_img=self.resizer1(cropped_img)
        res_label=self.resizer1(cropped_label)
        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in range 0-255.
        aug = iaa.Affine(rotate=(-10, 10))
        im_aug, seg_aug = aug(image=res_img, segmentation_maps=res_label)

        ten_cropped_imgs = list(self.tenCropper(res_img))
        ten_cropped_labels = list(self.tenCropper(res_label))
        for i in range(10):
            ten_cropped_imgs[i]=self.totensor(self.resizer(  ten_cropped_imgs[i]))
        for i in range(10):
            ten_cropped_labels[i] =self.totensor( self.resizer(ten_cropped_labels[i]))
        flipped_img =self.totensor( self.resizer(cropped_img.transpose(Image.FLIP_LEFT_RIGHT) )) # 水平翻转
        flipped_label =self.totensor(self.resizer( cropped_label.transpose(Image.FLIP_LEFT_RIGHT)) ) # 水平翻转
        imgs=[self.totensor(self.resizer(cropped_img))]+ten_cropped_imgs+[flipped_img]
        labels=[self.totensor(self.resizer(cropped_label))]+ten_cropped_labels +[flipped_label]
        img_tensor=torch.stack(tuple(imgs))
        label_tensor = torch.stack(tuple(labels))
        return {
            "imgs":img_tensor,
            "labels":label_tensor
        }
        # imgs = torch.stack((
        #     self.totensor(cropped_img.crop((0, 0, 224, 224))), self.totensor(cropped_img.crop((224, 0, 448, 224))),
        #     self.totensor(cropped_img.crop((448, 0, 672, 224))), self.totensor(cropped_img.crop((672, 0, 896, 224))),
        #     self.totensor(cropped_img.crop((0, 224, 224, 448))), self.totensor(cropped_img.crop((224, 224, 448, 448))),
        #     self.totensor(cropped_img.crop((448, 224, 672, 448))),
        #     self.totensor(cropped_img.crop((672, 224, 896, 448))),
        #     self.totensor(cropped_img.crop((0, 448, 224, 672))), self.totensor(cropped_img.crop((224, 448, 448, 672))),
        #     self.totensor(cropped_img.crop((448, 448, 672, 672))),
        #     self.totensor(cropped_img.crop((672, 448, 896, 672))),
        #     self.totensor(cropped_img.crop((0, 672, 224, 896))), self.totensor(cropped_img.crop((224, 672, 448, 896))),
        #     self.totensor(cropped_img.crop((448, 672, 672, 896))), self.totensor(cropped_img.crop((672, 672, 896, 896)))
        # ), dim=0)
        # labels = torch.stack((
        #     self.totensor(cropped_label.crop((0, 0, 224, 224))), self.totensor(cropped_label.crop((224, 0, 448, 224))),
        #     self.totensor(cropped_label.crop((448, 0, 672, 224))),
        #     self.totensor(cropped_label.crop((672, 0, 896, 224))),
        #     self.totensor(cropped_label.crop((0, 224, 224, 448))),
        #     self.totensor(cropped_label.crop((224, 224, 448, 448))),
        #     self.totensor(cropped_label.crop((448, 224, 672, 448))),
        #     self.totensor(cropped_label.crop((672, 224, 896, 448))),
        #     self.totensor(cropped_label.crop((0, 448, 224, 672))),
        #     self.totensor(cropped_label.crop((224, 448, 448, 672))),
        #     self.totensor(cropped_label.crop((448, 448, 672, 672))),
        #     self.totensor(cropped_label.crop((672, 448, 896, 672))),
        #     self.totensor(cropped_label.crop((0, 672, 224, 896))),
        #     self.totensor(cropped_label.crop((224, 672, 448, 896))),
        #     self.totensor(cropped_label.crop((448, 672, 672, 896))),
        #     self.totensor(cropped_label.crop((672, 672, 896, 896)))
        # ), dim=0)
        #
        # image_np = imgs[0].detach().numpy()
        # i1=np.rollaxis(image_np,0,3)
        #
        # # i2=image_np[1]
        # # i3 = image_np[2]
        # # i4 = image_np[3]
        # plt.imshow(i1)
        # plt.show()
        # plt.imshow(i2)
        # plt.show()
        # plt.imshow(i3)
        # plt.show()
        # plt.imshow(i4)
        # plt.show()
        # return {"image": imgs, "mask": labels}
        # fivecropper=torchvision.transforms.FiveCrop(1200)
        # five_cropped_img=list(fivecropper(cropped_img))
        # five_cropped_label = list(fivecropper(cropped_label))
        # test=img.crop((224,224,448,448))
        # for i in range(5):
        #     five_cropped_img[i] = self.totensor(self.resizer(five_cropped_img[i]))
        #     five_cropped_label[i] = self.totensor(self.resizer(five_cropped_label[i]))
        # imgs=torch.stack((five_cropped_img[0],five_cropped_img[1],five_cropped_img[2],five_cropped_img[3],five_cropped_img[4]),dim=0)
        # labels=torch.stack((five_cropped_label[0],five_cropped_label[1],five_cropped_label[2],five_cropped_label[3],five_cropped_label[4]),dim=0)
        # return{ "img":imgs,"mask":labels}
        # imgs,labels=[],[]

        # for i in range(5):
        #     imgs.append(five_cropped_img[i])
        # #     labels.append(five_cropped_label[i])
        #
        # return imgs,labels
        # 变为tensor,转换为统一大小
        # img = self.resizer(cropped_img)
        # label = self.resizer(cropped_label)
        # label.show()
        # img.show()
        # img = self.totensor(img)
        # label = self.totensor(label)
        # return img,label
