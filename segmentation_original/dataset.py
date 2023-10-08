import torchvision
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
SEGLABE_PATH='C:/Users/1/Desktop/DRsegmentation/dataset/DDR-dataset/lesion_segmentation/train/label/HE'
IMG_PATH='C:/Users/1/Desktop/DRsegmentation/dataset/DDR-dataset/lesion_segmentation/train/image'
INPUT_SIZE=512
class SEGData(Dataset):  # 继承Dataset这个类
    def __init__(self):
        self.img_path = IMG_PATH
        self.label_path = SEGLABE_PATH
        self.label_data = os.listdir(self.label_path)  # os.listdir(path) path:需要列出的目录路径 返回指定路径下的文件和文件夹list。
        # f = open(label_path, "r", encoding='utf-8')
        # self.label_data=f.read().splitlines()
        # f.close()
        # splitlines() 按照行(’\r’, ‘\r\n’, \n’)分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。
        # f = open(img_path, "r", encoding='utf-8')
        # self.img_data = f.read().splitlines()
        # f.close()
        self.totensor=torchvision.transforms.ToTensor()
        self.resizer=torchvision.transforms.Resize((INPUT_SIZE,INPUT_SIZE))
        # 调整PILImage对象的尺寸。PILImage对象size属性返回的是w, h，而resize的参数顺序是h, w。
        # parameter (h,w)，height and width，both int，resize input image to (h,w) size，equals force。
        # if use one int number，resize the shorter edge to the int number，longer edge gets adjusted proportionately。
    def __len__(self):
        return len(self.label_data)
    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放到同样尺度
        否则dataloader会报错。
        '''
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
        # 以最长边为基准，生成全0正方形矩阵
        slide = max(h, w)
        black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_label = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_img.paste(img, (0, 0, int(w), int(h)))  # paste在图中央和在左上角是一样的
        black_label.paste(label, (0, 0, int(w), int(h)))
        # 变为tensor,转换为统一大小
        img = self.resizer(black_img)
        label = self.resizer(black_label)
        img = self.totensor(img)
        label = self.totensor(label)
        return img,label
