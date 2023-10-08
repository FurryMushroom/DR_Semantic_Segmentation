import imageio
import numpy as np
import os
import random
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt

DATASET_PATH = 'C:/Users/1/Desktop/DRsegmentation/dataset/DDR-dataset/lesion_segmentation/train'
SAVE_PATH = "C:/Users/1/Desktop/DataAug/AugmentedData"

# img_path="C:/Users/1/Desktop/DataAug"
# EX_image = imageio.v3.imread(img_path+"/1.jpg")
# HE_image = imageio.v3.imread(img_path+"/2.jpg")
# SE_image = imageio.v3.imread(img_path+"/3.jpg")
# MA_image = imageio.v3.imread(img_path+"/4.jpg")
# img=np.zeros((512,512,3),dtype='uint8')
# img[:,:,0]=EX_image
# img[:,:,1]=HE_image | MA_image
# img[:,:,2]=SE_image | MA_image
# imageio.imwrite(img_path+"/result.jpg",img)


img_dir = DATASET_PATH + '/image'
label_dir_HE = DATASET_PATH + '/label/HE'
label_dir_MA = DATASET_PATH + '/label/MA'
label_dir_SE = DATASET_PATH + '/label/SE'
label_dir_EX = DATASET_PATH + '/label/EX'
label_names = os.listdir(label_dir_HE)
SIZE1,SIZE2,SIZE3,SIZE4=1536,1024,768,512

# test=imageio.v3.imread("C:/Users/1/Desktop/DataAug/AugmentedData/label/HE/007-1774-100_0.jpg")

for item in range(len(label_names)):
    label_name=label_names[item]
    img_name = label_name.split('.')
    img_name = img_name[0] + '.jpg'  # label is .tif format,while image is .jpg
    img_path = os.path.join(img_dir, img_name)
    label_path_HE = os.path.join(label_dir_HE, label_names[item])
    label_path_EX = os.path.join(label_dir_EX, label_names[item])
    label_path_MA = os.path.join(label_dir_MA, label_names[item])
    label_path_SE = os.path.join(label_dir_SE, label_names[item])
    image = imageio.v3.imread(img_path)
    label_HE= imageio.v3.imread(label_path_HE)
    label_EX = imageio.v3.imread(label_path_EX)
    label_MA = imageio.v3.imread(label_path_MA)
    label_SE = imageio.v3.imread(label_path_SE)
    label=np.expand_dims(
            np.dstack((label_EX,label_SE,label_MA,label_HE))
        ,axis=0)

    h,w=image.shape[0],image.shape[1]
    slide=min(h,w)
    if h>=w:
        area=(-int(h/2-slide/2),-0,-int(h/2-slide/2),0)
        centerCropper = iaa.CropAndPad(area,keep_size=False) # top right bottom left means remove how much
    else:
        area=(0, -int(w / 2 - slide / 2), -0, -int(w / 2 - slide / 2))
        centerCropper = iaa.CropAndPad( area ,keep_size=False) # top right bottom left

    square_img,square_label=centerCropper(image=image,segmentation_maps=label)
    SIZE = 512

    def generate(size,image_nums,ori_img,ori_label,nums_already):
        CURRENT_SIZE = size
        resizer = iaa.Resize(CURRENT_SIZE)
        total_img_num=nums_already
        square_img, square_label = resizer(image=ori_img, segmentation_maps=ori_label)
        for i in range(image_nums):
            rotator = iaa.Affine(rotate=random.randint(0, 360))
            n1, n2 = random.randint(0, CURRENT_SIZE - SIZE), random.randint(0, CURRENT_SIZE - SIZE)
            img_cut = square_img[n1:SIZE + n1,n2:SIZE + n2]
            label_cut = np.expand_dims(square_label[0][n1:SIZE + n1,n2:SIZE + n2],axis=0)
            rot_img, rot_label = rotator(image=img_cut, segmentation_maps=label_cut)
            if random.random() < 0.3:
                rot_img = iaa.imgcorruptlike.apply_gaussian_noise(rot_img, severity=1)
            if random.random() < 0.3:
                rot_img = iaa.imgcorruptlike.apply_brightness(rot_img, severity=random.randint(1, 3))
            if random.random() < 0.6:
                elastic = iaa.ElasticTransformation(alpha=(0, 140), sigma=(8, 12))
                rot_img, rot_label = elastic(image=rot_img, segmentation_maps=rot_label)
            imageio.imsave(SAVE_PATH + "/image/" + label_name.split('.')[0] + '_{}.jpg'.format(total_img_num), rot_img)
            imageio.imwrite(
                SAVE_PATH + "/label/EX/" + label_name.split('.')[0] + '_{}.jpg'.format(total_img_num),
                rot_label[0][:,:,0])
            imageio.imwrite(
                SAVE_PATH + "/label/SE/" + label_name.split('.')[0] + '_{}.jpg'.format(total_img_num),
                rot_label[0][:, :, 0])
            imageio.imwrite(
                SAVE_PATH + "/label/MA/" + label_name.split('.')[0] + '_{}.jpg'.format(total_img_num),
                rot_label[0][:, :, 0])
            imageio.imwrite(
                SAVE_PATH + "/label/HE/" + label_name.split('.')[0] + '_{}.jpg'.format(total_img_num),
                rot_label[0][:, :, 0])
            total_img_num = total_img_num + 1


    generate(slide,320,square_img,square_label,0)
    generate(SIZE1, 180, square_img, square_label,320)
    generate(SIZE2, 80, square_img, square_label,500)
    generate(SIZE3, 40, square_img, square_label,580)
    generate(SIZE4, 20, square_img, square_label,620)
