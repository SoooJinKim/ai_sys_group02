import torch.utils.data as data
from PIL import Image
import data.util as util
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import dlib
from skimage.draw import polygon

face_det = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("C:/Users/Serin Kim/workspace/AISYS/DMFN-master/triangle_coordinates/shape_predictor_68_face_landmarks.dat")

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist

def triangle_coordinates(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (400, 400))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_det(image_gray)
    
    triangle_coords = []

    if len(faces) != 0:
        for face in faces:
            # 랜드마크 검출
            lm = landmark_model(image_rgb, face)
            lm_point = np.array([[p.x, p.y] for p in lm.parts()])

        landmark_list = lm_point

        num = 1
        for i in range(14):
            triangle = [
                (landmark_list[29][0], landmark_list[29][1]),
                (landmark_list[num][0], landmark_list[num][1]),
                (landmark_list[num+1][0], landmark_list[num+1][1])
            ]
            triangle_coords.append(triangle)
            num += 1

        return triangle_coords
    else:
        print('\nFace landmarks are not recognizable\n')
        return None

class ImageFilelist(data.Dataset):
    def __init__(self, opt, flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(opt['image_list'])
        self.loader = loader
        self.opt = opt

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # [0, 1] --> [-1, 1]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(impath)
        img = self.resize(img, self.opt['img_shape'][2], self.opt['img_shape'][1])
        img_tensor = self.transform(img)  # Tensor [C, H, W], [-1, 1]
        
        # 마스크 생성
        mask_tensor = self.create_mask_tensor(impath)
        bbox = util.bbox(self.opt)
        bbox_tensor = torch.from_numpy(np.array(bbox))

        # generate mask, 1 represents masked point
        # mask: mask region {0, 1}
        # x_incomplete: incomplete image, [-1, 1]
        # returns: [-1, 1] as predicted image
        input_tensor = img_tensor * (1. - mask_tensor)  # [C, H, W]
        return {'input': input_tensor, 'bbox': bbox_tensor, 'mask': mask_tensor, 'target': img_tensor, 'paths': impath}

    def __len__(self):
        return len(self.imlist)

    def create_mask_tensor(self, image_path):
        triangle_coords = triangle_coordinates(image_path)
        if triangle_coords is None:
            raise ValueError('Face landmarks are not recognizable')

        # Create mask image
        mask_image = np.zeros((self.opt['img_shape'][1], self.opt['img_shape'][2]), dtype=np.float32)
        for triangle in triangle_coords:
            rr, cc = polygon(np.array([p[1] for p in triangle]), np.array([p[0] for p in triangle]), mask_image.shape)
            mask_image[rr, cc] = 1

        # Convert mask image to tensor
        mask_tensor = torch.from_numpy(np.expand_dims(mask_image, 0))  # Tensor, [1, H, W]
        return mask_tensor

    def resize(self, img, height, width, centerCrop=False):  # mainly for celeba dataset | place365 | paris_streetview
        imgw, imgh = img.size[0], img.size[1]  # [w, h, c]

        if imgh != imgw:
            if centerCrop:
                # center crop, mainly for celeba
                side = np.minimum(imgh, imgw)
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img.crop((i, j, side, side))
            else:
                # random crop, mainly for place365 and paris_streetview
                side = np.minimum(imgh, imgw)
                ix = random.randrange(0, imgw - side + 1)
                iy = random.randrange(0, imgh - side + 1)
                img = img.crop((ix, iy, side, side))

        img = img.resize((width, height), Image.BICUBIC)
        return img
