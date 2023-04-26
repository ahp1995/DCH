import torch
import numpy as np
import torchvision
import cv2
import os
from torchvision.utils import save_image
from utils import save_cifar10_png_images, save_cifar100_png_images

def mk_cifar10_dog(data_path, dst_train, num_classes):
    
    # separate image indices
    labels_all = []
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    indices_class = [[] for c in range(num_classes)]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    # check cifar10 png images
    if not os.path.exists(data_path + '/cifar10_image'):
        save_cifar10_png_images(data_path)
    
    # make dir for ImageFolder
    if not os.path.exists(data_path + '/cifar10_dogf'):
        os.mkdir(data_path + '/cifar10_dogf')
        for i in range(10):
            os.mkdir(data_path + '/cifar10_dogf/{}'.format(i))
        
    # Make cifar10 DoG filtering image   
    for i in range(50000):
        img = cv2.imread(data_path + '/cifar10_image/img{}.png'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # DoG mask
        gaussian1 = cv2.GaussianBlur(gray, (5, 5), 1.6)
        gaussian2 = cv2.GaussianBlur(gray, (5, 5), 1)
        
        DoG = np.zeros_like(gray)
        for a in range(height):
            for b in range(width):
                DoG[a][b] = float(gaussian1[a][b]) - float(gaussian2[a][b])
        
        DoG1 = np.expand_dims(DoG, axis=0)
        DoG2 = np.expand_dims(DoG, axis=0)
        DoG3 = np.expand_dims(DoG, axis=0)
        
        # 3 mask to 1 image put it in each channel
        DoG_img = np.concatenate((DoG1, DoG2, DoG3), axis=0)
        DoG_img = np.transpose(DoG_img, (1, 2, 0)) # [32, 32, 3]
        
        for j in range(10):
            if i in indices_class[j]:
                cv2.imwrite(data_path + '/cifar10_dogf/{0}/img{1}.png'.format(j, i), DoG_img)
                
                

def mk_cifar100_dog(data_path, dst_train, num_classes):
    
    # separate image indices
    labels_all = []
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    indices_class = [[] for c in range(num_classes)]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    # check cifar100 png images
    if not os.path.exists(data_path + '/cifar100_image'):
        save_cifar100_png_images(data_path)
    
    # make dir for ImageFolder
    if not os.path.exists(data_path + '/cifar100_dogf'):
        os.mkdir(data_path + '/cifar100_dogf')
        for i in range(100):
            os.mkdir(data_path + '/cifar100_dogf/{}'.format(i))
        
    # Make cifar100 DoG filtering image   
    for i in range(50000):
        img = cv2.imread(data_path + '/cifar100_image/img{}.png'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # DoG mask
        gaussian1 = cv2.GaussianBlur(gray, (5, 5), 1.6)
        gaussian2 = cv2.GaussianBlur(gray, (5, 5), 1)
        
        DoG = np.zeros_like(gray)
        for a in range(height):
            for b in range(width):
                DoG[a][b] = float(gaussian1[a][b]) - float(gaussian2[a][b])
        
        DoG1 = np.expand_dims(DoG, axis=0)
        DoG2 = np.expand_dims(DoG, axis=0)
        DoG3 = np.expand_dims(DoG, axis=0)
        
        # 3 mask to 1 image put it in each channel
        DoG_img = np.concatenate((DoG1, DoG2, DoG3), axis=0)
        DoG_img = np.transpose(DoG_img, (1, 2, 0)) # [32, 32, 3]
        
        for j in range(100):
            if i in indices_class[j]:
                cv2.imwrite(data_path + '/cifar100_dogf/{0}/img{1}.png'.format(j, i), DoG_img)
                
                

def mk_cifar10_lf(data_path, dst_train, num_classes):
    
    # separate image indices
    labels_all = []
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    indices_class = [[] for c in range(num_classes)]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    # check cifar10 png images
    if not os.path.exists(data_path + '/cifar10_image'):
        save_cifar10_png_images(data_path)
    
    # make dir for ImageFolder
    if not os.path.exists(data_path + '/cifar10_lf'):
        os.mkdir(data_path + '/cifar10_lf')
        for i in range(10):
            os.mkdir(data_path + '/cifar10_lf/{}'.format(i))
    
    # Make cifar10 Laplacian filtering image   
    for i in range(50000):
        img = cv2.imread(data_path + '/cifar10_image/img{}.png'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # mask
        mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # 3 laplacian mask
        laplacian1 = cv2.filter2D(gray, -1, mask1)
        laplacian2 = cv2.filter2D(gray, -1, mask2)
        laplacian3 = cv2.filter2D(gray, -1, mask3)
        
        laplacian1 = np.expand_dims(laplacian1, axis=0)
        laplacian2 = np.expand_dims(laplacian2, axis=0)
        laplacian3 = np.expand_dims(laplacian3, axis=0)
        
        # 3 mask to 1 image put it in each channel
        laplacian_img = np.concatenate((laplacian1, laplacian2, laplacian3), axis=0)
        laplacian_img = np.transpose(laplacian_img, (1, 2, 0)) # [32, 32, 3]
        
        for j in range(10):
            if i in indices_class[j]:
                cv2.imwrite(data_path + '/cifar10_lf/{0}/img{1}.png'.format(j, i), laplacian_img)
                
                
                
def mk_cifar100_lf(data_path, dst_train, num_classes):
    
    # separate image indices
    labels_all = []
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    indices_class = [[] for c in range(num_classes)]

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    # check cifar100 png images
    if not os.path.exists(data_path + '/cifar100_image'):
        save_cifar100_png_images(data_path)
    
    # make dir for ImageFolder
    if not os.path.exists(data_path + '/cifar100_lf'):
        os.mkdir(data_path + '/cifar100_lf')
        for i in range(100):
            os.mkdir(data_path + '/cifar100_lf/{}'.format(i))
    
    # Make cifar100 Laplacian filtering image   
    for i in range(50000):
        img = cv2.imread(data_path + '/cifar100_image/img{}.png'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # mask
        mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # 3 laplacian mask
        laplacian1 = cv2.filter2D(gray, -1, mask1)
        laplacian2 = cv2.filter2D(gray, -1, mask2)
        laplacian3 = cv2.filter2D(gray, -1, mask3)
        
        laplacian1 = np.expand_dims(laplacian1, axis=0)
        laplacian2 = np.expand_dims(laplacian2, axis=0)
        laplacian3 = np.expand_dims(laplacian3, axis=0)
        
        # 3 mask to 1 image put it in each channel
        laplacian_img = np.concatenate((laplacian1, laplacian2, laplacian3), axis=0)
        laplacian_img = np.transpose(laplacian_img, (1, 2, 0)) # [32, 32, 3]
        
        for j in range(100):
            if i in indices_class[j]:
                cv2.imwrite(data_path + '/cifar100_lf/{0}/img{1}.png'.format(j, i), laplacian_img)