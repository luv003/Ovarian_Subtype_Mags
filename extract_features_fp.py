import torch
import torch.nn as nn
import os
import time
import h5py
import openslide
import timm
import argparse

#from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet18_baseline,resnet50_baseline
from utils.utils import collate_features
from utils.file_utils import save_hdf5
from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import eval_transforms

import torch
from torchvision import transforms
import torchstain


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("torch device:", device, "\n")


from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

import random
from random import randrange

def eval_transforms(pretrained=False):
        if pretrained:
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)

        else:
                mean = (0.5,0.5,0.5)
                std = (0.5,0.5,0.5)

        trnsfrms_val = transforms.Compose(
                                        [
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = mean, std = std)
                                        ]
                                )

        return trnsfrms_val

class Whole_Slide_Bag(Dataset):
        def __init__(self,
                file_path,
                pretrained=False,
                custom_transforms=None,
                target_patch_size=-1,
                ):
                """
                Args:
                        file_path (string): Path to the .h5 file containing patched data.
                        pretrained (bool): Use ImageNet transforms
                        custom_transforms (callable, optional): Optional transform to be applied on a sample
                """
                self.pretrained=pretrained
                if target_patch_size > 0:
                        self.target_patch_size = (target_patch_size, target_patch_size)
                else:
                        self.target_patch_size = None

                if not custom_transforms:
                        self.roi_transforms = eval_transforms(pretrained=pretrained)
                else:
                        self.roi_transforms = custom_transforms

                self.file_path = file_path

                with h5py.File(self.file_path, "r") as f:
                        dset = f['imgs']
                        self.length = len(dset)

                self.summary()
                        
        def __len__(self):
                return self.length

        def summary(self):
                hdf5_file = h5py.File(self.file_path, "r")
                dset = hdf5_file['imgs']
                for name, value in dset.attrs.items():
                        print(name, value)

                print('pretrained:', self.pretrained)
                print('transformations:', self.roi_transforms)
                if self.target_patch_size is not None:
                        print('target_size: ', self.target_patch_size)

        def __getitem__(self, idx):
                with h5py.File(self.file_path,'r') as hdf5_file:
                        img = hdf5_file['imgs'][idx]
                        coord = hdf5_file['coords'][idx]
                
                img = Image.fromarray(img)
                if self.target_patch_size is not None:
                        img = img.resize(self.target_patch_size)
                img = self.roi_transforms(img).unsqueeze(0)
                return img, coord

class Whole_Slide_Bag_FP(Dataset):
        def __init__(self,
                file_path,
                wsi,
                pretrained=False,
                custom_transforms=None,
                custom_downsample=None,
                target_patch_size=-1,
                selected_idxs=None,
                max_patches_per_slide=None,
                model_architecture=None,
                batch_size=None,
                extract_features=False
                ):
                """
                Args:
                        file_path (string): Path to the .h5 file containing patched data.
                        pretrained (bool): Use ImageNet transforms
                        custom_transforms (callable, optional): Optional transform to be applied on a sample
                        custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
                        target_patch_size (int): Custom defined image size before embedding
                """
                self.pretrained = pretrained
                self.wsi = wsi
                self.max_patches_per_slide = max_patches_per_slide
                if not custom_transforms:
                        self.roi_transforms = eval_transforms(pretrained=pretrained)
                else:
                        self.roi_transforms = custom_transforms

                self.file_path = file_path
                self.extract_features = extract_features
                #print("file path:",self.file_path)
                with h5py.File(self.file_path, "r") as f:
                        self.coords=f['coords'][:len(f['coords'])]
                        if selected_idxs is None:
                            self.selected_coords=self.coords
                            #print(self.selected_coords)
                            #print(self.selected_coords[0])
                        else:
                            self.selected_coords = self.coords[sorted(list(set(selected_idxs)))]
                        #print("max patches per slide",self.max_patches_per_slide)
                        #print(self.selected_coords)
                        #print(self.selected_coords.dtype)
                        #if self.max_patches_per_slide:
                            #if self.max_patches_per_slide<len(self.selected_coords):
                                #self.selected_coords = random.sample(self.selected_coords,self.max_patches_per_slide)
                                #sample_keys = random.sample(list(self.selected_coords.keys()), self.max_patches_per_slide)
                                #self.selected_coords = {key: self.selected_coords[key] for key in sample_keys}
                                
                                ## below was working but turned off
                                #sample_idxs = random.sample(range(len(self.selected_coords)),self.max_patches_per_slide)
                        #        sample_idxs = np.random.choice(len(self.selected_coords),self.max_patches_per_slide)
                        #        self.selected_coords = torch.tensor(self.selected_coords[sorted(list(set(sample_idxs)))])
                        
                        #print("len selected_coords",len(self.selected_coords))
                        #print("selected coords:",self.selected_coords)
                        #print(self.selected_coords)
                        self.patch_level = f['coords'].attrs['patch_level']
                        self.patch_size = f['coords'].attrs['patch_size']
                        self.length = len(self.selected_coords)
                        if target_patch_size > 0:
                                self.target_patch_size = (target_patch_size, ) * 2
                        elif custom_downsample > 1:
                                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
                        else:
                                self.target_patch_size = None
        
        def __len__(self):
                return self.length

        def summary(self):
                hdf5_file = h5py.File(self.file_path, "r")
                dset = hdf5_file['coords']
                for name, value in dset.attrs.items():
                        print(name, value)

                print('\nfeature extraction settings')
                print('target patch size: ', self.target_patch_size)
                print('pretrained: ', self.pretrained)
                print('transformations: ', self.roi_transforms)
        
        def update_sample(self,selected_idxs):
                #print("updating sample to length ",len(selected_idxs))
                #with h5py.File(self.file_path, "r") as f:
                #print("self.coords",self.coords)
                self.selected_coords=self.coords[sorted(list(set(selected_idxs)))]
                #print("selected coords in update sample",self.selected_coords)
                #print("selected_coords.shape in update_sample",self.selected_coords.shape)
                #print("selected coords",self.selected_coords)
                self.length = len(self.selected_coords)
                #print("self.length",self.length)
        
        def coords(self,num):
                with h5py.File(self.file_path,'r') as hdf5_file:
                    coords = hdf5_file['coords'][:num]
                return coords

        def __getitem__(self, idx):
                coord=self.selected_coords[idx]
                #print("coord entered into read_region",(coord))
                #print("selected_coords before read_region",self.selected_coords)
                
                #print("transforms:",self.roi_transforms)
                img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
                if self.target_patch_size is not None:
                        img = img.resize(self.target_patch_size)
                transform = transforms.Compose([transforms.ToTensor()])
                #print("before transforms",transform(img))
                img = self.roi_transforms(img).unsqueeze(0)
                #print("after transforms",img)
                return img, coord

class Dataset_All_Bags(Dataset):

        def __init__(self, csv_path):
                self.df = pd.read_csv(csv_path)
        
        def __len__(self):
                return len(self.df)

        def __getitem__(self, idx):
                return self.df['slide_id'][idx]



def compute_w_loader(file_path, output_path, wsi, model,
        batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
        custom_downsample=2, target_patch_size=-1):
        """
        args:
                file_path: directory of bag (.h5 file)
                output_path: directory to save computed features (.h5 file)
                model: pytorch model
                batch_size: batch_size for computing features in batches
                verbose: level of feedback
                pretrained: use weights pretrained on imagenet
                custom_downsample: custom defined downscale factor of image patches
                target_patch_size: custom defined, rescaled image size before embedding
        """
        
        if args.use_transforms=='macenko':
            class MacenkoNormalisation:
                def __init__(self):
                    self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
                    self.failures=0

                def __call__(self,image):
                    try:
                        norm, _, _ = self.normalizer.normalize(I=image, stains=False)
                        norm = norm.permute(2, 0, 1)/255
                    except:
                        norm=image/255
                        self.failures=self.failures+1
                        print("failed patches: ",self.failures)
            t = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Lambda(lambda x: x*255),
                MacenkoNormalisation()])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)


        elif args.use_transforms=='all':
            t = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=90,translate=(0.1,0.1), scale=(0.9,1.1),shear=0.1),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        
        elif args.use_transforms=='spatial':
            t = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=90,translate=(0.1,0.1), scale=(0.9,1.1),shear=0.1),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        
        elif args.use_transforms=='HIPT':
            t = eval_transforms()
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        
        elif args.use_transforms=='HIPT_blur':
            t =  transforms.Compose(
                    [transforms.GaussianBlur(kernel_size=(1, 3), sigma=(7, 9)),
                    eval_transforms()
                    ])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)

        elif args.use_transforms=='HIPT_wang':
        ## augmentations from the baseline ATEC23 paper
            t = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=90),
                    transforms.ColorJitter(brightness=0.125, contrast=0.2, saturation=0.2),
                    eval_transforms()])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)

        elif args.use_transforms=='HIPT_augment_colour':
            ## same as HIPT_augment but no affine
            t = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    eval_transforms()])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        
        elif args.use_transforms=='HIPT_augment':
            t = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=5,translate=(0.025,0.025), scale=(0.975,1.025),shear=0.025),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    eval_transforms()])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        
        elif args.use_transforms=='HIPT_augment01':
            t = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=5,translate=(0.025,0.025), scale=(0.975,1.025),shear=0.025),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    eval_transforms()])
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, custom_transforms=t, pretrained=pretrained,
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)

        else:
            dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
                custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        dataset.update_sample(range(len(dataset)))
        x, y = dataset[0]
        
        if args.model_type=='resnet18':
            kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        elif args.model_type=='resnet50':
            kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        elif args.model_type=='levit_128s':
            kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
            tfms=torch.nn.Sequential(transforms.CenterCrop(224))
        elif args.model_type=='HIPT_4K':
            if args.hardware=='DGX':
                kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
            else:
                kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

        if verbose > 0:
                print('processing {}: total of {} batches'.format(file_path,len(loader)))

        mode = 'w'
        for count, (batch, coords) in enumerate(loader):
                with torch.no_grad():   
                        if count % print_every == 0:
                                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
                        batch = batch.to(device, non_blocking=True)
                        if args.model_type=='levit_128s':
                            batch=tfms(batch)
                        features = model(batch)
                        features = features.cpu().numpy()

                        asset_dict = {'features': features, 'coords': coords}
                        save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                        mode = 'a'
        
        return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--pretraining_dataset',type=str,choices=['ImageNet','Histo'],default='ImageNet')
parser.add_argument('--model_type',type=str,choices=['resnet18','resnet50','levit_128s','HIPT_4K'],default='resnet50')
parser.add_argument('--use_transforms',type=str,choices=['all','HIPT','HIPT_blur','HIPT_augment','HIPT_augment_colour','HIPT_wang','HIPT_augment01','spatial','macenko','none'],default='none')
parser.add_argument('--hardware',type=str,default="PC")
parser.add_argument('--graph_patches',type=str,choices=['none','small','big'],default='none')
args = parser.parse_args()


if __name__ == '__main__':

        print('initializing dataset')
        csv_path = args.csv_path
        if csv_path is None:
                raise NotImplementedError

        bags_dataset = Dataset_All_Bags(csv_path)
        
        os.makedirs(args.feat_dir, exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
        dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
        
        print('loading {} model'.format(args.model_type))
        if args.model_type=='resnet18':
            model = resnet18_baseline(pretrained=True,dataset=args.pretraining_dataset)
        elif args.model_type=='resnet50':
            model = resnet50_baseline(pretrained=True,dataset=args.pretraining_dataset)
        elif args.model_type=='levit_128s':
            model=timm.create_model('levit_256',pretrained=True, num_classes=0)    
        elif args.model_type=='HIPT_4K':
            if args.hardware=='DGX':
                 model = HIPT_4K(model256_path="/mnt/results/Checkpoints/vit256_small_dino.pth",model4k_path="/mnt/results/Checkpoints/vit4k_xs_dino.pth",device256=torch.device('cuda:0'),device4k=torch.device('cuda:0'))
            else:
                model = HIPT_4K(model256_path="HIPT_4K/ckpts/vit256_small_dino.pth",model4k_path="HIPT_4K/ckpts/vit4k_xs_dino.pth",device256=torch.device('cuda:0'),device4k=torch.device('cuda:0'))
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                
        model.eval()
        total = len(bags_dataset)

        unavailable_patch_files=0
        total_time_elapsed = 0.0
        for bag_candidate_idx in range(total):
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print('skipped unavailable slides: {}'.format(unavailable_patch_files))
            try:        
                slide_id = str(bags_dataset[bag_candidate_idx]).split(args.slide_ext)[0]
                bag_name = slide_id+'.h5'
                if args.graph_patches == 'big':
                    h5_file_path = os.path.join(args.data_h5_dir,'patches/big',bag_name)
                elif args.graph_patches == 'small':
                    h5_file_path = os.path.join(args.data_h5_dir,'patches/small',bag_name)
                else:
                    h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
                slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
                print(slide_id)

                if args.use_transforms == 'all':
                    if not args.no_auto_skip and slide_id+'aug1.pt' in dest_files:
                        print('skipped {}'.format(slide_id))
                        continue
                else:
                    if not args.no_auto_skip and slide_id+'.pt' in dest_files:
                        print('skipped {}'.format(slide_id))
                        continue 

                output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
                time_start = time.time()
                wsi = openslide.open_slide(slide_file_path)
                output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
                model = model, batch_size = args.batch_size, verbose = 1, print_every = 100, 
                custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
                time_elapsed = time.time() - time_start
                total_time_elapsed += time_elapsed
                print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
                file = h5py.File(output_file_path, "r")

                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)
                features = torch.from_numpy(features)
                bag_base, _ = os.path.splitext(bag_name)
                torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
            except KeyboardInterrupt:
                assert 1==2, "keyboard interrupt"
            except:
                print("patch file unavailable")
                continue
        print("finished running with {} unavailable slide patch files".format(unavailable_patch_files))
        print("total time: {}".format(total_time_elapsed))
