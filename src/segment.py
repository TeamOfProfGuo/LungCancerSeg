import os
import cv2
import sys
import pdb
import nrrd
import re,random,time,sys,scipy

import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn

from tqdm import tqdm

# import files
from models.UNet import UNet2D
from dataset.generator import ProcessedDataGenerator
from dataset.lung_extract_funcs import parse_dataset, resize_3d_img, max_connected_volume_extraction


def generate_segmentation(img, params, verbosity, model, thr=0.99):
    if verbosity:
        print('Segmentation started')
        st = time.time()

    # img: [S, H, W] -> [S, 1, H, W]
    img = torch.from_numpy(img).cuda().float().unsqueeze(1)
    B = 10

    for i in range((img.shape[0]//B)+1):
        with torch.no_grad():
            if i == (img.shape[0]//B):  # the last batch
                prediction= model(img[i*B:])
            else:
                prediction= model(img[i*B:(i+1)*B])

        # convert to binary prediction
        prediction = 1*(prediction.cpu().numpy()>thr)

        # concate the prediction
        if i == 0:
            total_pred = prediction
        else:
            total_pred = np.concatenate((total_pred, prediction), axis=0)
        
    total_pred = np.squeeze(total_pred)  # [S, H, W]

    if verbosity:
        print('Segmentation is finished')
        print('time spent: %s sec.'%(time.time()-st))
    predicted_arr_temp = max_connected_volume_extraction(total_pred)
    temporary_mask = np.zeros(params['normalized_shape'],np.uint8)

    if params['crop_type']:
        temporary_mask[params['z_st']:params['z_end'],...] = predicted_arr_temp[:,params['xy_st']:params['xy_end'],params['xy_st']:params['xy_end']]
    else:
        temporary_mask[params['z_st']:params['z_end'],params['xy_st']:params['xy_end'],params['xy_st']:params['xy_end']] = predicted_arr_temp

    if temporary_mask.shape != params['original_shape']:
        predicted_array  = np.array(1*(resize_3d_img(temporary_mask,params['original_shape'],cv2.INTER_NEAREST)>0.5),np.int8)
    else:
        predicted_array  = np.array(temporary_mask,np.int8)

    return predicted_array


def segment(data_path, model, output_path, verbosity):
    # need gpu to run preprocessing
    Patient_dict = parse_dataset(data_path, img_only=True)

    Patients_gen = ProcessedDataGenerator(Patient_dict, predict=True, batch_size=1, image_size=512,shuffle=True,
                                      use_window = True, window_params=[1500,-600],resample_int_val = True, resampling_step =25,   #25
                                      extract_lungs=True, size_eval=False,verbosity=verbosity,reshape = True, img_only=True)

    if model and Patient_dict and output_path:
        count=0

        for img,_,filename,params in tqdm(Patients_gen, desc='Progress'):
            # img shape: (45, 512, 512), it used to be (178, 512, 512), but after preprocessing, it throws away a lot of image slices
            filename=filename[0]
            params=params[0]
            img = np.squeeze(img)
            pdb.set_trace()

            nrrd.write(os.path.join(output_path,filename.split('/')[-2]+'_segment_torch','processed_image.nrrd'), img)

            # predicted_array = generate_segmentation(img, params, verbosity, model)  # (178, 512, 512)

            if count == len(Patients_gen):
                return 0

            count+=1



if __name__ == '__main__':
    # load model
    model = UNet2D()
    model.load_state_dict(torch.load('weights/model_weights.pth'))
    model = model.cuda()
    model.eval()

    data_path = 'test_data'
    verbosity = True
    output_path = './output'


    segment(data_path, model, output_path, verbosity)


