# ImrNet

This is the PyTorch implementation of ICASSP2021 paper Imrnet: An Iterative Motion Compensation and Residual Reconstruction Network for Video Compressed Sensing (https://ieeexplore.ieee.org/abstract/document/9414534) by Xin Yang, Chunling Yang.

South China University of Technology

## Description

(1) The training models of each sampling rate are saved in the check_point folder

(2) The dataloder.py file is the data loading code for non-keyframe network training 

        1. gop_size means GOP size + 1, if it is 8, then set the value to 9
        
        2. image_size indicates the image block size
        
        3. The load_filename parameter represents the parameter path list of the training data
        
(3) The dataloder1.py file is the data loading code for keyframe network training

        1. Gop_size is set to 1, which means that each training only takes a random frame in a sequence
        
        2. image_size indicates the image block size
        
        3. The load_filename parameter represents the parameter path list of the training data
        
(4) The model.py file implements the code for each model

        1. The sample_and_inirecon class is sampling and initial reconstruction, num_filters represents the number of sampling points, and B_size is the size of the sampling block
        
        2. The Biv_Shr class is a bivariate contraction in SPLNet
        
        3. The wiener_net class is the Wiener filter in SPLNet
        
        4. flownet_first and flownet are optical flow estimation subnetworks in SPYNet
        
        5. backWarp is a backward alignment function
        
        6. Prob_net is a network for weighted probability graphs
        
(5) The test_imrnet.py file is the test code

        1. rgb indicates whether to test color images, if yes, set it to True
        
        2. flag indicates whether to load the trained model, set to True
        
        3. block_size indicates the sampling block size
        
        4. gop_size indicates the GOP size
        
        5. image_width and image_height indicate that when the resolution cannot be divisible by the sampling block size, fill with 0, and the image size after filling
        
        6. img_w and img_h represent the size of the original image
        
        7. num_gop is the number of GOP
        
        8. test_video_name is the test folder, which can be placed in the same level directory as the current py file
        
        9. sr means non-key frame sampling rate
        
(9) The train_splnet_key.py file is the key frame network training code

#training (10)-(13) The py file needs to change the model path of the loading model part

(10) The train_step1_0.5_0.1.py file is the stage 1 motion estimation and motion compensation network training code

(11) The train_step2_0.5_0.1.py file is the stage 1 residual reconstruction network training code

(12) The train_step3_0.5_0.1.py file is the stage 2 motion estimation and motion compensation network training code

(13) The train_step4_0.5_0.1.py file is the stage 2 residual reconstruction network training code

(14) fnames_shuffle_test.npy, fnames_shuffle_train.npy and fnames_shuffle_val.npy represent the data path list of the training, testing and verification set data of the UCF-101 dataset

Note: All train and test.py files only need to run such as python3 train_splnet_key.py on the command line, you can modify the size of the image block in the py file and load the model to match and adapt to different settings

## Citation

If you find the code helpful in your resarch or work, please cite the following papers.

    @INPROCEEDINGS{9414534,   
      author={Yang, Xin and Yang, Chunling},     
      booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},     
      title={Imrnet: An Iterative Motion Compensation and Residual Reconstruction Network for Video Compressed Sensing},    
      year={2021},   
      volume={},      
      number={},    
      pages={2350-2354},  
      doi={10.1109/ICASSP39728.2021.9414534}}
