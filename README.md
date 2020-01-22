# Peking University/Baidu - Autonomous Driving
This is the code for my part of 7th place solution in [Peking University/Baidu - Autonomous Driving](https://www.kaggle.com/c/pku-autonomous-driving/).  
Our entire solution is in [this thread](https://www.kaggle.com/c/pku-autonomous-driving/discussion/127034)

I started with [hocop1's really great kernel](https://www.kaggle.com/hocop1/centernet-baseline), so basic part of my code is based on his.
My main contributions are below:
 - Change the CenterNet implementation to [this one](https://github.com/see--/kuzushiji-recognition)
 - Change backbone to resnet152
 - change the depth representation to 1 / (sigmoid(z) - 1
 - remove x, y, yaw(actually pitch), roll prediction(we can predict yaw and roll but it was almost same as train set's median)
 - remove optimize xyz, just convert rcz to xyz
 - increase image size to (832, 2080) or (960, 2400) or (1088, 2720)
 - change the distance threshold to 4
 - change the confidence threshold to 0.1
 - ensemble 5 folds
 - remove false positive by using masks in 'test_masks/'
 
 I tried 3 scale ensemble, but it might be not so different from just 5 fold ensemble of image_size=(1088,2720).  
 My single fold model achieve public0.097/private0.091 and 5 fold ensemble got public0.117/private0.105.  
 With ensmebling Phalanx's model and Jhui's model, we got public0.129/private0.119.  

## Usage

```
$ python resize_images.py --config config/003_r152d_img1088.yml
$ python split_folds.py --config config/003_r152d_img1088.yml
$ python train.py --config config/cls/003_r152d_img1088.yml
$ python prediction.py --config_dir config/003_r152d_img1088.yml --ensemble
```

## Reference

- My code for this competition specific part is based on [this great starter kernel](https://www.kaggle.com/hocop1/centernet-baseline)  
- CenterNet implementation is from [see--'s code for Kuzushiji competition](https://github.com/see--/kuzushiji-recognition)
- To evaluate models, [this tito's kernel](https://www.kaggle.com/its7171/metrics-evaluation-script) helps me a lot. 
But eventually we evaluate our model by f1-like score. Please check metric code for the detail.
