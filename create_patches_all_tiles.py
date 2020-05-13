import skimage.io as skio
from skimage.transform import resize
import sklearn.feature_extraction.image as skfi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


f_test = open('S2_tiles_testing.txt', 'r')
save_test_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/test/"

for i in f_test:
    print(i)
    im_name = i
    date = im_name[11:26]

    root_path = "/mnt/gpid07/users/oriol.esquena/images_sentinel/test/"

    # read the image and normalize its data
    im10 = skio.imread(root_path+"10m/"+date+".tiff")
    im20 = skio.imread(root_path+"20m/"+date+".tiff")

    # create patches out of the image
    i = 0
    j = 0
    num_patches = 90
    num_final_patches = 30
    size_im10 = 512
    resize_im10 = 256
    size_im20 = 256
    resize_im20 = 128
    channels10 = 4
    channels20 = 6
    max_pixel = np.round((im10.shape[0] - size_im10)/2).astype(dtype=np.int)
    # patch_8bit = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patch_norm10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patch_norm20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
    gauss10 = np.ndarray((num_final_patches, size_im10, size_im10, channels10))
    gauss20 = np.ndarray((num_final_patches, size_im20, size_im20, channels20))
    rs10 = np.ndarray((num_final_patches, resize_im10, resize_im10, channels10))
    rs20 = np.ndarray((num_final_patches, resize_im20, resize_im20, channels20))

    patches10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patches20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
    patches20_target = np.ndarray((num_final_patches, size_im20, size_im20, channels20))

    for i in range(num_patches):
        j10 = np.random.randint(max_pixel)*2
        j20 = np.round((j10/2)).astype(dtype=np.int)
        k10 = np.random.randint(max_pixel)*2
        k20 = np.round((k10 / 2) - 1).astype(dtype=np.int)
        patches10[i] = im10[j10:(j10+size_im10), k10:(k10+size_im10), :]
        patches20[i] = im20[j20:(j20+size_im20), k20:(k20+size_im20), :]

    for i in range(num_patches):
#         patch_norm10[i] = patches10[i]/(patches10[i].max())
#         patch_norm20[i] = patches20[i]/(patches20[i].max())
        if j < num_final_patches and ((patches10[i].max() < 5000 or patches20[i].max() < 4000) or
                                      (patches10[i].max() - patches10[i].min() > 800 or
                                       patches20[i].max() - patches20[i].min() > 800)):
            gauss10[j] = gaussian_filter(patches10[i], sigma=1 / 2)
            gauss20[j] = gaussian_filter(patches20[i], sigma=1 / 2)
            rs10[j] = resize(gauss10[j], (resize_im10, resize_im10))
            rs20[j] = resize(gauss20[j], (resize_im20, resize_im20))
            patches20_target[j] = patches20[i]
            j += 1

    np.save(save_test_path+"test_resized20_"+date+".npy", rs10)
    np.save(save_test_path+"test_resized40_"+date+".npy", rs20)
    np.save(save_test_path+"real20_target_test_"+date+".npy", patches20_target)

f_test.close()
print("Test finished")

save_train_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/train/"
f_train = open('S2_tiles_training.txt', 'r')

for i in f_train:
    im_name = i
    date = im_name[11:26]

    root_path = "/mnt/gpid07/users/oriol.esquena/images_sentinel/train/"

    # read the image and normalize its data
    im10 = skio.imread(root_path+"10m/"+date+".tiff")
    im20 = skio.imread(root_path+"20m/"+date+".tiff")

    # create patches out of the image
    i = 0
    j = 0
    num_patches = 300
    num_final_patches = 100
    size_im10 = 512
    resize_im10 = 256
    size_im20 = 256
    resize_im20 = 128
    channels10 = 4
    channels20 = 6
    max_pixel = np.round((im10.shape[0] - size_im10)/2).astype(dtype=np.int)
    # patch_8bit = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patch_norm10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patch_norm20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
    gauss10 = np.ndarray((num_final_patches, size_im10, size_im10, channels10))
    gauss20 = np.ndarray((num_final_patches, size_im20, size_im20, channels20))
    rs10 = np.ndarray((num_final_patches, resize_im10, resize_im10, channels10))
    rs20 = np.ndarray((num_final_patches, resize_im20, resize_im20, channels20))

    patches10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
    patches20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
    patches20_target = np.ndarray((num_final_patches, size_im20, size_im20, channels20))

    for i in range(num_patches):
        j10 = np.random.randint(max_pixel)*2
        j20 = np.round((j10/2)).astype(dtype=np.int)
        k10 = np.random.randint(max_pixel)*2
        k20 = np.round((k10 / 2) - 1).astype(dtype=np.int)
        patches10[i] = im10[j10:(j10+size_im10), k10:(k10+size_im10), :]
        patches20[i] = im20[j20:(j20+size_im20), k20:(k20+size_im20), :]
        print(i)

    for i in range(num_patches):
        patch_norm10[i] = patches10[i]/(patches10[i].max())
        patch_norm20[i] = patches20[i]/(patches20[i].max())
        if j < num_final_patches and ((patches10[i].max() < 5000 or patches20[i].max() < 4000) or
                                      (patches10[i].max() - patches10[i].min() > 800 or
                                       patches20[i].max() - patches20[i].min() > 800)):
            gauss10[j] = gaussian_filter(patches10[i], sigma=1 / 2)
            gauss20[j] = gaussian_filter(patches20[i], sigma=1 / 2)
            rs10[j] = resize(gauss10[j], (resize_im10, resize_im10))
            rs20[j] = resize(gauss20[j], (resize_im20, resize_im20))
            patches20_target[j] = patches20[i]
            j += 1

    np.save(save_train_path+"input10_resized20_"+date+".npy", rs10)
    np.save(save_train_path+"input20_resized40_"+date+".npy", rs20)
    np.save(save_train_path+"real20_target_"+date+".npy", patches20_target)

print("Train finished")
