import cv2
import numpy as np
import scipy.stats as stats
import os

def main():
    # 11..36 : 1..55
    features = np.array(0)
    for cat_n in range(11, 36 + 1):
        print(cat_n-10, '/', 36-10)
        for image_n in range(1, 55 + 1):
            path = get_image_path(cat_n, image_n)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            (_, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            image = clip_image(image)
            features = calc_features(image, features, cat_n - 11)
    
    path_to_csv = os.path.join('..', 'data', 'train.csv')
    np.savetxt(path_to_csv, features, delimiter=',', header='area,perimeter,comp,mean_x,mean_y,var_x,var_y,assym_x,assym_y,conj_x,conj_y,conj_xy,conj_xy2,target')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_path(cat_n, image_n):
    cat_name = 'Sample' + str(cat_n).zfill(3)
    image_name = 'img' + str(cat_n).zfill(3) + '-' + str(image_n).zfill(3) + '.png'
    path = os.path.join('..', 'data', cat_name, image_name)
    return path


def clip_image(image):
    ag_row_values = np.any(image, axis=0)
    min_j = np.argmax(ag_row_values)
    max_j = image.shape[1] - np.argmax(ag_row_values[::-1])
    
    ag_col_values = np.any(image, axis=1)
    min_i = np.argmax(ag_col_values)
    max_i = image.shape[0] - np.argmax(ag_col_values[::-1])
    
    image = image[min_i:max_i, min_j:max_j]
    return image


def calc_features(image, features, target):
    bin_image = (image > 0).astype('double');
    px_area = np.sum(bin_image)
    area = px_area / (image.shape[0] * image.shape[1])
    px_perimeter = calc_perimeter(bin_image)
    perimeter = px_perimeter / (2 * image.shape[0] + 2 * image.shape[1])
    comp = px_area / (px_perimeter * px_perimeter)
    (mean_x, mean_y, var_x, var_y, assym_x, assym_y) = calc_statistics(bin_image)
    (conj_x, conj_y, conj_xy, conj_xy2) = calc_conj(bin_image)
    (conj_x, conj_y, conj_xy, conj_xy2) = (conj_x, conj_y, conj_xy, conj_xy2) / px_area
    
    #~ print(px_area, px_perimeter, comp, (mean_x, mean_y, var_x, var_y, assym_x, assym_y), (conj_x, conj_y, conj_xy, conj_xy2))
    cur_feats = np.array([[area, perimeter, comp, mean_x, mean_y, var_x, var_y, assym_x, assym_y, conj_x, conj_y, conj_xy, conj_xy2, target]])
    if features.shape == ():
        features = cur_feats
    else:
        features = np.vstack((features, cur_feats))
    #~ print(image.shape, features.shape)
    return features


def calc_perimeter(bin_image):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(bin_image,kernel,iterations = 1)
    return np.sum(bin_image) - np.sum(erosion)


def calc_statistics(bin_image):
    ids = np.transpose(np.nonzero(bin_image));
    mean_y, mean_x = tuple(np.mean(ids, axis=0))
    var_y, var_x = tuple(np.var(ids, axis=0))
    m3_y, m3_x = tuple(stats.moment(ids, moment=3, axis=0))
    assym_y = m3_y / var_y ** (3/2)
    assym_x = m3_x / var_x ** (3/2)
    
    mean_y /= bin_image.shape[0]
    mean_x /= bin_image.shape[1]
    var_y /= bin_image.shape[0] * bin_image.shape[0]
    var_x /= bin_image.shape[1] * bin_image.shape[1]
    
    return (mean_x, mean_y, var_x, var_y, assym_x, assym_y)

def calc_conj(bin_image):
    rows, cols = bin_image.shape
    image_x = np.hstack((bin_image[:, 1:], np.zeros((rows, 1))))
    image_y = np.vstack((bin_image[1:, :], np.zeros((1, cols))))
    image_xy = np.vstack((np.hstack((bin_image[1:, 1:], np.zeros((rows-1, 1)))), np.zeros((1, cols))))
    image_xy2 = np.vstack((np.zeros((1, cols)), np.hstack((bin_image[:-1, 1:], np.zeros((rows-1, 1))))))
    
    conj_x = np.sum(np.logical_and(bin_image, image_x))
    conj_y = np.sum(np.logical_and(bin_image, image_y))
    conj_xy = np.sum(np.logical_and(bin_image, image_xy))
    conj_xy2 = np.sum(np.logical_and(bin_image, image_xy2))
    
    return (conj_x, conj_y, conj_xy, conj_xy2)
    
    

if __name__ == '__main__':
    main()


