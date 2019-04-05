import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_color(image, k=4, image_processing_size = None):
	if image_processing_size is not None:
		image = cv2.resize(image, image_processing_size,
                            interpolation = cv2.INTER_AREA)
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	clt = KMeans(n_clusters = k)
	labels = clt.fit_predict(image)
	label_counts = Counter(labels)
	dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
	return list(dominant_color);

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="Path to image to find dominant color of")

ap.add_argument("-k", "--clusters", default=4, type=int,
	help="Number of clusters to use in kmeans when finding dominant color")

args = vars(ap.parse_args())


image = cv2.imread(args['imagePath'])

dom_color = get_dominant_color(image, args['clusters'])
print(dom_color)
dom_color_hsv = np.full(image.shape, dom_color, dtype='uint8')

output_image = np.hstack((image, dom_color_hsv))
cv2.imshow("HSV Combined", output_image)

cv2.waitKey()
cv2.destroyAllWindows()
