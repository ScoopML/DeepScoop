import os
import urllib.request
import numpy as np

# Training set urls
train_img_url   = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
train_urls = [train_img_url, train_label_url]

# Test set urls
test_img_url   = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
test_urls = [test_img_url, test_label_url]

def download_train_set(dest=None):
	if not dest:
		dest = './data/train'
	if not os.path.exists(dest):
		os.makedirs(dest)
	for url in train_urls:
		stub = url.split('/')[-1]
		fname = dest + '/' + stub
		print(f"Downloading {url} to {dest}")
		urllib.request.urlretrieve(url, fname)

def download_test_set(dest=None):
	if not dest:
		dest = './data/test'
	if not os.path.exists(dest):
		os.makedirs(dest)
	for url in test_urls:
		stub = url.split('/')[-1]
		fname = dest + '/' + stub
		print(f"Downloading {url} to {dest}")
		urllib.request.urlretrieve(url, fname)

def download_from_url(urls, dest=None):
	if not dest:
		dest = './data/all'
	if not os.path.exists(dest):
		os.makedirs(dest)
	for url in urls:
		stub = url.split('/')[-1]
		fname = dest + '/' + stub
		print(f"Downloading {url} to {dest}")
		urllib.request.urlretrieve(url, fname)


def load_mnist(dest=None):
	DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
	urls = [DATA_URL]

	if not dest:
		dest = './data/all'
	path = dest
	if not os.path.exists(path):
		download_from_url(urls, dest=dest)

	with np.load(path) as data:
		train_examples = data['x_train']
		train_labels = data['y_train']
		test_examples = data['x_test']
		test_labels = data['y_test']

	return train_examples, train_labels, test_examples, test_labels

if __name__ == '__main__':
	X, y, X_test, y_test = load_mnist()
