'''
Created on December 12, 2018

@author: Alejandro Molina
'''
import os
# os.environ['R_HOME'] = '/path/to/lib/R'
import pickle

import sys
sys.path.append('../')


import numpy as np
from numpy.random.mtrand import RandomState
from scipy.misc import imresize
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats, get_structure_stats_dict
# from spn.structure.Base import Context
from new_base import Context
from spn.structure.leaves.parametric.Parametric import CategoricalDictionary, Categorical, Gaussian

from ScikitCSPNClassifier import CSPNClassifier
from experiments.img_tools import get_blocks, stitch_imgs, show_img, save_img
from structure.Conditional.Inference import add_conditional_inference_support
from structure.Conditional.Sampling import add_conditional_sampling_support
from structure.Conditional.utils import concatenate_yx

add_conditional_sampling_support()
add_conditional_inference_support()


def to_ohe(x, n_values):
    return np.eye(n_values)[x]


output_path = os.path.dirname(os.path.abspath(__file__)) + '/imgs_pixelcspn_faces/'
faces = fetch_olivetti_faces()

data = fetch_olivetti_faces()
images = np.round(data['images'] * 256).astype(dtype=np.uint8)
target = data['target']
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(images, target)

from sklearn.datasets import fetch_olivetti_faces

print("faces loaded", images.shape)
print("faces for training", X_train.shape)
print("faces for testing", X_test.shape)

#    result_scaled.append(imresize(sample_img_blocks[i], (64, 64))) code for scaling


save_img(np.concatenate(images, axis=1), output_path + "images/all_input_faces.png")

min_instances_slice = 40
alpha = 0.1


# Learn cspns for image blocks like this:
#   |0|1|
#   |2|3|
# P0(0|labels)
# P1(1|0,labels)
# P2(2|1,0,labels)
# P3(3|2,1,0,labels)

num_blocks = (8, 8)
block_size = int((X_train[0].shape[0] * X_train[0].shape[1]) / (num_blocks[0] * num_blocks[1]))

datasets = []

n_people = 40
ohe_person = to_ohe(y_train, n_people)
datasets.append((y_train.reshape(-1, 1), [-1]))

for b in range((num_blocks[0] * num_blocks[1])):
    # block for P(X_i|X_j) with j < i
    block = get_blocks(X_train, num_blocks=num_blocks, blocks=list(reversed(range(b + 1))))
    block_with_person_id = (np.concatenate((block[0], ohe_person), axis=1), block[1])
    datasets.append(block_with_person_id)

# datasets = [
#     # block of  0
#     get_blocks(images, num_blocks=(2, 2), blocks=[0]),
#     # block of  1|0
#     get_blocks(images, num_blocks=(2, 2), blocks=[1, 0]),
#     # block of  2|1,0
#     get_blocks(images, num_blocks=(2, 2), blocks=[2, 1, 0]),
#     # block of  3|2,1,0
#     get_blocks(images, num_blocks=(2, 2), blocks=[3, 2, 1, 0])
# ]
cspns = []

mpe_query_blocks = None
sample_query_blocks = None
for i, (tr_block, _) in enumerate(datasets):

    fname = output_path + "spns/spn_%s.bin" % i

    # try:
        # if i == 1:
        #    raise Exception('test')

    spn = pickle.load(open(fname, 'rb'))
    cspns.append(spn)
    print("loaded %s from cache " % i)
        # continue
    # except:
    #     pass

    if i > 40:
        break
    print("learning %s " % i)

    spn = None
    if i == 0:
        ds_context = Context(parametric_types=[CategoricalDictionary])
        ds_context.add_domains(tr_block)

        spn = learn_parametric(tr_block, ds_context, min_instances_slice=min_instances_slice, ohe=False)

    else:
        cspn = CSPNClassifier(parametric_types=[Gaussian] * block_size,
                              cluster_univariate=True, min_instances_slice=min_instances_slice,
                              alpha=alpha,
                              allow_sum_nodes=True
                              )

        y = tr_block[:, 0:block_size]
        X = tr_block[:, block_size:]
        cspn.fit(X, y)
        spn = cspn.cspn

    pickle.dump(spn, open(fname, "wb"))

    cspns.append(spn)

# for i, cspn in enumerate(cspns):
#     print("spn", i)
#     for t, c in get_structure_stats_dict(cspn)['count_per_type'].items():
#         print(t, c)
# 0/0
num_images = 40

sample_images = []
rng = RandomState(17)

for i, (tr_block, _) in enumerate(datasets):

    spn = cspns[i]

    if i == 0:
        y = np.zeros((num_images, 1))
        y[:] = np.nan
        sample_instances(spn, y, rng, in_place=True)

        y[:] = 0
        data = np.zeros_like(to_ohe(y[:, 0].astype(int), n_people))
        data = np.eye(n_people)
        # data[:, 9] = 1
        # data[:, 11] = 1
        # data[:] = 1
        sample_images.insert(0, data)


    else:
        y = np.zeros((num_images, block_size))
        y[:] = np.nan

        X = np.concatenate(sample_images, axis=1)

        tr_block = sample_instances(spn, concatenate_yx(y, X), rng, in_place=False)

        y = tr_block[:, 0:block_size]

        sample_images.insert(0, y)

all_sample_images = np.concatenate(sample_images, axis=1)
samples_person_id = np.argmax(all_sample_images[:, -n_people:], axis=1)
all_sample_images = all_sample_images[:, 0:-n_people]  # remove person id

block_ids = tuple(list(reversed(range((num_blocks[0] * num_blocks[1])))))

sample_img_blocks = stitch_imgs(all_sample_images.shape[0], img_size=images[0].shape, num_blocks=num_blocks,
                                blocks={block_ids: all_sample_images})
result_scaled = []
for i in range(num_images):
    result_scaled.append(imresize(sample_img_blocks[i], (64, 64)))

fname = output_path + "images/image_all.png"
save_img(np.concatenate(result_scaled, axis=1), fname)


