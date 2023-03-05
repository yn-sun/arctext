# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User interface for the NAS Benchmark dataset.

Before using this API, download the data files from the links in the README.

Usage:
  # Load the data from file (this will take some time)
  nasbench = api.NASBench('/path/to/nasbench.tfrecord')

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


  # Query this model from dataset
  data = nasbench.query(model_spec)

Adjacency matrices are expected to be upper-triangular 0-1 matrices within the
defined search space (7 vertices, 9 edges, 3 allowed ops). The first and last
operations must be 'input' and 'output'. The other operations should be from
config['available_ops']. Currently, the available operations are:
  CONV3X3 = "conv3x3-bn-relu"
  CONV1X1 = "conv1x1-bn-relu"
  MAXPOOL3X3 = "maxpool3x3"

When querying a spec, the spec will first be automatically pruned (removing
unused vertices and edges along with ops). If the pruned spec is still out of
the search space, an OutOfDomainError will be raised, otherwise the data is
returned.

The returned data object is a dictionary with the following keys:
  - module_adjacency: numpy array for the adjacency matrix
  - module_operations: list of operation labels
  - trainable_parameters: number of trainable parameters in the model
  - training_time: the total training time in seconds up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy

Instead of querying the dataset for a single run of a model, it is also possible
to retrieve all metrics for a given spec, using:

  fixed_stats, computed_stats = nasbench.get_metrics_from_spec(model_spec)

The fixed_stats is a dictionary with the keys:
  - module_adjacency
  - module_operations
  - trainable_parameters

The computed_stats is a dictionary from epoch count to a list of metric
dicts. For example, computed_stats[108][0] contains the metrics for the first
repeat of the provided model trained to 108 epochs. The available keys are:
  - halfway_training_time
  - halfway_train_accuracy
  - halfway_validation_accuracy
  - halfway_test_accuracy
  - final_training_time
  - final_train_accuracy
  - final_validation_accuracy
  - final_test_accuracy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import copy
import json
import os
import random
import time

import numpy as np
import mindspore.dataset as ds


class NASBench(object):
    """User-facing API for accessing the NASBench dataset."""

    def __init__(self, dataset_file, seed=None):
        """Initialize dataset, this should only be done once per experiment.

    Args:
      dataset_file: path to .json file containing the dataset.
      seed: random seed used for sampling queried models. Two NASBench objects
        created with the same seed will return the same data points when queried
        with the same models in the same order. By default, the seed is randomly
        generated.
    """
        print('Loading dataset from file... This may take a few minutes...')
        start = time.time()
        # Loading json file and create dataset object
        dataset = ds.TextFileDataset(dataset_file)
        # Stores the fixed statistics that are independent of evaluation (i.e.,
        # adjacency matrix, operations, and number of parameters).
        # hash --> metric name --> scalar
        self.fixed_statistics = {}
        # Stores the statistics that are computed via training and evaluating the
        # model on CIFAR-10. Statistics are computed for multiple repeats of each
        # model at each max epoch length.
        # hash --> epochs --> repeat index --> metric name --> scalar
        self.computed_statistics = {}
        # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
        # {108} for the smaller dataset with only the 108 epochs.
        self.valid_epochs = set()
        # Create iterator
        iterator = dataset.create_dict_iterator(output_numpy=True)
        for line in iterator:
            # Parse the data from the data file.
            data = line['text']
            data = json.loads(data.item())
            module_hash = data["module_hash"]
            epochs = data["epochs"]
            raw_adjacency = data["adjacency"]
            adjacency = np.array(raw_adjacency)
            operations = data["operations"]
            metrics = data["metrics"]
            if module_hash not in self.fixed_statistics:
                # First time seeing this module, initialize fixed statistics.
                new_entry = {}
                new_entry['module_adjacency'] = adjacency
                new_entry['module_operations'] = operations
                new_entry['trainable_parameters'] = metrics.get("trainableParameters")
                self.fixed_statistics[module_hash] = new_entry
                self.computed_statistics[module_hash] = {}

            self.valid_epochs.add(epochs)

            if epochs not in self.computed_statistics[module_hash]:
                self.computed_statistics[module_hash][epochs] = []

            # Each data_point consists of the metrics recorded from a single
            # train-and-evaluation of a model at a specific epoch length.
            data_point = {}

            # Note: metrics.evaluation_data[0] contains the computed metrics at the
            # start of training (step 0) but this is unused by this API.

            # Evaluation statistics at the half-way point of training
            half_evaluation = metrics.get("evaluationData")[1]
            data_point['halfway_training_time'] = half_evaluation.get("trainingTime")
            data_point['halfway_train_accuracy'] = half_evaluation.get("trainAccuracy")
            data_point['halfway_validation_accuracy'] = (
                half_evaluation.get("validationAccuracy"))
            data_point['halfway_test_accuracy'] = half_evaluation.get("testAccuracy")

            # Evaluation statistics at the end of training
            final_evaluation = metrics.get("evaluationData")[2]
            data_point['final_training_time'] = final_evaluation.get("trainingTime")
            data_point['final_train_accuracy'] = final_evaluation.get("trainAccuracy")
            data_point['final_validation_accuracy'] = (
                final_evaluation.get("validationAccuracy"))
            data_point['final_test_accuracy'] = final_evaluation.get("testAccuracy")

            self.computed_statistics[module_hash][epochs].append(data_point)

        elapsed = time.time() - start
        print('Loaded dataset in %d seconds' % elapsed)

        self.history = {}
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def hash_iterator(self):
        """Returns iterator over all unique model hashes."""
        return self.fixed_statistics.keys()

    def get_metrics_from_hash(self, module_hash):
        """Returns the metrics for all epochs and all repeats of a hash.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
        fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
        return fixed_stat, computed_stat


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy objects to JSON-serializable format."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Matrices converted to nested lists
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # Scalars converted to closest Python type
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)
