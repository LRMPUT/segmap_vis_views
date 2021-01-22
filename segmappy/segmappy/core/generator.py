from __future__ import print_function
import numpy as np

from ..tools.classifiertools import to_onehot


class Generator(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        n_classes,
        train=True,
        batch_size=16,
        shuffle=False,
        triplet=0
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.triplet = triplet

        self.n_segments = len(self.segment_ids)

        if self.triplet > 0:
            self.class_to_segment_id = {c: [] for c in range(self.n_classes)}
            self.classes = set()
            for seg_id in self.segment_ids:
                seg_class = self.preprocessor.classes[seg_id]

                self.class_to_segment_id[seg_class].append(seg_id)
                # if enough views for a triplet loss
                if len(self.class_to_segment_id[seg_class]) >= self.triplet:
                    self.classes.add(seg_class)
            self.classes = np.array(list(self.classes))
            if self.batch_size % self.triplet != 0:
                print('Error batch size not divisible by triplet')
                exit(-1)
            self.batch_classes = self.batch_size // self.triplet

            self.idxs = {c: 0 for c in self.classes}

            print('Found %d classes with enough views' % len(self.classes))
            self.n_batches = len(self.classes) // self.batch_classes
        else:
            self.n_batches = int(np.ceil(float(self.n_segments) / batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.triplet > 0:
            if self.shuffle and self._i == 0:
                np.random.shuffle(self.classes)

            self.batch_ids = []
            for di in range(self.batch_classes):
                cur_class = self.classes[self._i + di]
                if self.shuffle and self.idxs[cur_class] == 0:
                    np.random.shuffle(self.class_to_segment_id[cur_class])
                cur_seg_ids = self.class_to_segment_id[cur_class][self.idxs[cur_class]:
                                                                  self.idxs[cur_class] + self.triplet]
                self.idxs[cur_class] += self.triplet
                if self.idxs[cur_class] + self.triplet - 1 >= len(self.class_to_segment_id[cur_class]):
                    self.idxs[cur_class] = 0
                self.batch_ids.extend(list(cur_seg_ids))
            self.batch_ids = np.array(self.batch_ids)

            self._i = self._i + self.batch_classes
            if self._i + self.batch_classes - 1 >= len(self.classes):
                self._i = 0

        else:
            if self.shuffle and self._i == 0:
                np.random.shuffle(self.segment_ids)

            # TODO Check if this is correct during last batch
            self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]

            self._i = self._i + self.batch_size
            if self._i >= self.n_segments:
                self._i = 0

        batch_segments, batch_classes, batch_vis_views = self.preprocessor.get_processed(
            self.batch_ids, train=self.train
        )

        batch_segments = batch_segments[:, :, :, :, None]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes, batch_vis_views


class GeneratorFeatures(object):
    def __init__(self, features, classes, n_classes=2, batch_size=16, shuffle=True):
        self.features = features
        self.classes = np.asarray(classes)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = features.shape[0]
        self.n_batches = int(np.ceil(float(self.n_samples) / batch_size))
        self._i = 0

        self.sample_ids = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(self.sample_ids)

    def next(self):
        batch_ids = self.sample_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        batch_features = self.features[batch_ids, :]
        batch_classes = self.classes[batch_ids]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_features, batch_classes
