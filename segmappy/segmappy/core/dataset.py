from __future__ import print_function
import numpy as np
import os
import cv2

from .config import get_default_dataset_dir


class Dataset(object):
    # load config values
    def __init__(
        self,
        folder="dataset",
        base_dir=get_default_dataset_dir(),
        require_change=0.0,
        use_merges=True,
        keep_match_thresh=0.0,
        use_matches=True,
        min_class_size=1,
        require_relevance=0.0,
        require_diff_points=0,
        normalize_classes=True,
        use_visual=False,
        largest_vis_view=False
    ):
        abs_folder = os.path.abspath(os.path.join(base_dir, folder))
        try:
            assert os.path.isdir(abs_folder)
        except AssertionError:
            raise IOError("Dataset folder {} not found.".format(abs_folder))

        self.folder = abs_folder
        self.require_change = require_change
        self.use_merges = use_merges
        self.keep_match_thresh = keep_match_thresh
        self.use_matches = use_matches
        self.min_class_size = min_class_size
        self.require_relevance = require_relevance
        self.require_diff_points = require_diff_points
        self.normalize_classes = normalize_classes
        self.use_visual = use_visual
        self.largest_vis_view = largest_vis_view

    # load the segment dataset
    def load(self, preprocessor=None):
        from ..tools.import_export import load_segments, load_positions, load_features, load_vis_views

        # load all the csv files
        self.segments, sids, duplicate_sids = load_segments(folder=self.folder)
        self.positions, pids, duplicate_pids = load_positions(folder=self.folder)
        self.features, self.feature_names, fids, duplicate_fids = load_features(
            folder=self.folder
        )
        self.int_paths, self.mask_paths, self.range_paths, vids, duplicate_vids = load_vis_views(folder=self.folder)

        self.classes = np.array(sids)
        self.duplicate_classes = self.classes.copy()
        self.positions = np.array(self.positions)
        self.features = np.array(self.features)
        self.duplicate_ids = np.array(duplicate_sids)

        complete_id_to_vidx = {(sid, dsid): vidx for vidx, (sid, dsid) in enumerate(zip(vids, duplicate_vids))}
        # self._remove_no_vis(complete_id_to_vidx)

        vorder = [complete_id_to_vidx[csid] for csid in zip(self.classes, self.duplicate_ids)]
        self.int_paths = [self.int_paths[idx] for idx in vorder]
        self.mask_paths = [self.mask_paths[idx] for idx in vorder]
        self.range_paths = [self.range_paths[idx] for idx in vorder]
        vids = [vids[idx] for idx in vorder]
        duplicate_vids = [duplicate_vids[idx] for idx in vorder]

        for i in range(len(vids)):
            if (vids[i], duplicate_vids[i]) != (self.classes[i], self.duplicate_ids[i]):
                raise Exception('Error: ', (vids[i], duplicate_vids[i]), (self.classes[i], self.duplicate_ids[i]))

        # load labels
        from ..tools.import_export import load_labels

        self.labels, self.lids = load_labels(folder=self.folder)
        self.labels = np.array(self.labels)
        self.labels_dict = dict(zip(self.lids, self.labels))

        # load matches
        from ..tools.import_export import load_matches

        self.matches = load_matches(folder=self.folder)

        if self.require_change > 0.0:
            self._remove_unchanged()

        # combine sequences that are part of a merger
        if self.use_merges:
            from ..tools.import_export import load_merges

            merges, _ = load_merges(folder=self.folder)
            self._combine_sequences(merges)
            self.duplicate_classes = self.classes.copy()

        # remove small irrelevant segments
        if self.require_relevance > 0:
            self._remove_irrelevant()

        # only use segments that are different enough
        if self.require_diff_points > 0:
            assert preprocessor is not None
            self._remove_similar(preprocessor)

        self._remove_poorly_visible()

        if self.largest_vis_view:
            self._select_largest_vis_views(vids, duplicate_vids)

        # combine classes based on matches
        if self.use_matches:
            self._combine_classes()

        # normalize ids and remove small classes
        self._normalize_classes()

        print(
            "  Found",
            self.n_classes,
            "valid classes with",
            len(self.segments),
            "segments",
        )

        self._sort_ids()

        # self._calc_stats()

        return (
            self.segments,
            self.positions,
            self.classes,
            self.n_classes,
            self.features,
            self.matches,
            self.labels_dict,
            self.int_paths,
            self.mask_paths,
            self.range_paths
        )

    def _remove_unchanged(self):
        keep = np.ones(self.classes.size).astype(np.bool)
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]

            prev_size = self.segments[class_ids[0]].shape[0]
            for class_id in class_ids[1:]:
                size = self.segments[class_id].shape[0]
                if size < prev_size * (1.0 + self.require_change):
                    keep[class_id] = False
                else:
                    prev_size = size

        self._trim_data(keep)

        print("  Found %d segments that changed enough" % len(self.segments))

    # list of sequence pairs to merge and correct from the matches table
    def _combine_sequences(self, merges):
        # calculate the size of each sequence based on the last element
        last_sizes = {}
        subclasses = {}
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            last_id = class_ids[np.argmax(self.duplicate_ids[class_ids])]
            last_sizes[cls] = len(self.segments[last_id])
            subclasses[cls] = []

        # make merges and keep a list of the merged sequences for each class
        for merge in merges:
            merge_sequence, target_sequence = merge

            merge_ids = np.where(self.classes == merge_sequence)[0]
            target_ids = np.where(self.classes == target_sequence)[0]

            if merge_ids.size > 0 and target_ids.size > 0:
                self.classes[merge_ids] = target_sequence
                self.duplicate_ids[target_ids] += merge_ids.size

                subclasses[target_sequence].append(merge_sequence)
                subclasses[target_sequence] += subclasses[merge_sequence]
                del subclasses[merge_sequence]

        # calculate how relevant the merges are based on size
        relevant = {}
        new_class = {}
        for main_class in subclasses:
            relevant[main_class] = True
            new_class[main_class] = main_class

            main_size = last_sizes[main_class]
            for sub_class in subclasses[main_class]:
                new_class[sub_class] = main_class
                sub_size = last_sizes[sub_class]
                if float(sub_size) / main_size < self.keep_match_thresh:
                    relevant[sub_class] = False
                else:
                    relevant[sub_class] = True

        # ignore non-relevant merges and for the relevant merges replace
        # the merged class with the new class name
        new_matches = []
        for match in self.matches:
            new_match = []
            for cls in match:
                if relevant[cls]:
                    new_match.append(new_class[cls])

            if len(new_match) > 1:
                new_matches.append(new_match)

        print("  Found %d matches that are relevant after merges" % len(new_matches))

        self.matches = new_matches

    # combine the classes in a 1d vector of labeled classes based on a 2d
    # listing of segments that should share the same class
    def _combine_classes(self):
        # filtered out non-unique matches
        unique_matches = set()
        for match in self.matches:
            unique_match = []
            for cls in match:
                if cls not in unique_match:
                    unique_match.append(cls)

            if len(unique_match) > 1:
                unique_match = tuple(sorted(unique_match))
                if unique_match not in unique_matches:
                    unique_matches.add(unique_match)

        unique_matches = [list(match) for match in unique_matches]
        print("  Found %d matches that are unique" % len(unique_matches))

        # combine matches with classes in common
        groups = {}
        class_group = {}

        for i, cls in enumerate(np.unique(unique_matches)):
            groups[i] = [cls]
            class_group[cls] = i

        for match in unique_matches:
            main_group = class_group[match[0]]

            for cls in match:
                other_group = class_group[cls]
                if other_group != main_group:
                    for other_class in groups[other_group]:
                        if other_class not in groups[main_group]:
                            groups[main_group].append(other_class)
                            class_group[other_class] = main_group

                    del groups[other_group]

        self.matches = [groups[i] for i in groups]
        print("  Found %d matches after grouping" % len(self.matches))

        # combine the sequences into the same class
        for match in self.matches:
            assert len(match) > 1
            for other_class in match[1:]:
                self.classes[self.classes == other_class] = match[0]

    # make class ids sequential and remove classes that are too small
    def _normalize_classes(self):
        # mask of segments to keep
        keep = np.ones(self.classes.size).astype(np.bool)

        # number of classes and current class counter
        self.n_classes = 0
        for i in np.unique(self.classes):
            # find the elements in the class
            idx = self.classes == i
            if np.sum(idx) >= self.min_class_size:
                # if class is large enough keep and relabel
                if self.normalize_classes:
                    self.classes[idx] = self.n_classes

                # found one more class
                self.n_classes = self.n_classes + 1
            else:
                # mark class for removal and delete label information
                keep = np.logical_and(keep, np.logical_not(idx))

        # remove data on the removed classes
        self._trim_data(keep)

    # remove segments that are too small compared to the last
    # element in the sequence
    def _remove_irrelevant(self):
        keep = np.ones(self.classes.size).astype(np.bool)
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            last_id = class_ids[np.argmax(self.duplicate_ids[class_ids])]
            last_size = len(self.segments[last_id])

            for class_id in class_ids:
                segment_size = len(self.segments[class_id])
                if float(segment_size) / last_size < self.require_relevance:
                    keep[class_id] = False

        self._trim_data(keep)

        print("  Found %d segments that are relevant" % len(self.segments))

    # remove segments that are too similar based on hamming distance
    def _remove_similar(self, preprocessor):
        keep = np.ones(self.classes.size).astype(np.bool)
        for c in np.unique(self.classes):
            class_ids = np.where(self.classes == c)[0]

            # sort duplicates in chronological order
            class_ids = class_ids[np.argsort(self.duplicate_ids[class_ids])]

            segments_class = [self.segments[i] for i in class_ids]
            segments_class = preprocessor._rescale_coordinates(segments_class)
            segments_class = preprocessor._voxelize(segments_class)

            for i, segment_1 in enumerate(segments_class):
                for segment_2 in segments_class[i + 1 :]:
                    diff = np.sum(np.abs(segment_1 - segment_2))

                    if diff < self.require_diff_points:
                        keep[class_ids[i]] = False
                        break

        self._trim_data(keep)

        print("  Found %d segments that are dissimilar" % len(self.segments))

    # remove segments that are poorly visible
    def _remove_poorly_visible(self):
        keep = np.ones(self.classes.size).astype(np.bool)
        for i in range(self.classes.size):
            cur_mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_ANYDEPTH)
            cnt = np.nonzero(cur_mask)[0].size

            if cnt < 200:
                keep[i] = False

        self._trim_data(keep)

        print("  Found %d segments that are properly visible" % len(self.segments))

    def _remove_no_vis(self, complete_id_to_vidx):
        keep = np.ones(self.classes.size).astype(np.bool)
        for idx, csid in enumerate(zip(self.classes, self.duplicate_ids)):
            if csid not in complete_id_to_vidx:
                keep[idx] = False

        self._trim_data_no_vis(keep)

        print("  Found %d segments that have views" % len(self.segments))

    def _sort_ids(self):
        ordered_ids = []
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            class_sequences = self.duplicate_classes[class_ids]
            unique_sequences = np.unique(class_sequences)

            for unique_sequence in unique_sequences:
                sequence_ids = np.where(class_sequences == unique_sequence)[0]
                sequence_ids = class_ids[sequence_ids]
                sequence_frame_ids = self.duplicate_ids[sequence_ids]

                # order chronologically according to frame id
                sequence_ids = sequence_ids[np.argsort(sequence_frame_ids)]

                ordered_ids += sequence_ids.tolist()

        ordered_ids = np.array(ordered_ids)

        self.segments = [self.segments[i] for i in ordered_ids]
        self.classes = self.classes[ordered_ids]

        if self.positions.size > 0:
            self.positions = self.positions[ordered_ids]
        if self.features.size > 0:
            self.features = self.features[ordered_ids]
        if len(self.int_paths) > 0:
            self.int_paths = [self.int_paths[i] for i in ordered_ids]
        if len(self.mask_paths) > 0:
            self.mask_paths = [self.mask_paths[i] for i in ordered_ids]
        if len(self.range_paths) > 0:
            self.range_paths = [self.range_paths[i] for i in ordered_ids]

        self.duplicate_ids = self.duplicate_ids[ordered_ids]
        self.duplicate_classes = self.duplicate_classes[ordered_ids]

    # keep only segments and corresponding data where the keep parameter is true
    def _trim_data(self, keep):
        self.segments = [segment for (k, segment) in zip(keep, self.segments) if k]
        self.classes = self.classes[keep]

        if self.positions.size > 0:
            self.positions = self.positions[keep]
        if self.features.size > 0:
            self.features = self.features[keep]
        if len(self.int_paths) > 0:
            self.int_paths = [int_path for (k, int_path) in zip(keep, self.int_paths) if k]
        if len(self.mask_paths) > 0:
            self.mask_paths = [mask_path for (k, mask_path) in zip(keep, self.mask_paths) if k]
        if len(self.range_paths) > 0:
            self.range_paths = [range_path for (k, range_path) in zip(keep, self.range_paths) if k]

        self.duplicate_ids = self.duplicate_ids[keep]
        self.duplicate_classes = self.duplicate_classes[keep]

    def _trim_data_no_vis(self, keep):
        self.segments = [segment for (k, segment) in zip(keep, self.segments) if k]
        self.classes = self.classes[keep]

        if self.positions.size > 0:
            self.positions = self.positions[keep]
        if self.features.size > 0:
            self.features = self.features[keep]

        self.duplicate_ids = self.duplicate_ids[keep]
        self.duplicate_classes = self.duplicate_classes[keep]

    def _calc_stats(self):
        int_mean = 0.0
        range_mean = 0.0
        for i in range(len(self.int_paths)):
            cur_int = cv2.imread(self.int_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)
            # cur_mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)
            cur_range = cv2.imread(self.range_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)

            cur_int_mean = np.mean(cur_int[cur_range > 1000.0])
            cur_range_mean = np.mean(cur_range[cur_range > 1000.0])
            int_mean = float(i) / (i + 1) * int_mean + 1.0 / (i + 1) * cur_int_mean
            range_mean = float(i) / (i + 1) * range_mean + 1.0 / (i + 1) * cur_range_mean

        int_var = 0.0
        range_var = 0.0
        for i in range(len(self.int_paths)):
            cur_int = cv2.imread(self.int_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)
            # cur_mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)
            cur_range = cv2.imread(self.range_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float)

            cur_N = np.count_nonzero(cur_range > 2.0)
            cur_int_var = np.sum(np.square(cur_int[cur_range > 1000.0] - int_mean)) / (cur_N - 1)
            cur_range_var = np.sum(np.square(cur_range[cur_range > 1000.0] - range_mean)) / (cur_N - 1)
            int_var = float(i) / (i + 1) * int_var + 1.0 / (i + 1) * cur_int_var
            range_var = float(i) / (i + 1) * range_var + 1.0 / (i + 1) * cur_range_var

        print('int mean: ', int_mean)
        print('int stddev: ', np.sqrt(int_var))
        print('range mean: ', range_mean)
        print('range stddev: ', np.sqrt(range_var))

    def _select_largest_vis_views(self, vids, duplicate_vids):
        cvididxs = sorted(zip(self.classes, self.duplicate_ids, range(len(self.classes))))

        cid_to_idx = {}

        prev_id = -1
        largest_idx = 0
        largest_size = 0
        for id, did, idx in cvididxs:
            if id != prev_id:
                largest_idx = idx
                largest_size = 0
                prev_id = id

            cur_mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_ANYDEPTH)
            cur_size = np.nonzero(cur_mask)[0].size
            if cur_size > largest_size:
                largest_size = cur_size
                largest_idx = idx

            cid_to_idx[(id, did)] = largest_idx

        new_int_paths = []
        new_mask_paths = []
        new_range_paths = []

        for idx, csid in enumerate(zip(self.classes, self.duplicate_ids)):
            new_int_paths.append(self.int_paths[cid_to_idx[csid]])
            new_mask_paths.append(self.mask_paths[cid_to_idx[csid]])
            new_range_paths.append(self.range_paths[cid_to_idx[csid]])

        self.int_paths = new_int_paths
        self.mask_paths = new_mask_paths
        self.range_paths = new_range_paths
