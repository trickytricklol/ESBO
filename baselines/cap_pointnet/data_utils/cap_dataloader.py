import os
import numpy as np
import warnings
import pickle
import random

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class CAPModelNetDataLoader(Dataset):
    def __init__(
        self,
        root,
        split="train",
        process_data=True,
        num_point=1024,
        use_uniform_sample=True,
        use_normals=False,
        num_category=40,
    ):
        self.root = root
        self.npoints = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_test.txt"))
            ]
        else:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_test.txt"))
            ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (
                shape_names[i],
                os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt",
            )
            for i in range(len(shape_ids[split]))
        ]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints),
            )
        else:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts.dat" % (self.num_category, split, self.npoints),
            )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(
                    "Processing data %s (only running in the first time)..."
                    % self.save_path
                )
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0 : self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0 : self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def _get_additional_samples(self, label, index, num_same=16, num_diff=32):
        if self.process_data:
            same_class_indices = np.where(np.array(self.list_of_labels) == label)[0]
            diff_class_indices = np.where(np.array(self.list_of_labels) != label)[0]
        else:
            same_class_indices = [
                i
                for i, (_, fn) in enumerate(self.datapath)
                if self.classes[fn.split("/")[-2]] == label
            ]
            diff_class_indices = [
                i
                for i, (_, fn) in enumerate(self.datapath)
                if self.classes[fn.split("/")[-2]] != label
            ]

        same_class_indices = np.random.choice(
            same_class_indices, num_same, replace=len(same_class_indices) < num_same
        )
        same_class_samples = [self._get_item(idx) for idx in same_class_indices]

        diff_class_indices = np.random.choice(
            diff_class_indices, num_diff, replace=len(diff_class_indices) < num_diff
        )
        diff_class_samples = [self._get_item(idx) for idx in diff_class_indices]

        return same_class_samples, diff_class_samples

    def __getitem__(self, index):
        point_set, label = self._get_item(index)
        same_class_samples, diff_class_samples = self._get_additional_samples(
            label, index
        )

        # concatenate all samples
        all_points = (
            [point_set]
            + [item[0] for item in same_class_samples]
            + [item[0] for item in diff_class_samples]
        )
        all_labels = [label] + [label] * 16 + [item[1] for item in diff_class_samples]

        # print(all_labels)

        # print(np.stack(all_points).shape, np.array(all_labels).shape)

        return np.stack(all_points), np.array(all_labels)


if __name__ == "__main__":
    import torch

    data = CAPModelNetDataLoader("data/", split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for points, labels in DataLoader:
        print(points.shape)
        print(labels.shape)