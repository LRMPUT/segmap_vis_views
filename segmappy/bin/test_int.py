import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    vert_range = 35.68 * np.pi / 180.0

    # dataset_folder = '/mnt/data/datasets/JW/MulRan/DCC01/sensor_data/Ouster'
    dataset_folder = '/media/janw/JanW/datasets/JW/KITTI/2011_09_30_drive_0020_sync/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data'
    scan_files = sorted(os.listdir(dataset_folder))
    for i in range(0, len(scan_files), 10):
        scan_file = scan_files[i]

        print('image ', scan_file)

        data = np.fromfile(os.path.join(dataset_folder, scan_file), dtype=np.float32).reshape((-1, 4))

        int_im = np.zeros((64, 2048), dtype=np.float)
        hor_angles = []
        vert_angles = []
        hor_coords = []
        vert_coords = []
        for pt in data:
            range_val = np.linalg.norm(pt[0:3])
            if range_val > 1.0:
                hor_angle = np.arctan2(-pt[1], pt[0])
                if hor_angle < 0:
                    hor_angle += 2 * np.pi
                hor_coord = hor_angle / (2 * np.pi) * 1024

                hor_range = np.linalg.norm(pt[0:2])
                vert_angle = np.arctan2(pt[2], hor_range)
                vert_coord = (vert_range / 2.0 - vert_angle) / vert_range * (64 - 1)

                hor_angles.append(hor_angle)
                vert_angles.append(vert_angle)
                hor_coords.append(hor_coord)
                vert_coords.append(vert_coord)
                vert_coord = round(float(vert_coord))
                hor_coord = round(float(hor_coord))

                if 0 <= vert_coord < 64 and 0 <= hor_coord < 1024:
                    int_im[vert_coord, hor_coord] = pt[3]
                else:
                    # print('vert angle = ', vert_angle * 180.0 / np.pi)
                    pass

        # data = data.reshape((1024, 64, 4))
        # int_im = data[:, :, 3].transpose((1, 0))
        # int_im_r = int_im
        # int_im_r[1::4, :-6] = int_im_r[1::4, 6:]
        # int_im_r[2::4, :-12] = int_im_r[2::4, 12:]
        # int_im_r[3::4, :-18] = int_im_r[3::4, 18:]
        # int_im_r[0:16] = int_im[0::4]
        # int_im_r[16:32] = int_im[1::4]
        # int_im_r[32:48] = int_im[2::4]
        # int_im_r[48:64] = int_im[3::4]
        hor_angles = np.array(hor_angles)
        vert_angles = np.array(vert_angles)
        hor_coords = np.array(hor_coords)
        vert_coords = np.array(vert_coords)

        # vert_angles_q = (vert_angles * 180.0 / np.pi * 500).astype(int)
        # vert_angles_q = np.unique(vert_angles_q)/500.0
        # print('quantized = ', vert_angles_q)
        # print('diffs = ', np.diff(vert_angles_q))
        clust_tol = 0.01
        vert_angles_s = sorted(vert_angles * 180.0 / np.pi)
        vert_angles_c = []
        beg_idx = -1
        beg_val = -360.0
        for cur_idx, ang in enumerate(vert_angles_s):
            if ang - beg_val > clust_tol:
                if beg_idx >= 0:
                    val = np.mean(vert_angles_s[beg_idx: cur_idx])
                    vert_angles_c.append(val)
                beg_idx = cur_idx
                beg_val = ang
        val = np.mean(vert_angles_s[beg_idx:])
        vert_angles_c.append(val)
        np.set_printoptions(precision=4)
        print('clusters = ', np.array(vert_angles_c))
        print('diffs = ', np.diff(vert_angles_c))

        # plt.ion()
        # _ = plt.hist(hor_coords, bins=100, range=(510, 514))
        # _ = plt.hist(vert_coords, bins=512, range=(30, 34))
        # _ = plt.hist(hor_angles*180.0/np.pi, bins=200, range=(179, 181))
        # _ = plt.hist(vert_angles*180.0/np.pi, bins=512, range=(-18, 18))
        # plt.title("Histogram")
        # plt.show()

        cv2.imshow('int', np.minimum(int_im, 1500) / 1500)
        cv2.waitKey()
    pass


if __name__=='__main__':
    main()
