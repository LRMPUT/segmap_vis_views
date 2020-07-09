import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    vert_range = 35.5 * np.pi / 180.0

    dataset_folder = '/mnt/data/datasets/JW/MulRan/DCC01/sensor_data/Ouster'
    scan_files = sorted(os.listdir(dataset_folder))
    for i in range(0, len(scan_files), 10):
        scan_file = scan_files[i]

        print('image ', scan_file)

        data = np.fromfile(os.path.join(dataset_folder, scan_file), dtype=np.float32).reshape((-1, 4))

        int_im = np.zeros((64, 1024), dtype=np.float)
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
                vert_coord = (vert_range / 2.0 - vert_angle) / vert_range * 64

                hor_angles.append(hor_angle)
                vert_angles.append(vert_angle)
                hor_coords.append(hor_coord)
                vert_coords.append(vert_coord)
                vert_coord = round(float(vert_coord))
                hor_coord = round(float(hor_coord))

                if 0 <= vert_coord < 64 and 0 <= hor_coord < 1024:
                    int_im[vert_coord, hor_coord] = pt[3]

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

        # _ = plt.hist(hor_coords, bins=100, range=(510, 514))
        # _ = plt.hist(vert_coords, bins=512, range=(30, 34))
        # _ = plt.hist(vert_angles*180.0/np.pi, bins=100, range=(-1, 1))
        # plt.title("Histogram")
        # plt.show()

        cv2.imshow('int', np.minimum(int_im, 1500) / 1500)
        cv2.waitKey()
    pass


if __name__=='__main__':
    main()
