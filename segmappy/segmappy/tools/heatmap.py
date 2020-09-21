import numpy as np
import cv2
from scipy.interpolate import interpn
from scipy.ndimage import zoom
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def plot_heatmap_img(batch_segments,
                     batch_vis_views,
                     batch_classes,
                     batch_scales,
                     batch_conv_vis,
                     batch_descriptor,
                     sess,
                     cnn_descriptor,
                     cnn_input,
                     cnn_input_vis,
                     cnn_y_true,
                     cnn_scales
                     ):

    intensity = batch_vis_views[:, :, :, 0] * 173.09 + 209.30
    mask = batch_vis_views[:, :, :, 1] * 255.0
    weights = np.zeros((batch_conv_vis.shape[0], batch_conv_vis.shape[3]), dtype=np.float)
    for layer in range(batch_conv_vis.shape[3]):
        l_heatmap = batch_conv_vis[:, :, :, layer]
        cur_batch_vis_views = np.zeros(batch_vis_views.shape, dtype=np.uint8)
        for b in range(l_heatmap.shape[0]):
            cur_heatmap = cv2.resize(l_heatmap[b], (intensity.shape[2], intensity.shape[1]))
            cur_heatmap = np.maximum(cur_heatmap, 0.0)
            minh = np.min(cur_heatmap)
            maxh = np.max(cur_heatmap)
            cur_heatmap = (cur_heatmap - minh) / (maxh - minh)
            cur_batch_vis_views[b, :, :, :] = batch_vis_views[b, :, :, :] * np.expand_dims(cur_heatmap,
                                                                                           axis=-1)

        [cur_batch_descriptor] = sess.run(
                [cnn_descriptor],
                feed_dict={
                    cnn_input: batch_segments,
                    cnn_input_vis: cur_batch_vis_views,
                    cnn_y_true: batch_classes,
                    cnn_scales: batch_scales,
                },
        )
        weights[:, layer] = np.linalg.norm(batch_descriptor - cur_batch_descriptor, axis=-1)

    weights = 1.0 / np.maximum(weights, 1e-4)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    heatmap = batch_conv_vis * weights[:, np.newaxis, np.newaxis, :]
    heatmap = np.sum(heatmap, axis=-1)
    img_heatmap_np = np.zeros(intensity.shape + (3,), dtype=np.uint8)
    for b in range(heatmap.shape[0]):
        cur_heatmap = cv2.resize(heatmap[b], (intensity.shape[2], intensity.shape[1]))
        cur_heatmap = np.maximum(cur_heatmap, 0.0)
        cur_heatmap = cur_heatmap / np.max(cur_heatmap)
        cur_heatmap = cv2.applyColorMap(np.uint8(255 * cur_heatmap), cv2.COLORMAP_JET)
        cur_heatmap = cv2.cvtColor(cur_heatmap, cv2.COLOR_BGR2RGB)
        cur_img_heatmap = 0.5 * cur_heatmap + \
                          np.tile(np.expand_dims(np.minimum(intensity[b] * 255.0 / 1500.0, 255.0), axis=-1),
                                  [1, 1, 3])

        cur_img_heatmap[mask[b] > 128.0, :] = np.array([255, 0, 0], dtype=np.uint8)
        img_heatmap_np[b] = cur_img_heatmap

    return img_heatmap_np


def resize_voxels(voxels, size):
    # x = np.linspace(0, voxels.shape[0] - 1, num=voxels.shape[0])
    # y = np.linspace(0, voxels.shape[1] - 1, num=voxels.shape[1])
    # z = np.linspace(0, voxels.shape[2] - 1, num=voxels.shape[2])
    voxels_resized = zoom(voxels, size / voxels.shape)

    return voxels_resized


def plot_heatmap_vox(batch_segments,
                     batch_vis_views,
                     batch_classes,
                     batch_scales,
                     batch_conv,
                     batch_descriptor,
                     sess,
                     cnn_descriptor,
                     cnn_input,
                     cnn_input_vis,
                     cnn_y_true,
                     cnn_scales
                     ):

    weights = np.zeros((batch_conv.shape[0], batch_conv.shape[-1]), dtype=np.float)
    for layer in range(batch_conv.shape[-1]):
        l_heatmap = batch_conv[:, :, :, :, layer]
        cur_batch_vis_views = np.zeros(batch_segments.shape, dtype=np.uint8)
        for b in range(l_heatmap.shape[0]):
            cur_heatmap = resize_voxels(l_heatmap[b], (batch_segments.shape[1:]))
            cur_heatmap = np.maximum(cur_heatmap, 0.0)
            minh = np.min(cur_heatmap)
            maxh = np.max(cur_heatmap)
            cur_heatmap = (cur_heatmap - minh) / (maxh - minh)
            cur_batch_vis_views[b, :, :, :] = batch_segments[b, :, :, :] * cur_heatmap

        [cur_batch_descriptor] = sess.run(
                [cnn_descriptor],
                feed_dict={
                    cnn_input: batch_segments,
                    cnn_input_vis: cur_batch_vis_views,
                    cnn_y_true: batch_classes,
                    cnn_scales: batch_scales,
                },
        )
        weights[:, layer] = np.linalg.norm(batch_descriptor - cur_batch_descriptor, axis=-1)

    weights = 1.0 / np.maximum(weights, 1e-4)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    heatmap = batch_conv * weights[:, np.newaxis, np.newaxis, np.newaxis, :]
    heatmap = np.sum(heatmap, axis=-1)
    vox_heatmap_np = np.zeros(batch_segments.shape + (3,), dtype=np.uint8)
    for b in range(heatmap.shape[0]):
        # x_min = min(source_segment[:, 0].min(), target_segment[:, 0].min())
        # x_max = max(source_segment[:, 0].max(), target_segment[:, 0].max())
        # y_min = min(source_segment[:, 1].min(), target_segment[:, 1].min())
        # y_max = max(source_segment[:, 1].max(), target_segment[:, 1].max())
        # z_min = min(source_segment[:, 2].min(), target_segment[:, 2].min())
        # z_max = max(source_segment[:, 2].max(), target_segment[:, 2].max())
        #
        # fig = plt.figure(1)
        # canvas = FigureCanvas(fig)
        # ax.scatter(
        #         target_segment[:, 0],
        #         target_segment[:, 1],
        #         target_segment[:, 2],
        #         color="blue",
        #         marker=".",
        # )
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        # ax.set_zlim(z_min, z_max)
        #
        # plt.draw()

    return vox_heatmap_np