import numpy as np
import cv2
from scipy.interpolate import interpn
from scipy.ndimage import zoom
import matplotlib as mpl
mpl.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_heatmap_img(batch_segments,
                     batch_vis_views,
                     batch_scales,
                     batch_conv_vis,
                     batch_descriptor,
                     sess,
                     cnn_descriptor,
                     cnn_input,
                     cnn_input_vis,
                     cnn_scales
                     ):

    # MulRan
    intensity = batch_vis_views[:, :, :, 0] * 173.09 + 209.30
    # KITTI
    # intensity = (batch_vis_views[:, :, :, 0] * 8297.86 + 19020.73) * 1500.0 / 65535.0
    mask = batch_vis_views[:, :, :, 1] * 255.0
    weights = np.zeros((batch_conv_vis.shape[0], batch_conv_vis.shape[3]), dtype=np.float)
    for layer in range(batch_conv_vis.shape[3]):
        l_heatmap = batch_conv_vis[:, :, :, layer]
        cur_batch_vis_views = np.zeros(batch_vis_views.shape, dtype=batch_vis_views.dtype)
        for b in range(l_heatmap.shape[0]):
            cur_heatmap = cv2.resize(l_heatmap[b], (intensity.shape[2], intensity.shape[1]))
            cur_heatmap = np.maximum(cur_heatmap, 0.0)
            minh = np.min(cur_heatmap)
            maxh = np.max(cur_heatmap)
            cur_heatmap = (cur_heatmap - minh) / max(maxh - minh, 1e-4)
            cur_batch_vis_views[b, :, :, :] = batch_vis_views[b, :, :, :] * np.expand_dims(cur_heatmap,
                                                                                           axis=-1)

        [cur_batch_descriptor] = sess.run(
                [cnn_descriptor],
                feed_dict={
                    cnn_input: batch_segments,
                    cnn_input_vis: cur_batch_vis_views,
                    cnn_scales: batch_scales,
                },
        )
        weights[:, layer] = np.linalg.norm(batch_descriptor - cur_batch_descriptor, axis=-1)

    weights = 1.0 / np.maximum(weights, 1e-4)
    scores = np.mean(weights, axis=-1)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    heatmap = batch_conv_vis * weights[:, np.newaxis, np.newaxis, :]
    heatmap = np.sum(heatmap, axis=-1)
    img_heatmap_np = np.zeros(intensity.shape + (3,), dtype=np.uint8)
    val_heatmap_np = np.zeros(intensity.shape, dtype=np.float)
    for b in range(heatmap.shape[0]):
        cur_heatmap = cv2.resize(heatmap[b], (intensity.shape[2], intensity.shape[1]))
        cur_heatmap = np.maximum(cur_heatmap, 0.0)
        cur_heatmap = cur_heatmap / np.max(cur_heatmap)
        cur_heatmap_color = cv2.applyColorMap(np.uint8(255 * cur_heatmap), cv2.COLORMAP_JET)
        cur_heatmap_color = cv2.cvtColor(cur_heatmap_color, cv2.COLOR_BGR2RGB)
        cur_img_heatmap = 0.25 * cur_heatmap_color + \
                          0.75 * np.tile(np.expand_dims(np.minimum(intensity[b] * 255.0 / 1500.0, 255.0), axis=-1),
                                  [1, 1, 3])

        cur_img_heatmap[mask[b] > 128.0, :] = np.array([255, 0, 0], dtype=np.uint8)
        img_heatmap_np[b] = cur_img_heatmap
        val_heatmap_np[b] = cur_heatmap

    return img_heatmap_np, val_heatmap_np, scores


def resize_voxels(voxels, size):
    # x = np.linspace(0, voxels.shape[0] - 1, num=voxels.shape[0])
    # y = np.linspace(0, voxels.shape[1] - 1, num=voxels.shape[1])
    # z = np.linspace(0, voxels.shape[2] - 1, num=voxels.shape[2])
    voxels_resized = zoom(voxels, (size[0] / voxels.shape[0],
                                   size[1] / voxels.shape[1],
                                   size[2] / voxels.shape[2]))

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
                     cnn_scales,
                     height,
                     width
                     ):

    weights = np.zeros((batch_conv.shape[0], batch_conv.shape[-1]), dtype=np.float)
    for layer in range(batch_conv.shape[-1]):
        l_heatmap = batch_conv[:, :, :, :, layer]
        cur_batch_segments = np.zeros(batch_segments.shape, dtype=batch_segments.dtype)
        for b in range(l_heatmap.shape[0]):
            cur_heatmap = resize_voxels(l_heatmap[b], (batch_segments.shape[1:-1]))
            cur_heatmap = np.maximum(cur_heatmap, 0.0)
            minh = np.min(cur_heatmap)
            maxh = np.max(cur_heatmap)
            cur_heatmap = (cur_heatmap - minh) / max(maxh - minh, 1e-4)
            cur_batch_segments[b, :, :, :, :] = batch_segments[b, :, :, :, :] * np.expand_dims(cur_heatmap, axis=-1)

        [cur_batch_descriptor] = sess.run(
                [cnn_descriptor],
                feed_dict={
                    cnn_input: cur_batch_segments,
                    cnn_input_vis: batch_vis_views,
                    cnn_y_true: batch_classes,
                    cnn_scales: batch_scales,
                },
        )
        weights[:, layer] = np.linalg.norm(batch_descriptor - cur_batch_descriptor, axis=-1)

    weights = 1.0 / np.maximum(weights, 1e-4)
    scores = np.mean(weights, axis=-1)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    heatmap = batch_conv * weights[:, np.newaxis, np.newaxis, np.newaxis, :]
    heatmap = np.sum(heatmap, axis=-1)
    vox_heatmap_np = []

    x_min = 0
    x_max = batch_segments.shape[1]
    y_min = 0
    y_max = batch_segments.shape[2]
    z_min = 0
    z_max = batch_segments.shape[3]

    xs = np.linspace(0, batch_segments.shape[1] - 1, batch_segments.shape[1])
    ys = np.linspace(0, batch_segments.shape[2] - 1, batch_segments.shape[2])
    zs = np.linspace(0, batch_segments.shape[3] - 1, batch_segments.shape[3])
    xv, yv, zv = np.meshgrid(xs, ys, zs)
    for b in range(heatmap.shape[0]):
        cur_heatmap = resize_voxels(heatmap[b], (batch_segments.shape[1:-1]))

        dpi = 160
        fig = plt.figure(1, figsize=(width/dpi, height/dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')

        cur_xv = xv[batch_segments[b, :, :, :, 0] > 0.5]
        cur_yv = yv[batch_segments[b, :, :, :, 0] > 0.5]
        cur_zv = zv[batch_segments[b, :, :, :, 0] > 0.5]
        ax.scatter(
                cur_xv.reshape(-1),
                cur_yv.reshape(-1),
                cur_zv.reshape(-1),
                color='black',
                s=9,
                marker=".",
        )
        ax.scatter(
                xv.reshape(-1),
                yv.reshape(-1),
                zv.reshape(-1),
                c=cur_heatmap.reshape(-1),
                s=4,
                cmap='jet',
                alpha=0.2,
                marker=".",
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        canvas.draw()
        # plt.draw()
        vox_heatmap_np.append(np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3)))

    return np.array(vox_heatmap_np), scores
