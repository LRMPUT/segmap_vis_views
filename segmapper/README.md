SegMapper
==================

A node for performing multi-robot LiDAR based pose-graph SLAM using *SegMap*.

Launching:

```
LD_LIBRARY_PATH=$HOME/anaconda3/envs/tf_gpu_python2/lib:$LD_LIBRARY_PATH roslaunch segmapper cnn_loam_loop_closure.launch
```