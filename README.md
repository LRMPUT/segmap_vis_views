## SegMap with visual views 

This an extended version of SegMap system that uses intensity images synthesized from LiDAR measurement to improve segment description.

### Related publications

J. Wietrzykowski, P. Skrzypczyński, On the descriptive power of LiDAR intensity images for segment-based loop closing in 3-D SLAM, 2021


### Installation

For installation instructions consult the original [SegMap repository](https://github.com/ethz-asl/segmap).

### Error *libfontconfig.so.1: undefined symbol: FT_Done_MM_Var* when using Conda

Modify CMakeLists.txt:
```
target_link_libraries(segmapper_node /opt/conda/lib/libfreetype.so.6 fontconfig harfbuzz ${PROJECT_NAME})