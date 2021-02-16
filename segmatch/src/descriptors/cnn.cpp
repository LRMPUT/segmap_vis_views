#include "segmatch/descriptors/cnn.hpp"

#include <algorithm>
#include <math.h>
#include <stdlib.h> /* system, NULL, EXIT_FAILURE */
#include <string>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <laser_slam/benchmarker.hpp>

#include "segmatch/database.hpp"
#include "segmatch/utilities.hpp"

#include <cstdio>

namespace segmatch {

void CNNDescriptor::describe(const Segment& segment, Features* features) {
  CHECK(false) << "Not implemented";
}

template <typename T>
std::vector<size_t> getIndexesInDecreasingOrdering(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

void CNNDescriptor::describe(SegmentedCloud* segmented_cloud_ptr) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  constexpr double kMinChangeBeforeDescription = 0.1; // 0.2

  BENCHMARK_START("SM.Worker.Describe.Preprocess");
  BENCHMARK_RECORD_VALUE("SM.Worker.Describe.NumSegmentsTotal",
                         segmented_cloud_ptr->getNumberOfValidSegments());

  std::vector<tf_graph_executor::Array3D> batch_nn_input;
  std::vector<tf_graph_executor::Array3D> batch_nn_input_vis;
  std::vector<Id> described_segment_ids;
  std::vector<PclPoint> scales;
  std::vector<PclPoint> thresholded_scales;
  std::vector<std::vector<float> > scales_as_vectors;
  std::vector<PclPoint> rescaled_point_cloud_centroids;
  std::vector<PclPoint> point_mins;
  std::vector<double> alignments_rad;
  std::vector<size_t> nums_occupied_voxels;

  for (std::unordered_map<Id, Segment>::iterator it = segmented_cloud_ptr->begin();
      it != segmented_cloud_ptr->end(); ++it) {

    const PointCloud& point_cloud = it->second.getLastView().point_cloud;
    const size_t num_points = point_cloud.size();

    // Skip describing the segment if it did not change enough.
    if (static_cast<double>(num_points) < static_cast<double>(
        it->second.getLastView().n_points_when_last_described) *
        (1.0 + kMinChangeBeforeDescription)) continue;

    if (params_.use_vis_views && it->second.bestViewPts < 50) {
      continue;
    }

    described_segment_ids.push_back(it->second.segment_id);

    // Align with PCA.
    double alignment_rad;
    Eigen::Vector4f pca_centroid;
    pcl::compute3DCentroid(point_cloud, pca_centroid);
    Eigen::Matrix3f covariance_3d;
    computeCovarianceMatrixNormalized(point_cloud, pca_centroid, covariance_3d);
    const Eigen::Matrix2f covariance_2d = covariance_3d.block(0, 0, 2u, 2u);
    Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver(covariance_2d, true);

    alignment_rad = atan2(eigen_solver.eigenvectors()(1,0).real(),
                          eigen_solver.eigenvectors()(0,0).real());

    if (eigen_solver.eigenvalues()(0).real() <
        eigen_solver.eigenvalues()(1).real()) {
      alignment_rad += 0.5*M_PI;
    }

    // Rotate the segment.
    alignment_rad = -alignment_rad;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(alignment_rad, Eigen::Vector3f::UnitZ()));
    PointCloud rotated_point_cloud;
    pcl::transformPointCloud(point_cloud, rotated_point_cloud, transform);

    // Get most points on the lower half of y axis (by rotation).
    PclPoint point_min, point_max;
    pcl::getMinMax3D(rotated_point_cloud, point_min, point_max);
    double centroid_y = point_min.y + (point_max.y - point_min.y) / 2.0;
    unsigned int n_below = 0;
    for (const auto& point : rotated_point_cloud.points) {
      if (point.y < centroid_y) ++n_below;
    }
    if (static_cast<double>(n_below) < static_cast<double>(rotated_point_cloud.size()) / 2.0) {
      alignment_rad += M_PI;
      Eigen::Affine3f adjustment_transform = Eigen::Affine3f::Identity();
      adjustment_transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
      pcl::transformPointCloud(rotated_point_cloud, rotated_point_cloud, adjustment_transform);
    }
    
    alignments_rad.push_back(alignment_rad);

    if (save_debug_data_) {
      Segment aligned_segment = it->second;
      aligned_segment.getLastView().point_cloud = rotated_point_cloud;
      aligned_segments_.addValidSegment(aligned_segment);
    }

    PointCloud rescaled_point_cloud;
    pcl::getMinMax3D(rotated_point_cloud, point_min, point_max);
    point_mins.push_back(point_min);

    // "Fit scaling" using the largest dimension as scale.
    PclPoint scale;
    scale.x = point_max.x - point_min.x;
    scale.y = point_max.y - point_min.y;
    scale.z = point_max.z - point_min.z;
    scales.push_back(scale);
    
    PclPoint thresholded_scale;
    thresholded_scale.x = std::max(scale.x, min_x_scale_m_);
    thresholded_scale.y = std::max(scale.y, min_y_scale_m_);
    thresholded_scale.z = std::max(scale.z, min_z_scale_m_);
    thresholded_scales.push_back(thresholded_scale);
    
    std::vector<float> scales_as_vector;
    scales_as_vector.push_back(scale.x);
    scales_as_vector.push_back(scale.y);
    scales_as_vector.push_back(scale.z);
    scales_as_vectors.push_back(scales_as_vector);

    for (const auto& point: rotated_point_cloud.points) {
      PclPoint point_new;

      point_new.x = (point.x - point_min.x) / thresholded_scale.x
          * static_cast<float>(n_voxels_x_dim_ - 1u);
      point_new.y = (point.y - point_min.y) / thresholded_scale.y
          * static_cast<float>(n_voxels_y_dim_ - 1u);
      point_new.z = (point.z - point_min.z) / thresholded_scale.z
          * static_cast<float>(n_voxels_z_dim_ - 1u);

      rescaled_point_cloud.points.push_back(point_new);
    }
    rescaled_point_cloud.width = 1;
    rescaled_point_cloud.height = rescaled_point_cloud.points.size();

    PclPoint centroid = calculateCentroid(rescaled_point_cloud);
    rescaled_point_cloud_centroids.push_back(centroid);

    unsigned int n_occupied_voxels = 0;
    tf_graph_executor::Array3D nn_input(n_voxels_x_dim_, n_voxels_y_dim_, n_voxels_z_dim_);
    for (const auto& point: rescaled_point_cloud.points) {
      const unsigned int ind_x = floor(point.x + static_cast<float>(n_voxels_x_dim_ - 1) / 2.0
                                       - centroid.x);
      const unsigned int ind_y = floor(point.y + static_cast<float>(n_voxels_y_dim_ - 1) / 2.0
                                       - centroid.y);
      const unsigned int ind_z = floor(point.z + static_cast<float>(n_voxels_z_dim_ - 1) / 2.0
                                       - centroid.z);

      if (ind_x >= 0 && ind_x < n_voxels_x_dim_ &&
          ind_y >= 0 && ind_y < n_voxels_y_dim_ &&
          ind_z >= 0 && ind_z < n_voxels_z_dim_){

        if (nn_input.container[ind_x][ind_y][ind_z] == 0.0) {
          ++n_occupied_voxels;
        }
        nn_input.container[ind_x][ind_y][ind_z] = 1.0;
      }
    }
    nums_occupied_voxels.push_back(n_occupied_voxels);

    it->second.getLastView().n_occupied_voxels = n_occupied_voxels;
    it->second.getLastView().n_points_when_last_described = num_points;

    batch_nn_input.push_back(nn_input);

    tf_graph_executor::Array3D nn_input_vis(n_vis_h_dim_, n_vis_w_dim_, n_vis_c_dim_);

    if (params_.use_vis_views) {
      const auto &visViews = segmented_cloud_ptr->getVisViews();

      // LOG(INFO) << "it->second.bestViewTs = " << it->second.bestViewTs;
      int bestV = -1;
      for (int v = 0; v < visViews.size(); ++v) {
        // LOG(INFO) << "visViews[v].getTime() = " << visViews[v].getTime();
        if (it->second.bestViewTs == visViews[v].getTime()) {
          bestV = v;
          break;
        }
      }
      if (bestV == -1) {
        LOG(INFO) << "Found bestV == -1 ";
        LOG(INFO) << "it->second.segment_id = " << it->second.segment_id;
        LOG(INFO) << "it->second.getLastView().timestamp_ns = " << it->second.getLastView().timestamp_ns;
        for (const auto &view : it->second.views) {
          LOG(INFO) << "view.timestamp_ns = " << view.timestamp_ns;
        }
        LOG(INFO) << "it->second.bestViewPts = " << it->second.bestViewPts;
        LOG(INFO) << "it->second.bestViewTs = " << it->second.bestViewTs;
        for (int v = 0; v < visViews.size(); ++v) {
          LOG(INFO) << "visViews[v].getTime() = " << visViews[v].getTime();
        }
      }
      CHECK_GE(bestV, 0);
      // Could be precomputed, but would need extra memory
      // laser_slam_ros::VisualView::MatrixInt mask = visViews[bestV].getMask(it->second.getLastView().point_cloud);
      laser_slam_ros::VisualView::MatrixInt mask = it->second.bestMask;
      const laser_slam_ros::VisualView::Matrix &intensity = visViews[bestV].getIntensity();
      const laser_slam_ros::VisualView::Matrix &range = visViews[bestV].getRange();
      float meanMaskRange = 0.0f;
      int maskRangeCnt = 0;
      for (int r = 0; r < n_vis_h_dim_; ++r) {
        for (int c = 0; c < n_vis_w_dim_; ++c) {
          if (mask(r, c) > 0) {
            meanMaskRange += range(r, c);
            maskRangeCnt += 1;
            mask(r, c) = 1.0;
          }
        }
      }
      if (maskRangeCnt < it->second.bestViewPts - 20) {
        LOG(INFO) << "\n\n\n\n\nmaskRangeCnt = " << maskRangeCnt;
        LOG(INFO) << "it->second.bestViewPts = " << it->second.bestViewPts;
        LOG(INFO) << "it->second.bestViewTs = " << it->second.bestViewTs;
        LOG(INFO) << "it->second.getLastView().point_cloud.size() = " << it->second.getLastView().point_cloud.size() << "\n\n\n\n\n";
      }
      CHECK_GT(maskRangeCnt, 0);
      meanMaskRange /= maskRangeCnt;
      // {
      //   auto intensityScaled = (intensity.array() - 209.30) / 173.09;
      //   auto rangeScaled = (range.array() - meanMaskRange) * 500.0 / (7632.0);
      //   LOG(INFO) << "int mean = " << intensityScaled.mean();
      //   LOG(INFO) << "int stddev = " << std::sqrt((intensityScaled - intensityScaled.mean()).square().mean());
      //   LOG(INFO) << "range mean = " << rangeScaled.mean();
      //   LOG(INFO) << "range stddev = " << std::sqrt((rangeScaled - rangeScaled.mean()).square().mean());
      //   // LOG(INFO) << "mask(r, c) = \n" << mask;
      // }
      for (int r = 0; r < n_vis_h_dim_; ++r) {
        for (int c = 0; c < n_vis_w_dim_; ++c) {
          // MulRan
          nn_input_vis.container[r][c][0] = (intensity(r, c) - 209.30) / 173.09;
          nn_input_vis.container[r][c][1] = mask(r, c);
          nn_input_vis.container[r][c][2] = (range(r, c) - meanMaskRange) * 500.0 / (7632.0);
        }
      }

      if (true) {
        std::string dir("/tmp/online_matcher/debug");
        // const laser_slam_ros::VisualView::Matrix &intensity = visViews[bestV].getIntensity();
        // const laser_slam_ros::VisualView::Matrix &range = visViews[bestV].getRange();
        // const laser_slam_ros::VisualView::MatrixInt &mask = vis_view.getMask(segment_view.point_cloud);

        std::string segmentDir = dir;
        {
          char dirname[100];
          sprintf(dirname, "%06ld", it->second.segment_id);
          segmentDir = (boost::filesystem::path(dir) / std::string(dirname)).string();
          database::ensureDirectoryExists(segmentDir);
        }

        cv::Mat intensityMat(intensity.rows(), intensity.cols(), CV_16UC1, cv::Scalar(0));
        cv::Mat rangeMat(range.rows(), range.cols(), CV_16UC1, cv::Scalar(0));
        // cv::Mat intensityMono(intensity.rows(), intensity.cols(), CV_8UC1, cv::Scalar(0));
        cv::Mat maskMat(mask.rows(), mask.cols(), CV_8UC1, cv::Scalar(0));
        for (int r = 0; r < intensity.rows(); ++r) {
          for (int c = 0; c < intensity.cols(); ++c) {
            // KITTI
            // intensityMat.at<uint16_t>(r, c) = intensity(r, c)*65535.0f;
            // MulRan
            intensityMat.at<uint16_t>(r, c) = intensity(r, c);
            // intensityMono.at<uint8_t>(r, c) = std::min((int)(intensity(r, c)*255.0/1500.0), 255);
            // up to 65.535 * 2 m
            rangeMat.at<uint16_t>(r, c) = std::min(range(r, c) * 500.0f, 65535.0f);
          }
        }
        for (int r = 0; r < mask.rows(); ++r) {
          for (int c = 0; c < mask.cols(); ++c) {
            if (mask(r, c) > 0) {
              maskMat.at<uint8_t>(r, c) = 255;
            } else {
              maskMat.at<uint8_t>(r, c) = 0;
            }
          }
        }

        char filename[100];
        // sprintf(filename, "%06ld_%03d", segment_id, view_idx);
        sprintf(filename, "%l_%l", it->second.getLastView().timestamp_ns, it->second.bestViewTs);
        cv::imwrite((boost::filesystem::path(segmentDir) / (filename + std::string("_int.png"))).string(), intensityMat);
        cv::imwrite((boost::filesystem::path(segmentDir) / (filename + std::string("_range.png"))).string(), rangeMat);
        cv::imwrite((boost::filesystem::path(segmentDir) / (filename + std::string("_mask.png"))).string(), maskMat);
      }
    }

    batch_nn_input_vis.push_back(nn_input_vis);
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Describe.NumSegmentsDescribed",
                         batch_nn_input.size());
  BENCHMARK_STOP("SM.Worker.Describe.Preprocess");

  if (!batch_nn_input.empty()) {
    BENCHMARK_START("SM.Worker.Describe.ForwardPass");
    std::vector<std::vector<float> > cnn_descriptors;
    std::vector<tf_graph_executor::Array3D> reconstructions;
    std::vector<std::vector<float> > semantics;
    if (batch_nn_input.size() < mini_batch_size_) {
      if (!params_.use_vis_views) {
        graph_executor_->batchFullForwardPass(batch_nn_input,
                                              kInputTensorName,
                                              scales_as_vectors,
                                              kScalesTensorName,
                                              kFeaturesTensorName,
                                              kReconstructionTensorName,
                                              cnn_descriptors,
                                              reconstructions);
      }
      else {
        // LOG(INFO) << "Executing with visual views batch_nn_input.size() < mini_batch_size_";
        graph_executor_->batchFullForwardPassVisViews(batch_nn_input,
                                                      kInputTensorName,
                                                      batch_nn_input_vis,
                                                      kInputVisTensorName,
                                                      scales_as_vectors,
                                                      kScalesTensorName,
                                                      kFeaturesTensorName,
                                                      kReconstructionTensorName,
                                                      cnn_descriptors,
                                                      reconstructions);
      }

    } else {
      std::vector<tf_graph_executor::Array3D> mini_batch;
      std::vector<tf_graph_executor::Array3D> mini_batch_vis;
      std::vector<std::vector<float> > mini_batch_scales;
      for (size_t i = 0u; i < batch_nn_input.size(); ++i) {
        mini_batch.push_back(batch_nn_input[i]);
        mini_batch_vis.push_back(batch_nn_input_vis[i]);
        mini_batch_scales.push_back(scales_as_vectors[i]);
        if (mini_batch.size() == mini_batch_size_) {
          std::vector<std::vector<float> > mini_batch_cnn_descriptors;
          std::vector<tf_graph_executor::Array3D> mini_batch_reconstructions;

          if (!params_.use_vis_views) {
            graph_executor_->batchFullForwardPass(mini_batch,
                                                  kInputTensorName,
                                                  mini_batch_scales,
                                                  kScalesTensorName,
                                                  kFeaturesTensorName,
                                                  kReconstructionTensorName,
                                                  mini_batch_cnn_descriptors,
                                                  mini_batch_reconstructions);
          }
          else {
            // LOG(INFO) << "Executing with visual views mini_batch.size() == mini_batch_size_";
            graph_executor_->batchFullForwardPassVisViews(mini_batch,
                                                          kInputTensorName,
                                                          mini_batch_vis,
                                                          kInputVisTensorName,
                                                          mini_batch_scales,
                                                          kScalesTensorName,
                                                          kFeaturesTensorName,
                                                          kReconstructionTensorName,
                                                          mini_batch_cnn_descriptors,
                                                          mini_batch_reconstructions);
          }

          cnn_descriptors.insert(cnn_descriptors.end(),
                                 mini_batch_cnn_descriptors.begin(),
                                 mini_batch_cnn_descriptors.end());
          reconstructions.insert(reconstructions.end(),
                                 mini_batch_reconstructions.begin(),
                                 mini_batch_reconstructions.end());
          mini_batch_scales.clear();
          mini_batch.clear();
          mini_batch_vis.clear();
        }
      }
      if (!mini_batch.empty()) {
        std::vector<std::vector<float> > mini_batch_cnn_descriptors;
        std::vector<tf_graph_executor::Array3D> mini_batch_reconstructions;

        if (!params_.use_vis_views) {
          graph_executor_->batchFullForwardPass(mini_batch,
                                                kInputTensorName,
                                                mini_batch_scales,
                                                kScalesTensorName,
                                                kFeaturesTensorName,
                                                kReconstructionTensorName,
                                                mini_batch_cnn_descriptors,
                                                mini_batch_reconstructions);
        }
        else {
          // LOG(INFO) << "Executing with visual views !mini_batch.empty()";
          graph_executor_->batchFullForwardPassVisViews(mini_batch,
                                                        kInputTensorName,
                                                        mini_batch_vis,
                                                        kInputVisTensorName,
                                                        mini_batch_scales,
                                                        kScalesTensorName,
                                                        kFeaturesTensorName,
                                                        kReconstructionTensorName,
                                                        mini_batch_cnn_descriptors,
                                                        mini_batch_reconstructions);
        }

        cnn_descriptors.insert(cnn_descriptors.end(),
                               mini_batch_cnn_descriptors.begin(),
                               mini_batch_cnn_descriptors.end());
        reconstructions.insert(reconstructions.end(),
                               mini_batch_reconstructions.begin(),
                               mini_batch_reconstructions.end());
      }
    }

    // Execute semantics graph.
    semantics = semantics_graph_executor_->batchExecuteGraph(
        cnn_descriptors, kInputTensorName, kSemanticsOutputName);

    CHECK_EQ(cnn_descriptors.size(), described_segment_ids.size());
    BENCHMARK_STOP("SM.Worker.Describe.ForwardPass");


    BENCHMARK_START("SM.Worker.Describe.SaveFeatures");
    // Write the features.
    for (size_t i = 0u; i < described_segment_ids.size(); ++i) {
      Segment* segment;
      CHECK(segmented_cloud_ptr->findValidSegmentPtrById(described_segment_ids[i], &segment));
      std::vector<float> nn_output = cnn_descriptors[i];

      Feature cnn_feature("cnn");
      for (size_t j = 0u; j < nn_output.size(); ++j) {
        cnn_feature.push_back(FeatureValue("cnn_" + std::to_string(j), nn_output[j]));
      }

      // Push the scales.
      cnn_feature.push_back(FeatureValue("cnn_scale_x", scales[i].x));
      cnn_feature.push_back(FeatureValue("cnn_scale_y", scales[i].y));
      cnn_feature.push_back(FeatureValue("cnn_scale_z", scales[i].z));

      segment->getLastView().features.replaceByName(cnn_feature);

      std::vector<float> semantic_nn_output = semantics[i];
      segment->getLastView().semantic = std::distance(semantic_nn_output.begin(),
                                                      std::max_element(semantic_nn_output.begin(),
                                                                       semantic_nn_output.end()));

      // Generate the reconstructions.
      PointCloud reconstruction;
      const double reconstruction_threshold = 0.75;
      const double ratio_voxels_to_reconstruct = 1.5;
      const bool reconstruct_by_probability = true;

      PclPoint point;
      const PclPoint point_min = point_mins[i];
      const PclPoint scale = thresholded_scales[i];
      const PclPoint centroid = rescaled_point_cloud_centroids[i];

      if (reconstruct_by_probability) {
        for (unsigned int x = 0u; x < n_voxels_x_dim_; ++x) {
          for (unsigned int y = 0u; y < n_voxels_y_dim_; ++y) {
            for (unsigned int z = 0u; z < n_voxels_z_dim_; ++z) { 
                if (reconstructions[i].container[x][y][z] >= reconstruction_threshold) {
                  point.x = point_min.x + scale.x * 
                    (static_cast<float>(x) - x_dim_min_1_ / 2.0 + centroid.x) / x_dim_min_1_;
                  point.y = point_min.y + scale.y * 
                    (static_cast<float>(y) - y_dim_min_1_ / 2.0 + centroid.y) / y_dim_min_1_;
                  point.z = point_min.z + scale.z * 
                    (static_cast<float>(z) - z_dim_min_1_ / 2.0 + centroid.z) / z_dim_min_1_;
                  reconstruction.points.push_back(point);
                }
            }
          }
        }
      } else {
                const unsigned int n_voxels_in_original_segment = nums_occupied_voxels[i];
      const unsigned int n_voxels_in_reconstructed_segment =
        floor(double(n_voxels_in_original_segment) * ratio_voxels_to_reconstruct);
      
      // Order by occupancy probability.
      std::vector<double> probs;
      std::vector<PclPoint> indices;
      for (unsigned int x = 0u; x < n_voxels_x_dim_; ++x) {
        for (unsigned int y = 0u; y < n_voxels_y_dim_; ++y) {
          for (unsigned int z = 0u; z < n_voxels_z_dim_; ++z) { 
              probs.push_back(reconstructions[i].container[x][y][z]);
              PclPoint indice;
              indice.x = x;
              indice.y = y;
              indice.z = z;
              indices.push_back(indice);
          }
        }
      }
      std::vector<size_t> indexes_in_decreasing_order = getIndexesInDecreasingOrdering(probs);
      
      for (unsigned int j = 0u; j < n_voxels_in_reconstructed_segment; ++j){
          point.x = point_min.x + scale.x * 
          (static_cast<float>(indices[indexes_in_decreasing_order[j]].x) - x_dim_min_1_ / 2.0 + centroid.x) / x_dim_min_1_;
          point.y = point_min.y + scale.y * 
          (static_cast<float>(indices[indexes_in_decreasing_order[j]].y) - y_dim_min_1_ / 2.0 + centroid.y) / y_dim_min_1_;
          point.z = point_min.z + scale.z * 
          (static_cast<float>(indices[indexes_in_decreasing_order[j]].z) - z_dim_min_1_ / 2.0 + centroid.z) / z_dim_min_1_;
          reconstruction.points.push_back(point);
      }
      }

      reconstruction.width = 1;
      reconstruction.height = reconstruction.points.size();

      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.rotate(Eigen::AngleAxisf(-alignments_rad[i], Eigen::Vector3f::UnitZ()));
      pcl::transformPointCloud(reconstruction, reconstruction, transform);
      segment->getLastView().reconstruction = reconstruction;
      
      // TODO RD remove if compressing reconstruction not needed.
      /*unsigned int publish_every_x_points = 5;
      segment->getLastView().reconstruction_compressed.clear();
      segment->getLastView().reconstruction_compressed.reserve(reconstruction.points.size() / publish_every_x_points);
      unsigned int z = 0;
      for (const auto& point : reconstruction.points) {
        if (z % publish_every_x_points == 0) {
            segment->getLastView().reconstruction_compressed.points.emplace_back(point.x, point.y, point.z);
        }
        ++z;
      }*/
    }
    BENCHMARK_STOP("SM.Worker.Describe.SaveFeatures");
  }
}

void CNNDescriptor::exportData() const {
  if (save_debug_data_) {
    database::exportSegmentsAndFeatures("/tmp/aligned_segments",
                                        aligned_segments_, true);
  }
}

} // namespace segmatch
