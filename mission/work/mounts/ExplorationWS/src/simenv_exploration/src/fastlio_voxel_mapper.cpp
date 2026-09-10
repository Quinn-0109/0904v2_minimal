#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <octomap/OcTree.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <simenv_exploration/CheckTwinCylinder.h>

namespace {

bool finite3(double x, double y, double z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (const char c : value) {
    if (c == '\\' || c == '"') out << '\\';
    out << c;
  }
  return out.str();
}

uint64_t saturatedMultiply(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a)
    return std::numeric_limits<uint64_t>::max();
  return a * b;
}

}  // namespace

class FastlioVoxelMapper {
 public:
  FastlioVoxelMapper()
      : nh_(), private_nh_("~"), stabilized_odom_nh_(nh_),
        tree_(readResolution()) {
    private_nh_.param<std::string>("cloud_topic", cloud_topic_, "/cloud_registered");
    private_nh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    private_nh_.param<std::string>("output_dir", output_dir_, "results/latest");
    private_nh_.param<double>("maximum_range", maximum_range_, 30.0);
    private_nh_.param<double>("minimum_range", minimum_range_, 0.30);
    private_nh_.param<double>("self_filter_radius", self_filter_radius_, 0.0);
    private_nh_.param<double>("self_filter_min_relative_z",
                              self_filter_min_relative_z_, -0.75);
    private_nh_.param<double>("self_filter_max_relative_z",
                              self_filter_max_relative_z_, -0.08);
    private_nh_.param<int>("maximum_scan_points", maximum_scan_points_, 12000);
    private_nh_.param<int>("save_every_updates", save_every_updates_, 20);
    private_nh_.param<double>("hit_probability", hit_probability_, 0.70);
    private_nh_.param<double>("miss_probability", miss_probability_, 0.40);
    private_nh_.param<double>("clamping_min", clamping_min_, 0.12);
    private_nh_.param<double>("clamping_max", clamping_max_, 0.97);
    private_nh_.param<double>("occupancy_threshold", occupancy_threshold_, 0.50);
    private_nh_.param<std::string>("projection_topic", projection_topic_,
                                   "/simenv/voxel_floor_projection");
    private_nh_.param<int>("projection_every_updates", projection_every_updates_, 5);
    private_nh_.param<double>("projection_min_height", projection_min_height_, -0.05);
    private_nh_.param<double>("projection_max_height", projection_max_height_, 1.20);
    private_nh_.param<double>("telemetry_map_rate_hz", telemetry_map_rate_hz_, 0.5);
    private_nh_.param<double>("map_statistics_warn_ms", map_statistics_warn_ms_, 50.0);
    private_nh_.param<double>("maximum_odom_cloud_sync_error",
                              maximum_odom_cloud_sync_error_, 0.20);
    private_nh_.param<bool>("planar_height_stabilization_enabled",
                            planar_height_stabilization_enabled_, false);
    private_nh_.param<bool>("collision_request_pose_is_stabilized",
                            collision_request_pose_is_stabilized_, false);
    private_nh_.param<std::string>("stabilized_odom_topic",
                                   stabilized_odom_topic_, "");
 
    tree_.setProbHit(hit_probability_);
    tree_.setProbMiss(miss_probability_);
    tree_.setClampingThresMin(clamping_min_);
    tree_.setClampingThresMax(clamping_max_);
    tree_.setOccupancyThres(occupancy_threshold_);

    bt_path_ = output_dir_ + "/voxel_map.bt";
    stats_path_ = output_dir_ + "/map_statistics.json";
    map_growth_path_ = output_dir_ + "/map_growth_timeseries.jsonl";
    std::ofstream(map_growth_path_, std::ios::trunc).close();
    cloud_sub_ = nh_.subscribe(cloud_topic_, 2, &FastlioVoxelMapper::cloudCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 50, &FastlioVoxelMapper::odomCallback, this);
    done_sub_ = nh_.subscribe("/simenv/mission_complete", 1,
                              &FastlioVoxelMapper::doneCallback, this);
    finalize_sub_ = nh_.subscribe("/simenv/finalize_voxel_map", 1,
                                  &FastlioVoxelMapper::finalizeCallback, this);
    active_goal_sub_ = nh_.subscribe("/simenv/active_goal_id", 5,
                                     &FastlioVoxelMapper::activeGoalCallback, this);
    stair_state_sub_ = nh_.subscribe("/simenv/stair_transition_state", 10,
                                     &FastlioVoxelMapper::transitionStateCallback,
                                     this);
    elevator_state_sub_ = nh_.subscribe("/simenv/elevator_transition_state", 10,
                                        &FastlioVoxelMapper::transitionStateCallback,
                                        this);
    second_floor_state_sub_ = nh_.subscribe(
        "/simenv/second_floor_state", 10,
        &FastlioVoxelMapper::secondFloorStateCallback, this);
    upper_floor_stair_state_sub_ = nh_.subscribe(
        "/simenv/second_to_third_floor_stair_state", 10,
        &FastlioVoxelMapper::upperFloorTransitionStateCallback, this);
    third_floor_state_sub_ = nh_.subscribe(
        "/simenv/third_floor_state", 10,
        &FastlioVoxelMapper::thirdFloorStateCallback, this);
    projection_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(projection_topic_, 1, true);
    saved_pub_ = nh_.advertise<std_msgs::Bool>("/simenv/voxel_map_saved", 1, true);
    statistics_duration_pub_ = nh_.advertise<std_msgs::Float64>(
        "/simenv/map_statistics_duration_ms", 5);
    if (!stabilized_odom_topic_.empty()) {
      stabilized_odom_pub_ = nh_.advertise<nav_msgs::Odometry>(
          stabilized_odom_topic_, 50);
      // OctoMap insertion and projection can take longer than the navigation
      // watchdog permits.  Relay x/y/yaw from raw FAST-LIO on an independent
      // callback queue so map work can never make a healthy pose look stale.
      // Only the scalar planar-z correction is shared atomically with the
      // mapper; the room, corridor and stair planners remain unchanged.
      stabilized_odom_nh_.setCallbackQueue(&stabilized_odom_queue_);
      stabilized_odom_sub_ = stabilized_odom_nh_.subscribe(
          odom_topic_, 50, &FastlioVoxelMapper::stabilizedOdomCallback, this);
      stabilized_odom_spinner_.reset(
          new ros::AsyncSpinner(1, &stabilized_odom_queue_));
      stabilized_odom_spinner_->start();
    }
    collision_service_ = private_nh_.advertiseService(
        "check_twin_cylinder", &FastlioVoxelMapper::checkTwinCylinder, this);
    if (telemetry_map_rate_hz_ > 0.0)
      map_timer_ = nh_.createTimer(ros::Duration(1.0 / telemetry_map_rate_hz_),
                                  &FastlioVoxelMapper::mapTelemetryCallback, this);
    started_ros_ = ros::Time::now();
    started_wall_ = ros::WallTime::now();
    ROS_INFO_STREAM("FAST-LIO voxel mapper ready: cloud=" << cloud_topic_
                    << " odom=" << odom_topic_ << " resolution="
                    << tree_.getResolution() << " m output=" << output_dir_);
  }

  ~FastlioVoxelMapper() {
    if (stabilized_odom_spinner_) stabilized_odom_spinner_->stop();
    save("ros_shutdown");
  }

 private:
  double readResolution() {
    double resolution = 0.15;
    private_nh_.param<double>("resolution", resolution, 0.15);
    if (resolution < 0.10 || resolution > 0.20) {
      ROS_WARN_STREAM("Requested resolution " << resolution
                      << " is outside 0.10..0.20 m; clamping it");
      resolution = std::max(0.10, std::min(0.20, resolution));
    }
    return resolution;
  }

  void transitionStateCallback(const std_msgs::StringConstPtr& message) {
    const std::string& phase = message->data;
    const bool vertical_motion =
        phase.find("STAIR_ASCENT") != std::string::npos ||
        phase.find("STAIR_LANDING") != std::string::npos ||
        phase.find("SECOND_FLOOR") != std::string::npos ||
        phase.find("ELEVATOR_MOVING") != std::string::npos ||
        phase.find("ELEVATOR_ASCENT") != std::string::npos;
    if (!vertical_motion) return;
    std::lock_guard<std::mutex> lock(mutex_);
    // Once the F2 corridor callback has installed a new planar reference,
    // late stair completion/status messages are no longer vertical motion.
    // Run46 received a SECOND_FLOOR_* handoff state after re-anchoring; that
    // re-froze compensation and allowed raw FAST-LIO z to fall from 2.8 m to
    // 1.4 m while Gazebo truth remained at 2.92 m.  Ignore such late states
    // only after the F2 re-anchor.  F1 and both physical stair flights still
    // execute the unchanged freeze path above.
    if (second_floor_planar_reanchored_) return;
    if (vertical_transition_started_) return;
    vertical_transition_started_ = true;
    frozen_planar_height_offset_ = current_planar_height_offset_;
    ROS_INFO("Voxel mapper froze planar z-drift compensation at %.3f m "
             "for vertical transition phase %s",
             frozen_planar_height_offset_, phase.c_str());
  }

  void secondFloorStateCallback(const std_msgs::StringConstPtr& message) {
    // Stair motion freezes F1 compensation so the real ascent is retained.
    // At the F2 corridor entrance, establish a new planar reference from the
    // next raw odometry sample.  Subsequent LIO z drift is then removed while
    // the already accumulated floor-to-floor elevation stays in the map.
    if (message->data != "SECOND_FLOOR_CORRIDOR_ENTRY_REACHED" ||
        !planar_height_stabilization_enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (second_floor_planar_reanchored_ ||
        second_floor_planar_reanchor_pending_) return;
    second_floor_planar_reanchor_pending_ = true;
    // Navigation odometry is relayed on a separate callback queue so map
    // insertion can never make it stale.  Give that fast path its own F2
    // height latch: otherwise a sudden raw-LIO z correction can be published
    // before this slower mapper callback updates navigation_height_offset_.
    // This activates only after the truth-validated upper corridor handoff;
    // F1 and both stair flights retain their original vertical motion.
    second_floor_navigation_height_lock_pending_.store(
        true, std::memory_order_release);
    ROS_INFO("Voxel mapper queued F2 planar z re-anchor at corridor entry");
  }

  // The F2 corridor-entry re-anchor pins the relayed navigation odometry z to
  // the F2 floor with a latch that has no release path.  Once the F2->F3 stair
  // manager starts the upper climb, raw FAST-LIO z legitimately rises ~2.6 m;
  // a consumer of stabilized odometry (the F3 localization stabilization
  // check) must then see the new floor height.  Release the F2 latch and
  // freeze the small F2 residual drift for the climb so the relayed z tracks
  // truth to the third floor.  The F3 corridor-entry re-anchor below then
  // installs the F3 reference.
  void upperFloorTransitionStateCallback(
      const std_msgs::StringConstPtr& message) {
    const std::string& phase = message->data;
    if (phase.find("STAIR_ASCENT") == std::string::npos) return;
    if (upper_floor_climb_started_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    upper_floor_climb_started_ = true;
    second_floor_navigation_height_lock_pending_.store(
        false, std::memory_order_relaxed);
    second_floor_navigation_height_lock_active_.store(
        false, std::memory_order_release);
    vertical_transition_started_ = true;
    frozen_planar_height_offset_ =
        navigation_height_offset_.load(std::memory_order_relaxed);
    ROS_INFO("Voxel mapper released F2 navigation height lock and froze "
             "planar z-drift compensation at %.3f m for F2->F3 stair ascent",
             frozen_planar_height_offset_);
  }

  // Mirror of secondFloorStateCallback for the third floor.  Without an F3
  // re-anchor the map projection would stay pinned ~2.6 m below the robot on
  // F3 and the relayed navigation odometry would keep reporting the F2 height.
  // At the F3 corridor entrance, install a fresh planar reference and re-latch
  // the navigation height to the F3 floor, matching the F2 design.
  //
  // F3 phantom-wall root fix: the F2->F3 climb leaves the odometry frame on
  // the F2 plane until FAST-LIO's truth-motion guard snaps the pose (whole
  // pose x/y/z/yaw) to truth.  Every scan inserted before that snap lives in
  // the old drifted frame and survives forever because the octree is never
  // cleared, so the F3 map mixes two frames ~2.5 m / up to a metre in x/y
  // apart and door openings are filled by phantom walls (A* rejects every
  // entry candidate).  THIRD_FLOOR_LOCALIZATION_STABILIZING is published by
  // the F3 manager when it begins its stationary pose-consistency hold, i.e.
  // right around the snap; queue an octree clear + planar re-anchor that the
  // odometry callback executes as soon as the snap has actually lifted z.
  void thirdFloorStateCallback(const std_msgs::StringConstPtr& message) {
    if (!planar_height_stabilization_enabled_) return;
    if (message->data == "THIRD_FLOOR_LOCALIZATION_STABILIZING") {
      std::lock_guard<std::mutex> lock(mutex_);
      if (third_floor_map_rebuilt_) return;
      third_floor_stabilization_clear_pending_ = true;
      ROS_INFO("Voxel mapper queued F3 octree clear + re-anchor on "
               "THIRD_FLOOR_LOCALIZATION_STABILIZING");
      return;
    }
    if (message->data != "THIRD_FLOOR_CORRIDOR_ENTRY_REACHED") return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (third_floor_planar_reanchored_ || third_floor_planar_reanchor_pending_)
      return;
    third_floor_planar_reanchor_pending_ = true;
    second_floor_navigation_height_lock_pending_.store(
        true, std::memory_order_release);
    ROS_INFO("Voxel mapper queued F3 planar z re-anchor at corridor entry");
  }

  void odomCallback(const nav_msgs::OdometryConstPtr& message) {
    const auto& p = message->pose.pose.position;
    if (!finite3(p.x, p.y, p.z)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (planar_height_stabilization_enabled_ &&
        third_floor_stabilization_clear_pending_) {
      // Fire only once the truth-guard snap has lifted z off the F2 plane
      // (0.75 m matches FAST-LIO's height-recovery threshold); before that,
      // clearing would just re-insert the same drifted scans.  After the
      // clear, every scan is built from the snapped (truth-frame) pose and
      // the F3 map no longer contains pre-snap phantom walls.
      if (p.z - planar_height_reference_ > k_third_floor_clear_z_jump_m) {
        tree_.clear();
        occupied_progress_.clear();
        ROS_WARN("Voxel mapper cleared octree after F3 pose snap "
                 "(odom z %.3f -> %.3f); rebuilding F3 map in truth frame",
                 planar_height_reference_, p.z);
        third_floor_stabilization_clear_pending_ = false;
        third_floor_map_rebuilt_ = true;
        third_floor_planar_reanchored_ = true;
        third_floor_planar_reanchor_pending_ = false;
        planar_height_reference_ = p.z;
        planar_height_reference_valid_ = true;
        current_planar_height_offset_ = 0.0;
        frozen_planar_height_offset_ = 0.0;
        vertical_transition_started_ = false;
        ROS_INFO("Voxel mapper F3 planar height reference re-initialized at "
                 "%.3f m after map rebuild", planar_height_reference_);
      }
    }
    if (planar_height_stabilization_enabled_ &&
        third_floor_planar_reanchor_pending_) {
      planar_height_reference_ = p.z;
      planar_height_reference_valid_ = true;
      current_planar_height_offset_ = 0.0;
      frozen_planar_height_offset_ = 0.0;
      vertical_transition_started_ = false;
      third_floor_planar_reanchor_pending_ = false;
      third_floor_planar_reanchored_ = true;
      ROS_INFO("Voxel mapper F3 planar height reference initialized at %.3f m",
               planar_height_reference_);
    }
    if (planar_height_stabilization_enabled_ &&
        second_floor_planar_reanchor_pending_) {
      planar_height_reference_ = p.z;
      planar_height_reference_valid_ = true;
      current_planar_height_offset_ = 0.0;
      frozen_planar_height_offset_ = 0.0;
      vertical_transition_started_ = false;
      second_floor_planar_reanchor_pending_ = false;
      second_floor_planar_reanchored_ = true;
      ROS_INFO("Voxel mapper F2 planar height reference initialized at %.3f m",
               planar_height_reference_);
    }
    if (planar_height_stabilization_enabled_ &&
        !planar_height_reference_valid_) {
      planar_height_reference_ = p.z;
      planar_height_reference_valid_ = true;
      ROS_INFO("Voxel mapper planar height reference initialized at %.3f m",
               planar_height_reference_);
    }
    double height_offset = 0.0;
    if (planar_height_stabilization_enabled_ &&
        planar_height_reference_valid_) {
      if (vertical_transition_started_) {
        height_offset = frozen_planar_height_offset_;
      } else {
        height_offset = p.z - planar_height_reference_;
        current_planar_height_offset_ = height_offset;
        maximum_planar_height_offset_ = std::max(
            maximum_planar_height_offset_, std::abs(height_offset));
      }
    }
    navigation_height_offset_.store(height_offset,
                                    std::memory_order_relaxed);
    if (have_origin_)
      cumulative_trajectory_length_ += std::hypot(
          p.x - sensor_origin_.x(), p.y - sensor_origin_.y());
    // Keep FAST-LIO's EKF state untouched.  Only the mapper's copy of the
    // sensor origin and registered scan is translated vertically so a planar
    // floor cannot migrate through the projection slice as LIO z drifts.
    sensor_origin_ = octomap::point3d(p.x, p.y, p.z - height_offset);
    odom_frame_ = message->header.frame_id;
    odom_child_frame_ = message->child_frame_id;
    odom_stamp_ = message->header.stamp;
    have_origin_ = true;
    if (!odom_stamp_.isZero()) {
      odom_history_.push_back({odom_stamp_, sensor_origin_, height_offset});
      while (odom_history_.size() > 400) odom_history_.pop_front();
    }
  }

  void stabilizedOdomCallback(const nav_msgs::OdometryConstPtr& message) {
    const auto& p = message->pose.pose.position;
    if (!finite3(p.x, p.y, p.z) || !stabilized_odom_pub_) return;
    nav_msgs::Odometry stabilized = *message;
    if (second_floor_navigation_height_lock_pending_.exchange(
            false, std::memory_order_acq_rel)) {
      second_floor_navigation_height_reference_.store(
          p.z, std::memory_order_relaxed);
      second_floor_navigation_height_lock_active_.store(
          true, std::memory_order_release);
      ROS_INFO("Stabilized odometry locked F2 navigation height at %.3f m",
               p.z);
    }
    if (second_floor_navigation_height_lock_active_.load(
            std::memory_order_acquire)) {
      stabilized.pose.pose.position.z =
          second_floor_navigation_height_reference_.load(
              std::memory_order_relaxed);
      stabilized.twist.twist.linear.z = 0.0;
    } else {
      stabilized.pose.pose.position.z = p.z - navigation_height_offset_.load(
          std::memory_order_relaxed);
    }
    stabilized_odom_pub_.publish(stabilized);
  }

  void activeGoalCallback(const std_msgs::Int32ConstPtr& message) {
    active_goal_id_.store(message->data);
  }

  bool checkTwinCylinder(simenv_exploration::CheckTwinCylinder::Request& request,
                         simenv_exploration::CheckTwinCylinder::Response& response) {
    const double values[] = {request.pose_x, request.pose_y, request.pose_z, request.yaw,
      request.front_offset, request.rear_offset, request.radius,
      request.min_height, request.max_height, request.clearance_search_radius};
    for (double value : values) {
      if (!std::isfinite(value)) {
        response.status = "invalid_geometry";
        return true;
      }
    }
    if (request.radius <= 0.0 || request.min_height >= request.max_height) {
      response.status = "invalid_geometry";
      return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    response.map_available = update_count_.load() > 0;
    response.frame_id = !last_cloud_frame_.empty() ? last_cloud_frame_ : odom_frame_;
    if (!response.map_available) {
      response.status = "map_unavailable";
      return true;
    }
    const double c = std::cos(request.yaw), s = std::sin(request.yaw);
    response.front_center_x = request.pose_x + request.front_offset * c;
    response.front_center_y = request.pose_y + request.front_offset * s;
    response.rear_center_x = request.pose_x + request.rear_offset * c;
    response.rear_center_y = request.pose_y + request.rear_offset * s;
    const double resolution = tree_.getResolution();
    // The full-mission manager consumes ``stabilized_odom_topic`` and sends
    // that already map-aligned z in this request.  Subtracting the raw-LIO
    // drift a second time moved run19's footprint query about 0.21 m below
    // the robot and into the floor, so three valid forward corridor paths
    // were rejected as start collisions.  Keep the legacy raw-odom contract
    // available for other mapper configurations through the explicit flag.
    const double map_pose_z = request.pose_z -
        (collision_request_pose_is_stabilized_
             ? 0.0 : currentMapHeightOffsetLocked());
    const double z_min = map_pose_z + request.min_height;
    const double z_max = map_pose_z + request.max_height;
    const double centers[2][2] = {{response.front_center_x, response.front_center_y},
                                  {response.rear_center_x, response.rear_center_y}};
    double minimum_clearance = std::numeric_limits<double>::infinity();
    for (const auto& center : centers) {
      const int horizontal = static_cast<int>(std::ceil(request.radius / resolution));
      for (int iy = -horizontal; iy <= horizontal; ++iy) {
        for (int ix = -horizontal; ix <= horizontal; ++ix) {
          const double dx = ix * resolution, dy = iy * resolution;
          if (dx * dx + dy * dy > request.radius * request.radius + 1e-9) continue;
          for (double z = z_min; z <= z_max + 1e-9; z += resolution) {
            auto* node = tree_.search(center[0] + dx, center[1] + dy, z);
            if (!node) {
              ++response.unknown_queries;
            } else if (tree_.isNodeOccupied(node)) {
              ++response.occupied_queries;
              response.occupied_collision = true;
            }
          }
        }
      }
      const double search_radius = std::max(request.radius,
                                             request.clearance_search_radius);
      for (auto it = tree_.begin_leafs_bbx(
               octomap::point3d(center[0] - search_radius, center[1] - search_radius, z_min),
               octomap::point3d(center[0] + search_radius, center[1] + search_radius, z_max));
           it != tree_.end_leafs_bbx(); ++it) {
        if (!tree_.isNodeOccupied(*it)) continue;
        const double half = it.getSize() * 0.5;
        const double dx = std::max(0.0, std::abs(it.getX() - center[0]) - half);
        const double dy = std::max(0.0, std::abs(it.getY() - center[1]) - half);
        minimum_clearance = std::min(
            minimum_clearance, std::max(0.0, std::hypot(dx, dy) - request.radius));
      }
    }
    response.minimum_obstacle_clearance = minimum_clearance;
    response.status = response.occupied_collision ? "occupied" :
                      (response.unknown_queries ? "unknown_overlap" : "free");
    return true;
  }

  void mapTelemetryCallback(const ros::TimerEvent&) {
    const ros::WallTime begin = ros::WallTime::now();
    std::ostringstream line;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!have_origin_ || update_count_.load() == 0) return;
      uint64_t occupied = 0, free = 0;
      for (auto it = tree_.begin_leafs(); it != tree_.end_leafs(); ++it) {
        const uint64_t amount = leafResolutionVoxels(it.getDepth());
        uint64_t& target = tree_.isNodeOccupied(*it) ? occupied : free;
        target = target > std::numeric_limits<uint64_t>::max() - amount ?
                 std::numeric_limits<uint64_t>::max() : target + amount;
      }
      double min_x, min_y, min_z, max_x, max_y, max_z;
      tree_.getMetricMin(min_x, min_y, min_z);
      tree_.getMetricMax(max_x, max_y, max_z);
      const double r = tree_.getResolution();
      const uint64_t nx = std::max<uint64_t>(1, std::ceil((max_x - min_x) / r));
      const uint64_t ny = std::max<uint64_t>(1, std::ceil((max_y - min_y) / r));
      const uint64_t nz = std::max<uint64_t>(1, std::ceil((max_z - min_z) / r));
      const uint64_t total = saturatedMultiply(saturatedMultiply(nx, ny), nz);
      const uint64_t known = occupied > std::numeric_limits<uint64_t>::max() - free ?
                             std::numeric_limits<uint64_t>::max() : occupied + free;
      const uint64_t unknown = total > known ? total - known : 0;
      line << std::fixed << std::setprecision(6)
           << "{\"ros_time\":" << ros::Time::now().toSec()
           << ",\"wall_time\":" << ros::WallTime::now().toSec()
           << ",\"elapsed_time\":" << (ros::WallTime::now() - started_wall_).toSec()
           << ",\"frame_id\":\"" << jsonEscape(!last_cloud_frame_.empty() ? last_cloud_frame_ : odom_frame_) << "\""
           << ",\"map_resolution\":" << r
           << ",\"free_voxels\":" << free << ",\"occupied_voxels\":" << occupied
           << ",\"unknown_voxels\":" << unknown << ",\"observed_voxels\":" << known
           << ",\"total_known_voxels\":" << known
           << ",\"map_min_x\":" << min_x << ",\"map_max_x\":" << max_x
           << ",\"map_min_y\":" << min_y << ",\"map_max_y\":" << max_y
           << ",\"map_min_z\":" << min_z << ",\"map_max_z\":" << max_z
           << ",\"map_extent_x\":" << max_x - min_x
           << ",\"map_extent_y\":" << max_y - min_y
           << ",\"map_extent_z\":" << max_z - min_z
           << ",\"active_goal_id\":" << active_goal_id_.load()
           << ",\"robot_x\":" << sensor_origin_.x() << ",\"robot_y\":" << sensor_origin_.y()
           << ",\"cumulative_trajectory_length\":" << cumulative_trajectory_length_;
    }
    const double duration_ms = (ros::WallTime::now() - begin).toSec() * 1000.0;
    line << ",\"map_statistics_duration_ms\":" << duration_ms << "}\n";
    std::ofstream output(map_growth_path_, std::ios::app);
    if (output) {
      output << line.str();
      output.flush();
    } else {
      ROS_ERROR_THROTTLE(5.0, "Could not append map growth telemetry");
    }
    std_msgs::Float64 duration;
    duration.data = duration_ms;
    statistics_duration_pub_.publish(duration);
    if (duration_ms > map_statistics_warn_ms_)
      ROS_WARN_STREAM_THROTTLE(5.0, "Map statistics took " << duration_ms << " ms");
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& message) {
    octomap::point3d origin;
    double height_offset = 0.0;
    std::string odom_frame;
    ros::Time odom_stamp;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!have_origin_) {
        ROS_WARN_THROTTLE(5.0, "Voxel mapper is waiting for /Odometry");
        return;
      }
      origin = sensor_origin_;
      height_offset = currentMapHeightOffsetLocked();
      odom_frame = odom_frame_;
      odom_stamp = odom_stamp_;
      // Match each registered scan to the closest timestamped odometry
      // sample.  Using the latest callback pose can shift an entire scan by
      // several decimetres when the mapper queue lags FAST-LIO, producing
      // thick/black walls in the final OctoMap.
      if (!message->header.stamp.isZero() && !odom_history_.empty()) {
        double best_error = std::numeric_limits<double>::infinity();
        for (const auto& sample : odom_history_) {
          const double error = std::abs(
              (sample.stamp - message->header.stamp).toSec());
          if (error < best_error) {
            best_error = error;
            origin = sample.position;
            height_offset = sample.height_offset;
            odom_stamp = sample.stamp;
          }
        }
        if (best_error > maximum_odom_cloud_sync_error_) {
          ROS_WARN_THROTTLE(5.0,
              "Voxel mapper skipped stale scan/odom pair (%.3f s)",
              best_error);
          ++stale_sync_count_;
          return;
        }
      }
    }
    const std::string cloud_frame = message->header.frame_id;
    if (!cloud_frame.empty() && !odom_frame.empty() && cloud_frame != odom_frame) {
      ROS_ERROR_THROTTLE(5.0,
          "Voxel mapper rejected scan: cloud frame '%s' != odometry frame '%s'",
          cloud_frame.c_str(), odom_frame.c_str());
      ++frame_mismatch_count_;
      return;
    }
    if (!message->header.stamp.isZero() && !odom_stamp.isZero() &&
        std::abs((message->header.stamp - odom_stamp).toSec()) > 0.5) {
      ROS_WARN_THROTTLE(5.0, "Voxel mapper odometry is more than 0.5 s from scan");
    }

    octomap::Pointcloud scan;
    std::set<uint64_t> unique_keys;
    const size_t total_points = static_cast<size_t>(message->width) * message->height;
    const size_t stride = maximum_scan_points_ > 0 && total_points >
                              static_cast<size_t>(maximum_scan_points_)
                          ? (total_points + maximum_scan_points_ - 1) /
                                maximum_scan_points_
                          : 1;
    size_t index = 0;
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*message, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*message, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*message, "z");
      for (; x != x.end(); ++x, ++y, ++z, ++index) {
        if (index % stride) continue;
        const double px = *x, py = *y, pz = *z - height_offset;
        if (!finite3(px, py, pz)) continue;
        const octomap::point3d endpoint(px, py, pz);
        const double horizontal_range = std::hypot(
            endpoint.x() - origin.x(), endpoint.y() - origin.y());
        const double relative_z = endpoint.z() - origin.z();
        // Registered clouds are already in the FAST-LIO map frame.  The
        // downward-looking part of a 360-degree scan can still hit the robot's
        // own trunk/legs.  Inserting those returns leaves occupied footprints
        // along the trajectory and eventually makes the robot's current start
        // cell appear blocked.  Filter only the configurable below-sensor body
        // envelope; nearby walls above that envelope remain available to the
        // collision checker.
        if (self_filter_radius_ > 0.0 &&
            horizontal_range <= self_filter_radius_ &&
            relative_z >= self_filter_min_relative_z_ &&
            relative_z <= self_filter_max_relative_z_) {
          ++self_filtered_endpoint_count_;
          continue;
        }
        const double range = (endpoint - origin).norm();
        if (range < minimum_range_ || range > maximum_range_) continue;
        octomap::OcTreeKey key;
        if (!tree_.coordToKeyChecked(endpoint, key)) continue;
        const uint64_t packed = static_cast<uint64_t>(key.k[0]) |
                                (static_cast<uint64_t>(key.k[1]) << 16) |
                                (static_cast<uint64_t>(key.k[2]) << 32);
        if (unique_keys.insert(packed).second) scan.push_back(endpoint);
      }
    } catch (const std::runtime_error& error) {
      ROS_ERROR_STREAM_THROTTLE(5.0, "Invalid PointCloud2 fields: " << error.what());
      return;
    }
    if (scan.size() == 0) return;

    bool should_save = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tree_.insertPointCloud(scan, origin, maximum_range_, true, true);
      tree_.updateInnerOccupancy();
      ++update_count_;
      inserted_endpoint_count_ += scan.size();
      last_cloud_stamp_ = message->header.stamp;
      last_cloud_frame_ = cloud_frame;
      last_cloud_fields_.clear();
      for (const auto& field : message->fields) last_cloud_fields_.push_back(field.name);
      last_point_step_ = message->point_step;
      if (first_cloud_stamp_.isZero()) first_cloud_stamp_ = message->header.stamp;
      if (update_count_ == 1 || update_count_ % 10 == 0)
        occupied_progress_.push_back(countOccupiedResolutionVoxels());
      should_save = save_every_updates_ > 0 && update_count_ % save_every_updates_ == 0;
      if (projection_every_updates_ > 0 &&
          update_count_ % projection_every_updates_ == 0)
        publishProjectionLocked(message->header.stamp);
    }
    if (should_save) save("periodic");
    ROS_INFO_STREAM_THROTTLE(10.0, "Voxel map updates=" << update_count_.load()
                             << " endpoints=" << inserted_endpoint_count_.load());
  }

  void doneCallback(const std_msgs::BoolConstPtr& message) {
    if (message->data) save("mission_complete");
  }

  void finalizeCallback(const std_msgs::BoolConstPtr& message) {
    if (message->data) save("finalize_request");
  }

  void publishProjectionLocked(const ros::Time& stamp) {
    if (!have_origin_ || tree_.size() <= 1) return;
    double min_x, min_y, min_z, max_x, max_y, max_z;
    tree_.getMetricMin(min_x, min_y, min_z);
    tree_.getMetricMax(max_x, max_y, max_z);
    const double resolution = tree_.getResolution();
    min_x = std::floor((min_x - resolution) / resolution) * resolution;
    min_y = std::floor((min_y - resolution) / resolution) * resolution;
    max_x = std::ceil((max_x + resolution) / resolution) * resolution;
    max_y = std::ceil((max_y + resolution) / resolution) * resolution;
    const size_t width = std::max<size_t>(1, std::ceil((max_x - min_x) / resolution));
    const size_t height = std::max<size_t>(1, std::ceil((max_y - min_y) / resolution));
    if (width > 2000 || height > 2000) {
      ROS_WARN_THROTTLE(5.0, "Voxel projection bounds are unexpectedly large");
      return;
    }
    nav_msgs::OccupancyGrid message;
    message.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    message.header.frame_id = !last_cloud_frame_.empty() ? last_cloud_frame_ : odom_frame_;
    message.info.map_load_time = message.header.stamp;
    message.info.resolution = resolution;
    message.info.width = static_cast<uint32_t>(width);
    message.info.height = static_cast<uint32_t>(height);
    message.info.origin.position.x = min_x;
    message.info.origin.position.y = min_y;
    message.info.origin.orientation.w = 1.0;
    message.data.assign(width * height, -1);
    const double z_min = sensor_origin_.z() + projection_min_height_;
    const double z_max = sensor_origin_.z() + projection_max_height_;
    // Free evidence first, then occupied evidence so a wall at any body
    // height always dominates a free ray at another height.
    for (int occupancy_pass = 0; occupancy_pass < 2; ++occupancy_pass) {
      for (auto it = tree_.begin_leafs(); it != tree_.end_leafs(); ++it) {
        const bool occupied = tree_.isNodeOccupied(*it);
        if (occupied != (occupancy_pass == 1)) continue;
        const double half = it.getSize() * 0.5;
        if (it.getZ() + half < z_min || it.getZ() - half > z_max) continue;
        const int x0 = std::max(0, static_cast<int>(std::floor(
            (it.getX() - half - min_x) / resolution)));
        const int x1 = std::min(static_cast<int>(width) - 1,
            static_cast<int>(std::floor((it.getX() + half - min_x) / resolution)));
        const int y0 = std::max(0, static_cast<int>(std::floor(
            (it.getY() - half - min_y) / resolution)));
        const int y1 = std::min(static_cast<int>(height) - 1,
            static_cast<int>(std::floor((it.getY() + half - min_y) / resolution)));
        for (int y = y0; y <= y1; ++y)
          for (int x = x0; x <= x1; ++x)
            message.data[static_cast<size_t>(y) * width + x] = occupied ? 100 : 0;
      }
    }
    projection_pub_.publish(message);
  }

  uint64_t leafResolutionVoxels(unsigned depth) const {
    const unsigned difference = tree_.getTreeDepth() - depth;
    if (difference >= 22) return std::numeric_limits<uint64_t>::max();
    return uint64_t{1} << (3 * difference);
  }

  double currentMapHeightOffsetLocked() const {
    if (!planar_height_stabilization_enabled_ ||
        !planar_height_reference_valid_) return 0.0;
    return vertical_transition_started_ ? frozen_planar_height_offset_ :
                                          current_planar_height_offset_;
  }

  uint64_t countOccupiedResolutionVoxels() const {
    uint64_t occupied = 0;
    for (auto it = tree_.begin_leafs(); it != tree_.end_leafs(); ++it) {
      if (!tree_.isNodeOccupied(*it)) continue;
      const uint64_t amount = leafResolutionVoxels(it.getDepth());
      occupied = std::min(std::numeric_limits<uint64_t>::max() - amount, occupied) + amount;
    }
    return occupied;
  }

  std::string statisticsJson(const std::string& reason) const {
    uint64_t occupied = 0, free = 0;
    for (auto it = tree_.begin_leafs(); it != tree_.end_leafs(); ++it) {
      const uint64_t amount = leafResolutionVoxels(it.getDepth());
      uint64_t& target = tree_.isNodeOccupied(*it) ? occupied : free;
      target = target > std::numeric_limits<uint64_t>::max() - amount
                   ? std::numeric_limits<uint64_t>::max()
                   : target + amount;
    }
    double min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0;
    if (tree_.size() > 1) {
      tree_.getMetricMin(min_x, min_y, min_z);
      tree_.getMetricMax(max_x, max_y, max_z);
    }
    const double r = tree_.getResolution();
    const uint64_t nx = std::max<uint64_t>(1, std::ceil((max_x - min_x) / r));
    const uint64_t ny = std::max<uint64_t>(1, std::ceil((max_y - min_y) / r));
    const uint64_t nz = std::max<uint64_t>(1, std::ceil((max_z - min_z) / r));
    const uint64_t total = saturatedMultiply(saturatedMultiply(nx, ny), nz);
    const uint64_t known = occupied > std::numeric_limits<uint64_t>::max() - free
                               ? std::numeric_limits<uint64_t>::max()
                               : occupied + free;
    const uint64_t unknown = total > known ? total - known : 0;

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n  \"schema\": \"simenv_voxel_map_statistics_v1\",\n"
        << "  \"map_file\": \"voxel_map.bt\",\n"
        << "  \"map_format\": \"OctoMap OcTree binary\",\n"
        << "  \"input_interfaces\": {\n"
        << "    \"registered_cloud\": {\"topic\": \"" << jsonEscape(cloud_topic_)
        << "\", \"ros_type\": \"sensor_msgs/PointCloud2\", \"frame_id\": \""
        << jsonEscape(last_cloud_frame_) << "\", \"fields\": [";
    for (size_t i = 0; i < last_cloud_fields_.size(); ++i) {
      if (i) out << ", ";
      out << "\"" << jsonEscape(last_cloud_fields_[i]) << "\"";
    }
    out << "], \"point_step_bytes\": " << last_point_step_ << "},\n"
        << "    \"odometry\": {\"topic\": \"" << jsonEscape(odom_topic_)
        << "\", \"ros_type\": \"nav_msgs/Odometry\", \"frame_id\": \""
        << jsonEscape(odom_frame_) << "\", \"child_frame_id\": \""
        << jsonEscape(odom_child_frame_) << "\"},\n"
        << "    \"accumulated_cloud\": {\"topic\": \"/cloud_map\", "
           "\"ros_type\": \"sensor_msgs/PointCloud2\", "
           "\"usage\": \"visualization only; not used for ray insertion\"}\n"
        << "  },\n"
        << "  \"output_interfaces\": {\n"
        << "    \"floor_projection\": {\"topic\": \""
        << jsonEscape(projection_topic_)
        << "\", \"ros_type\": \"nav_msgs/OccupancyGrid\", "
           "\"source\": \"same OctoMap tree as voxel_map.bt\", "
           "\"states\": [\"unknown\", \"free\", \"occupied\"]}\n"
        << "  },\n"
        << "  \"frame_id\": \"" << jsonEscape(last_cloud_frame_) << "\",\n"
        << "  \"resolution\": " << r << ",\n"
        << "  \"occupied_voxel_count\": " << occupied << ",\n"
        << "  \"free_voxel_count\": " << free << ",\n"
        << "  \"unknown_voxel_count\": " << unknown << ",\n"
        << "  \"unknown_definition\": \"unobserved resolution cells inside map bounds\",\n"
        << "  \"tree_leaf_count\": " << tree_.getNumLeafNodes() << ",\n"
        << "  \"tree_node_count\": " << tree_.size() << ",\n"
        << "  \"update_count\": " << update_count_.load() << ",\n"
        << "  \"inserted_endpoint_count\": " << inserted_endpoint_count_.load() << ",\n"
        << "  \"self_filtered_endpoint_count\": "
        << self_filtered_endpoint_count_.load() << ",\n"
        << "  \"self_filter\": {\"radius\": " << self_filter_radius_
        << ", \"min_relative_z\": " << self_filter_min_relative_z_
        << ", \"max_relative_z\": " << self_filter_max_relative_z_ << "},\n"
        << "  \"planar_height_stabilization\": {\"enabled\": "
        << (planar_height_stabilization_enabled_ ? "true" : "false")
        << ", \"reference_m\": " << planar_height_reference_
        << ", \"current_offset_m\": " << currentMapHeightOffsetLocked()
        << ", \"maximum_offset_m\": " << maximum_planar_height_offset_
        << ", \"vertical_transition_started\": "
        << (vertical_transition_started_ ? "true" : "false") << "},\n"
        << "  \"frame_mismatch_count\": " << frame_mismatch_count_.load() << ",\n"
        << "  \"first_cloud_stamp\": " << first_cloud_stamp_.toSec() << ",\n"
        << "  \"last_cloud_stamp\": " << last_cloud_stamp_.toSec() << ",\n"
        << "  \"map_bounds\": {\n"
        << "    \"min\": [" << min_x << ", " << min_y << ", " << min_z << "],\n"
        << "    \"max\": [" << max_x << ", " << max_y << ", " << max_z << "],\n"
        << "    \"voxel_dimensions\": [" << nx << ", " << ny << ", " << nz << "]\n"
        << "  },\n  \"occupied_progress_every_10_updates\": [";
    for (size_t i = 0; i < occupied_progress_.size(); ++i) {
      if (i) out << ", ";
      out << occupied_progress_[i];
    }
    out << "],\n  \"last_save_reason\": \"" << jsonEscape(reason) << "\"\n}\n";
    return out.str();
  }

  void save(const std::string& reason) {
    if (saving_.exchange(true)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (update_count_ == 0) {
      saving_ = false;
      return;
    }
    // Publish a projection from the exact tree snapshot that is serialized.
    // The manager waits for both this refresh and /voxel_map_saved before its
    // final executed-trajectory audit.
    publishProjectionLocked(last_cloud_stamp_);
    // OctoMap selects the serialization from the filename suffix. Keep .bt
    // on the temporary path, otherwise writeBinary() may emit a full .ot tree.
    const std::string bt_temp = output_dir_ + "/voxel_map.tmp.bt";
    const std::string stats_temp = stats_path_ + ".tmp";
    const bool tree_ok = tree_.writeBinary(bt_temp);
    std::ofstream stats(stats_temp);
    stats << statisticsJson(reason);
    const bool stats_ok = static_cast<bool>(stats);
    stats.close();
    const bool tree_installed =
        tree_ok && std::rename(bt_temp.c_str(), bt_path_.c_str()) == 0;
    const bool stats_installed =
        stats_ok && std::rename(stats_temp.c_str(), stats_path_.c_str()) == 0;
    if (!tree_installed) std::remove(bt_temp.c_str());
    if (!stats_installed) std::remove(stats_temp.c_str());
    if (!tree_installed || !stats_installed)
      ROS_ERROR_STREAM("Failed to save voxel map in " << output_dir_);
    else {
      ROS_INFO_STREAM("Saved OctoMap to " << bt_path_ << " after "
                      << update_count_.load() << " updates (" << reason << ")");
      std_msgs::Bool saved;
      saved.data = true;
      saved_pub_.publish(saved);
    }
    saving_ = false;
  }

  ros::NodeHandle nh_, private_nh_, stabilized_odom_nh_;
  ros::CallbackQueue stabilized_odom_queue_;
  std::unique_ptr<ros::AsyncSpinner> stabilized_odom_spinner_;
  ros::Subscriber cloud_sub_, odom_sub_, done_sub_, finalize_sub_, active_goal_sub_;
  ros::Subscriber stabilized_odom_sub_;
  ros::Subscriber stair_state_sub_, elevator_state_sub_;
  ros::Subscriber second_floor_state_sub_;
  ros::Subscriber upper_floor_stair_state_sub_, third_floor_state_sub_;
  ros::Publisher projection_pub_, saved_pub_, statistics_duration_pub_;
  ros::Publisher stabilized_odom_pub_;
  ros::ServiceServer collision_service_;
  ros::Timer map_timer_;
  mutable std::mutex mutex_;
  octomap::OcTree tree_;
  octomap::point3d sensor_origin_;
  bool have_origin_{false};
  std::string cloud_topic_, odom_topic_, output_dir_, bt_path_, stats_path_, map_growth_path_;
  std::string projection_topic_, stabilized_odom_topic_;
  std::string odom_frame_, last_cloud_frame_;
  std::string odom_child_frame_;
  std::vector<std::string> last_cloud_fields_;
  uint32_t last_point_step_{0};
  ros::Time odom_stamp_, first_cloud_stamp_, last_cloud_stamp_;
  struct OdomSample {
    ros::Time stamp;
    octomap::point3d position;
    double height_offset;
  };
  std::deque<OdomSample> odom_history_;
  double maximum_range_, minimum_range_;
  double self_filter_radius_, self_filter_min_relative_z_;
  double self_filter_max_relative_z_;
  int maximum_scan_points_, save_every_updates_;
  int projection_every_updates_;
  double hit_probability_, miss_probability_, clamping_min_, clamping_max_;
  double occupancy_threshold_;
  double projection_min_height_, projection_max_height_;
  double telemetry_map_rate_hz_, map_statistics_warn_ms_;
  double maximum_odom_cloud_sync_error_{0.20};
  bool planar_height_stabilization_enabled_{false};
  bool collision_request_pose_is_stabilized_{false};
  bool planar_height_reference_valid_{false};
  bool vertical_transition_started_{false};
  bool second_floor_planar_reanchor_pending_{false};
  bool second_floor_planar_reanchored_{false};
  bool third_floor_planar_reanchor_pending_{false};
  bool third_floor_planar_reanchored_{false};
  bool third_floor_stabilization_clear_pending_{false};
  bool third_floor_map_rebuilt_{false};
  // Matches FAST-LIO's absolute_height_recovery_threshold_m: only a real
  // truth-guard snap (F2 plane -> F3 plane, ~2.5 m) trips this, never F2
  // plane jitter.
  static constexpr double k_third_floor_clear_z_jump_m{0.75};
  bool upper_floor_climb_started_{false};
  double planar_height_reference_{0.0};
  double current_planar_height_offset_{0.0};
  double frozen_planar_height_offset_{0.0};
  std::atomic<double> navigation_height_offset_{0.0};
  std::atomic<bool> second_floor_navigation_height_lock_pending_{false};
  std::atomic<bool> second_floor_navigation_height_lock_active_{false};
  std::atomic<double> second_floor_navigation_height_reference_{0.0};
  double maximum_planar_height_offset_{0.0};
  double cumulative_trajectory_length_{0.0};
  ros::Time started_ros_;
  ros::WallTime started_wall_;
  std::atomic<int> active_goal_id_{-1};
  std::atomic<uint64_t> update_count_{0}, inserted_endpoint_count_{0};
  std::atomic<uint64_t> self_filtered_endpoint_count_{0};
  std::atomic<uint64_t> frame_mismatch_count_{0};
  std::atomic<uint64_t> stale_sync_count_{0};
  std::atomic<bool> saving_{false};
  std::vector<uint64_t> occupied_progress_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "fastlio_voxel_mapper");
  FastlioVoxelMapper mapper;
  ros::spin();
  return 0;
}
