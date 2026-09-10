#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <octomap/OcTree.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Cell {
  int x{0}, y{0}, z{0};
  bool operator==(const Cell& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash {
  size_t operator()(const Cell& value) const {
    size_t seed = std::hash<int>{}(value.x);
    seed ^= std::hash<int>{}(value.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(value.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct Cluster {
  int id{-1};
  std::vector<Cell> cells;
  std::array<double, 3> center{};
  std::array<double, 3> minimum{};
  std::array<double, 3> maximum{};
  double robot_distance{0.0};
};

struct Candidate {
  int id{-1};
  int cluster_id{-1};
  double x{0}, y{0}, z{0}, yaw{0};
  double frontier_distance{0}, robot_distance{0}, path_distance{0};
  double clearance{0}, gain{0}, collision_cost{0}, score{0};
  double original_information_gain{0}, room_information_gain{0};
  double local_unknown_volume{0}, cluster_size{0}, novelty_bonus{0};
  double revisit_penalty{0}, depth_breadth_bonus{0};
  // Diagnostic-only copy of the distance already used to derive novelty and
  // revisit terms.  It is not an additional scoring input.
  double history_separation{4.0};
  double execution_history_distance{std::numeric_limits<double>::infinity()};
  double frontier_x{0}, frontier_y{0};
  bool adaptive_clearance{false};
  bool execution_duplicate{false};
  bool transit_revisit_fallback{false};
};

struct RayGainDebug {
  int ray_id{-1};
  double direction_x{0.0}, direction_y{0.0}, direction_z{0.0};
  double ray_length{0.0};
  double first_unknown_distance{-1.0};
  double first_occupied_distance{-1.0};
  size_t visible_unknown_cells{0};
  size_t visible_free_cells{0};
  size_t visible_occupied_cells{0};
  size_t unknown_after_first_unknown{0};
  bool stopped_by_obstacle{false};
  bool reached_max_range{false};
};

struct CandidateGainDebug {
  std::vector<RayGainDebug> rays;
  size_t total_unknown_cells{0};
  size_t unknown_before_first_obstacle{0};
  size_t unknown_after_first_obstacle{0};
  size_t unknown_after_first_unknown{0};
};

struct HistoryPoint {
  double x{0.0}, y{0.0};
  std::string source{"legacy"};
};

struct ClusterAudit {
  Cluster cluster;
  std::string reason{"unprocessed"};
  size_t candidate_count_before_safety{0};
  size_t candidate_count_center_safe{0};
  size_t candidate_count_after_safety{0};
  size_t candidate_count_after_connectivity{0};
  size_t candidate_count_after_gain{0};
  size_t candidate_count_before_duplicate_filter{0};
  size_t candidate_count_duplicate_or_blacklisted{0};
  bool used_adaptive_clearance{false};
};

struct CandidateGeneration {
  std::vector<Candidate> connected;
  size_t before_safety{0}, center_safe{0}, footprint_safe{0};
  size_t connected_count{0};
  bool used_adaptive{false};
};

std::string escapeJson(const std::string& value) {
  std::ostringstream out;
  for (const char c : value) {
    if (c == '\\' || c == '"') out << '\\';
    out << c;
  }
  return out.str();
}

bool finitePose(double x, double y, double z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

}  // namespace

class FuelLitePlanner {
 public:
  FuelLitePlanner() : nh_(), pnh_("~") {
    pnh_.param<std::string>("map_file", map_file_, "results/latest/voxel_map.bt");
    pnh_.param<std::string>("output_dir", output_dir_, "results/latest");
    pnh_.param<std::string>("frame_id", frame_id_, "camera_init");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    pnh_.param<bool>("use_pose_parameters", use_pose_parameters_, false);
    pnh_.param<bool>("exit_after_plan", exit_after_plan_, false);
    pnh_.param<double>("robot_x", robot_x_, 0.0);
    pnh_.param<double>("robot_y", robot_y_, 0.0);
    pnh_.param<double>("robot_z", robot_z_, 0.45);
    pnh_.param<double>("frontier_slice_half_height", slice_half_height_, 0.30);
    pnh_.param<int>("minimum_cluster_size", minimum_cluster_size_, 8);
    pnh_.param<int>("maximum_clusters", maximum_clusters_, 30);
    pnh_.param<double>("frontier_partition_span", frontier_partition_span_, 3.0);
    pnh_.param<int>("candidates_per_cluster", candidates_per_cluster_, 6);
    pnh_.param<double>("candidate_min_distance", candidate_min_distance_, 0.5);
    pnh_.param<double>("candidate_max_distance", candidate_max_distance_, 2.0);
    pnh_.param<double>("safety_clearance", safety_clearance_, 0.45);
    pnh_.param<double>("adaptive_clearance", adaptive_clearance_, 0.30);
    pnh_.param<double>("minimum_information_gain", minimum_information_gain_, 5.0);
    pnh_.param<double>("sensor_range", sensor_range_, 6.0);
    pnh_.param<std::string>("exploration_mode", exploration_mode_, "GENERIC");
    pnh_.param<bool>("enable_room_information_gain",
                     enable_room_information_gain_, true);
    pnh_.param<double>("room_unknown_decay_lambda",
                       room_unknown_decay_lambda_, 0.35);
    pnh_.param<double>("rolling_goal_lookahead", rolling_goal_lookahead_, 3.0);
    pnh_.param<bool>("use_entry_halfspace", use_entry_halfspace_, false);
    pnh_.param<double>("entry_boundary_x", entry_boundary_x_, 0.0);
    pnh_.param<double>("entry_boundary_y", entry_boundary_y_, 0.0);
    pnh_.param<double>("entry_forward_x", entry_forward_x_, 1.0);
    pnh_.param<double>("entry_forward_y", entry_forward_y_, 0.0);
    pnh_.param<double>("alpha", alpha_, 0.25);
    pnh_.param<double>("beta", beta_, 0.3);
    pnh_.param<double>("gamma", gamma_, 1.0);
    pnh_.param<double>("unknown_weight", unknown_weight_, 5.0);
    pnh_.param<double>("cluster_weight", cluster_weight_, 1.0);
    pnh_.param<double>("novelty_weight", novelty_weight_, 600.0);
    pnh_.param<double>("revisit_weight", revisit_weight_, 500.0);
    pnh_.param<double>("execution_duplicate_radius", execution_duplicate_radius_, 0.35);
    pnh_.param<double>("transit_revisit_min_progress", transit_revisit_min_progress_, 0.40);
    pnh_.param<bool>("allow_transit_revisit_fallback",
                     allow_transit_revisit_fallback_, true);
    pnh_.param<double>("minimum_selected_score", minimum_selected_score_,
                       -std::numeric_limits<double>::infinity());
    pnh_.param<bool>("enable_region_exclusion", enable_region_exclusion_, false);
    pnh_.param<double>("excluded_region_x", excluded_region_x_, 0.0);
    pnh_.param<double>("excluded_region_y", excluded_region_y_, 0.0);
    pnh_.param<double>("excluded_region_radius", excluded_region_radius_, 0.0);
    pnh_.param<std::string>("depth_breadth_phase", depth_breadth_phase_, "DISABLED");
    pnh_.param<double>("depth_breadth_anchor_x", depth_breadth_anchor_x_, 0.0);
    pnh_.param<double>("depth_breadth_anchor_y", depth_breadth_anchor_y_, 0.0);
    pnh_.param<double>("depth_breadth_depth_weight", depth_breadth_depth_weight_, 200.0);
    pnh_.param<double>("depth_breadth_separation_weight",
                       depth_breadth_separation_weight_, 250.0);
    pnh_.param<std::string>("history_file", history_file_, "");

    candidates_per_cluster_ = std::max(3, std::min(10, candidates_per_cluster_));
    room_unknown_decay_lambda_ = std::max(0.0, room_unknown_decay_lambda_);
    goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/exploration_goal", 1, true);
    if (use_pose_parameters_) {
      plan_timer_ = nh_.createTimer(ros::Duration(0.5), &FuelLitePlanner::timerCallback,
                                    this, true);
    } else {
      odom_sub_ = nh_.subscribe(odom_topic_, 5, &FuelLitePlanner::odomCallback, this);
    }
  }

 private:
  void timerCallback(const ros::TimerEvent&) { planOnce(); }

  void odomCallback(const nav_msgs::OdometryConstPtr& message) {
    if (planned_) return;
    robot_x_ = message->pose.pose.position.x;
    robot_y_ = message->pose.pose.position.y;
    robot_z_ = message->pose.pose.position.z;
    if (!message->header.frame_id.empty()) frame_id_ = message->header.frame_id;
    planOnce();
  }

  int gx(double x) const { return static_cast<int>(std::floor((x - min_x_) / resolution_)); }
  int gy(double y) const { return static_cast<int>(std::floor((y - min_y_) / resolution_)); }
  int gz(double z) const { return static_cast<int>(std::floor((z - min_z_) / resolution_)); }
  double wx(int x) const { return min_x_ + (x + 0.5) * resolution_; }
  double wy(int y) const { return min_y_ + (y + 0.5) * resolution_; }
  double wz(int z) const { return min_z_ + (z + 0.5) * resolution_; }
  int index2(int x, int y) const { return y * nx_ + x; }
  bool inside2(int x, int y) const { return x >= 0 && y >= 0 && x < nx_ && y < ny_; }

  bool isFree(double x, double y, double z) const {
    const auto* node = tree_->search(x, y, z);
    return node && !tree_->isNodeOccupied(node);
  }

  bool isUnknown(double x, double y, double z) const {
    return tree_->search(x, y, z) == nullptr;
  }

  bool loadMap() {
    try {
      tree_.reset(new octomap::OcTree(map_file_));
    } catch (const std::exception& error) {
      ROS_ERROR_STREAM("Failed to load OctoMap: " << error.what());
      return false;
    }
    if (!tree_ || tree_->size() <= 1) return false;
    resolution_ = tree_->getResolution();
    tree_->getMetricMin(min_x_, min_y_, min_z_);
    tree_->getMetricMax(max_x_, max_y_, max_z_);
    nx_ = std::max(1, static_cast<int>(std::ceil((max_x_ - min_x_) / resolution_)));
    ny_ = std::max(1, static_cast<int>(std::ceil((max_y_ - min_y_) / resolution_)));
    return true;
  }

  std::vector<Cell> detectFrontiers() const {
    std::vector<Cell> frontiers;
    const int z0 = std::max(0, gz(robot_z_ - slice_half_height_));
    const int z1 = std::min(static_cast<int>(std::ceil((max_z_ - min_z_) / resolution_)) - 1,
                            gz(robot_z_ + slice_half_height_));
    for (int z = z0; z <= z1; ++z) {
      for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
          if (!isFree(wx(x), wy(y), wz(z))) continue;
          bool unknown_neighbor = false;
          for (int dz = -1; dz <= 1 && !unknown_neighbor; ++dz)
            for (int dy = -1; dy <= 1 && !unknown_neighbor; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (isUnknown(wx(x + dx), wy(y + dy), wz(z + dz))) {
                  unknown_neighbor = true;
                  break;
                }
              }
          if (unknown_neighbor) frontiers.push_back({x, y, z});
        }
      }
    }
    return frontiers;
  }

  std::vector<Cluster> clusterFrontiers(const std::vector<Cell>& frontiers) const {
    std::unordered_set<Cell, CellHash> remaining(frontiers.begin(), frontiers.end());
    std::vector<Cluster> clusters;
    while (!remaining.empty()) {
      Cluster cluster;
      std::queue<Cell> queue;
      const Cell seed = *remaining.begin();
      remaining.erase(seed);
      queue.push(seed);
      while (!queue.empty()) {
        const Cell current = queue.front();
        queue.pop();
        cluster.cells.push_back(current);
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              Cell next{current.x + dx, current.y + dy, current.z + dz};
              auto found = remaining.find(next);
              if (found != remaining.end()) {
                remaining.erase(found);
                queue.push(next);
              }
            }
      }
      // Keep small connected components for the diagnostic chain.  The size
      // threshold is applied later so a disappearing frontier is explainable.
      cluster.minimum = {{std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max()}};
      cluster.maximum = {{-std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max()}};
      for (const Cell& cell : cluster.cells) {
        const std::array<double, 3> point{{wx(cell.x), wy(cell.y), wz(cell.z)}};
        for (int axis = 0; axis < 3; ++axis) {
          cluster.center[axis] += point[axis];
          cluster.minimum[axis] = std::min(cluster.minimum[axis], point[axis]);
          cluster.maximum[axis] = std::max(cluster.maximum[axis], point[axis]);
        }
      }
      for (double& value : cluster.center) value /= cluster.cells.size();
      cluster.robot_distance = std::hypot(cluster.center[0] - robot_x_,
                                          cluster.center[1] - robot_y_);
      clusters.push_back(std::move(cluster));
    }
    // A fresh 360-degree scan often creates one connected frontier ring around
    // the robot.  Using that ring's global centroid yields viewpoints in the
    // already observed centre and therefore no valid cold-start goal.  Split
    // only oversized connected components into local, robot-relative spatial
    // tiles.  Candidate generation and FUEL score terms remain unchanged.
    std::vector<Cluster> partitioned;
    for (const Cluster& cluster : clusters) {
      const double span_x = cluster.maximum[0] - cluster.minimum[0];
      const double span_y = cluster.maximum[1] - cluster.minimum[1];
      if (std::max(span_x, span_y) <= frontier_partition_span_) {
        partitioned.push_back(cluster);
        continue;
      }
      std::unordered_map<long long, std::vector<Cell>> tiles;
      for (const Cell& cell : cluster.cells) {
        const int tx = static_cast<int>(std::floor((wx(cell.x) - robot_x_) /
                                                   frontier_partition_span_));
        const int ty = static_cast<int>(std::floor((wy(cell.y) - robot_y_) /
                                                   frontier_partition_span_));
        const long long key = static_cast<long long>(
            (static_cast<uint64_t>(static_cast<uint32_t>(tx)) << 32) |
            static_cast<uint32_t>(ty));
        tiles[key].push_back(cell);
      }
      for (const auto& entry : tiles) {
        Cluster tile;
        tile.cells = entry.second;
        tile.minimum = {{std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max()}};
        tile.maximum = {{-std::numeric_limits<double>::max(),
                         -std::numeric_limits<double>::max(),
                         -std::numeric_limits<double>::max()}};
        for (const Cell& cell : tile.cells) {
          const std::array<double, 3> point{{wx(cell.x), wy(cell.y), wz(cell.z)}};
          for (int axis = 0; axis < 3; ++axis) {
            tile.center[axis] += point[axis];
            tile.minimum[axis] = std::min(tile.minimum[axis], point[axis]);
            tile.maximum[axis] = std::max(tile.maximum[axis], point[axis]);
          }
        }
        for (double& value : tile.center) value /= tile.cells.size();
        tile.robot_distance = std::hypot(tile.center[0] - robot_x_,
                                         tile.center[1] - robot_y_);
        partitioned.push_back(std::move(tile));
      }
    }
    clusters.swap(partitioned);
    std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
      if (a.cells.size() != b.cells.size()) return a.cells.size() > b.cells.size();
      return a.robot_distance < b.robot_distance;
    });
    // Do not truncate here: every connected component must remain visible in
    // the filter audit. Candidate ranking naturally limits executable output.
    return clusters;
  }

  void buildTraversability() {
    cv::Mat obstacle(ny_, nx_, CV_8UC1, cv::Scalar(255));
    // Navigation obstacles are evaluated above the floor/leg-return band.
    // FAST-LIO's local origin is at the body, so including 0.30 m below it
    // turns floor and self echoes into a solid obstacle around the start.
    const double low_z = robot_z_ - 0.05;
    const double high_z = robot_z_ + 1.20;
    for (auto it = tree_->begin_leafs(); it != tree_->end_leafs(); ++it) {
      if (!tree_->isNodeOccupied(*it)) continue;
      const double half = it.getSize() * 0.5;
      if (it.getZ() + half < low_z || it.getZ() - half > high_z) continue;
      const int x0 = std::max(0, gx(it.getX() - half));
      const int x1 = std::min(nx_ - 1, gx(it.getX() + half));
      const int y0 = std::max(0, gy(it.getY() - half));
      const int y1 = std::min(ny_ - 1, gy(it.getY() + half));
      for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) obstacle.at<uint8_t>(y, x) = 0;
    }
    cv::distanceTransform(obstacle, clearance_pixels_, cv::DIST_L2, 5);
    traversable_.assign(nx_ * ny_, false);
    adaptive_traversable_.assign(nx_ * ny_, false);
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x) {
        const double clearance = clearance_pixels_.at<float>(y, x) * resolution_;
        // Distance transform measures between grid-cell centres while the
        // footprint audit measures to occupied voxel faces. Add half a voxel
        // so a nominal 0.30/0.45 m grid clearance is not 0.225/0.375 m in the
        // exact OctoMap check.
        const double face_margin = resolution_ * 0.5;
        traversable_[index2(x, y)] = clearance >= safety_clearance_ + face_margin &&
                                      isFree(wx(x), wy(y), robot_z_);
        adaptive_traversable_[index2(x, y)] =
            clearance >= adaptive_clearance_ + face_margin &&
                                               isFree(wx(x), wy(y), robot_z_);
      }

    path_distance_ = computePathDistances(traversable_);
    adaptive_path_distance_ = computePathDistances(adaptive_traversable_);
  }

  std::vector<double> computePathDistances(const std::vector<bool>& mask) const {
    int start_x = gx(robot_x_), start_y = gy(robot_y_);
    if (!inside2(start_x, start_y) || !mask[index2(start_x, start_y)]) {
      double best = std::numeric_limits<double>::max();
      for (int radius = 1; radius <= 15; ++radius)
        for (int dy = -radius; dy <= radius; ++dy)
          for (int dx = -radius; dx <= radius; ++dx) {
            const int x = start_x + dx, y = start_y + dy;
            if (!inside2(x, y) || !mask[index2(x, y)]) continue;
            const double distance = std::hypot(dx, dy);
            if (distance < best) {
              best = distance;
              start_x = x;
              start_y = y;
            }
          }
    }
    std::vector<double> distances(nx_ * ny_, std::numeric_limits<double>::infinity());
    if (!inside2(start_x, start_y) || !mask[index2(start_x, start_y)]) return distances;
    using Entry = std::pair<double, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    distances[index2(start_x, start_y)] = 0.0;
    queue.push({0.0, index2(start_x, start_y)});
    const int directions[8][2] = {{1,0},{-1,0},{0,1},{0,-1},
                                  {1,1},{1,-1},{-1,1},{-1,-1}};
    while (!queue.empty()) {
      const auto current = queue.top(); queue.pop();
      if (current.first != distances[current.second]) continue;
      const int x = current.second % nx_, y = current.second / nx_;
      for (const auto& direction : directions) {
        const int xx = x + direction[0], yy = y + direction[1];
        if (!inside2(xx, yy) || !mask[index2(xx, yy)]) continue;
        const double step = resolution_ * (direction[0] && direction[1] ? std::sqrt(2.0) : 1.0);
        const int next = index2(xx, yy);
        if (current.first + step < distances[next]) {
          distances[next] = current.first + step;
          queue.push({distances[next], next});
        }
      }
    }
    return distances;
  }

  double nearestFrontierDistance(double x, double y, double z,
                                 const Cluster& cluster) const {
    double best = std::numeric_limits<double>::max();
    for (const Cell& cell : cluster.cells) {
      const double dx = wx(cell.x) - x, dy = wy(cell.y) - y, dz = wz(cell.z) - z;
      best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return best;
  }

  double gridClearance(double x, double y) const {
    const int cell_x = gx(x), cell_y = gy(y);
    if (!inside2(cell_x, cell_y)) return 0.0;
    return clearance_pixels_.at<float>(cell_y, cell_x) * resolution_;
  }

  bool directPathTraversable(double x, double y, const std::vector<bool>& mask) const {
    const double distance = std::hypot(x - robot_x_, y - robot_y_);
    const int steps = std::max(1, static_cast<int>(std::ceil(distance /
                                                             (resolution_ * 0.5))));
    bool entered_traversable = false;
    for (int step = 0; step <= steps; ++step) {
      const double ratio = static_cast<double>(step) / steps;
      const int gx_value = gx(robot_x_ + ratio * (x - robot_x_));
      const int gy_value = gy(robot_y_ + ratio * (y - robot_y_));
      if (!inside2(gx_value, gy_value)) return false;
      if (mask[index2(gx_value, gy_value)]) {
        entered_traversable = true;
        continue;
      }
      // The robot can temporarily occupy a cell excluded by its own inflated
      // footprint or a fresh self echo. Allow only the first 0.60 m to bridge
      // into the nearest Dijkstra-safe cell; never bridge an obstacle after
      // the path has entered traversable space.
      if (entered_traversable || ratio * distance > 0.60) return false;
    }
    return entered_traversable;
  }

  CandidateGeneration generateCandidates(const Cluster& cluster, bool adaptive) const {
    CandidateGeneration generated;
    generated.used_adaptive = adaptive;
    const auto& mask = adaptive ? adaptive_traversable_ : traversable_;
    const auto& paths = adaptive ? adaptive_path_distance_ : path_distance_;
    std::set<std::pair<int, int>> used;
    std::vector<Cell> anchors;
    const size_t stride = std::max<size_t>(1, cluster.cells.size() / 12);
    for (size_t i = 0; i < cluster.cells.size(); i += stride) anchors.push_back(cluster.cells[i]);
    anchors.push_back({gx(cluster.center[0]), gy(cluster.center[1]), gz(cluster.center[2])});
    const std::array<double, 4> radii{{0.55, 0.85, 1.20, 1.65}};
    for (const Cell& anchor : anchors) {
      const double ax = wx(anchor.x), ay = wy(anchor.y);
      const double toward_robot = std::atan2(robot_y_ - ay, robot_x_ - ax);
      for (double radius : radii) {
        for (int offset_index = -2; offset_index <= 2; ++offset_index) {
          const double angle = toward_robot + offset_index * kPi / 8.0;
          const int x = gx(ax + radius * std::cos(angle));
          const int y = gy(ay + radius * std::sin(angle));
          if (used.count({x, y})) continue;
          used.insert({x, y});
          ++generated.before_safety;
          if (!inside2(x, y) || !isFree(wx(x), wy(y), robot_z_)) continue;
          ++generated.center_safe;
          if (!mask[index2(x, y)]) continue;
          ++generated.footprint_safe;
          if (!std::isfinite(paths[index2(x, y)])) continue;
          ++generated.connected_count;
        const double px = wx(x), py = wy(y);
        if (use_entry_halfspace_ &&
            (px - entry_boundary_x_) * entry_forward_x_ +
            (py - entry_boundary_y_) * entry_forward_y_ < 0.0) continue;
        const double frontier_distance = nearestFrontierDistance(px, py, robot_z_, cluster);
        if (frontier_distance < candidate_min_distance_ ||
            frontier_distance > candidate_max_distance_) continue;
        Candidate candidate;
        candidate.cluster_id = cluster.id;
        candidate.frontier_x = cluster.center[0];
        candidate.frontier_y = cluster.center[1];
        candidate.x = px; candidate.y = py; candidate.z = robot_z_;
        candidate.yaw = std::atan2(cluster.center[1] - py, cluster.center[0] - px);
        candidate.frontier_distance = frontier_distance;
        candidate.robot_distance = std::hypot(px - robot_x_, py - robot_y_);
        candidate.path_distance = paths[index2(x, y)];
        candidate.clearance = clearance_pixels_.at<float>(y, x) * resolution_;
        candidate.adaptive_clearance = adaptive;
        candidate.cluster_size = static_cast<double>(cluster.cells.size());
        generated.connected.push_back(candidate);
        if (static_cast<int>(generated.connected.size()) >= candidates_per_cluster_)
          return generated;
        }
      }
    }
    return generated;
  }

  std::pair<double, double> executionPoint(const Candidate& candidate) const {
    const auto& mask = candidate.adaptive_clearance ? adaptive_traversable_ : traversable_;
    const auto& distances = candidate.adaptive_clearance ? adaptive_path_distance_ : path_distance_;
    int x = gx(candidate.x), y = gy(candidate.y);
    if (!inside2(x, y) || !std::isfinite(distances[index2(x, y)]))
      return {candidate.x, candidate.y};
    std::vector<std::pair<int, int>> reverse_path;
    reverse_path.push_back({x, y});
    const int directions[8][2] = {{1,0},{-1,0},{0,1},{0,-1},
                                  {1,1},{1,-1},{-1,1},{-1,-1}};
    for (int guard = 0; guard < nx_ * ny_; ++guard) {
      const double current = distances[index2(x, y)];
      if (current <= resolution_ * 1.5) break;
      int best_x = x, best_y = y;
      double best = current;
      for (const auto& direction : directions) {
        const int xx = x + direction[0], yy = y + direction[1];
        if (!inside2(xx, yy) || !mask[index2(xx, yy)]) continue;
        if (distances[index2(xx, yy)] < best) {
          best = distances[index2(xx, yy)]; best_x = xx; best_y = yy;
        }
      }
      if (best_x == x && best_y == y) break;
      x = best_x; y = best_y; reverse_path.push_back({x, y});
    }
    std::reverse(reverse_path.begin(), reverse_path.end());
    std::pair<double, double> selected{robot_x_, robot_y_};
    for (const auto& cell : reverse_path) {
      const double px = wx(cell.first), py = wy(cell.second);
      if (std::hypot(px - robot_x_, py - robot_y_) > rolling_goal_lookahead_) break;
      if (!directPathTraversable(px, py, mask)) break;
      selected = {px, py};
    }
    return selected;
  }

  double executionYaw(const Candidate& candidate) const {
    const auto point = executionPoint(candidate);
    if (std::hypot(point.first - candidate.x, point.second - candidate.y) > 0.35)
      return std::atan2(point.second - robot_y_, point.first - robot_x_);
    return candidate.yaw;
  }

  int informationGain(const Candidate& candidate) const {
    std::unordered_set<Cell, CellHash> visible_unknown;
    const double vertical_angles[] = {-0.35, -0.22, -0.10, 0.0, 0.10, 0.22, 0.35};
    const double step = resolution_ * 0.75;
    for (int azimuth_index = 0; azimuth_index < 72; ++azimuth_index) {
      const double azimuth = candidate.yaw - kPi + 2.0 * kPi * azimuth_index / 72.0;
      for (double elevation : vertical_angles) {
        const double horizontal = std::cos(elevation);
        for (double range = step; range <= sensor_range_; range += step) {
          const double x = candidate.x + range * horizontal * std::cos(azimuth);
          const double y = candidate.y + range * horizontal * std::sin(azimuth);
          const double z = candidate.z + range * std::sin(elevation);
          if (x < min_x_ || x > max_x_ || y < min_y_ || y > max_y_ ||
              z < min_z_ || z > max_z_) break;
          const auto* node = tree_->search(x, y, z);
          if (node && tree_->isNodeOccupied(node)) break;
          if (!node) visible_unknown.insert({gx(x), gy(y), gz(z)});
        }
      }
    }
    return static_cast<int>(visible_unknown.size());
  }

  double roomInformationGain(const Candidate& candidate) const {
    // Room-only alternative to informationGain().  Occupied termination and
    // ray geometry are identical; only successive unknown cells on each ray
    // receive exponentially decaying weights.  A voxel seen by multiple rays
    // contributes its highest (closest-layer) weight once.
    std::unordered_map<Cell, double, CellHash> visible_unknown_weights;
    const double vertical_angles[] = {-0.35, -0.22, -0.10, 0.0, 0.10, 0.22, 0.35};
    const double step = resolution_ * 0.75;
    for (int azimuth_index = 0; azimuth_index < 72; ++azimuth_index) {
      const double azimuth = candidate.yaw - kPi +
                             2.0 * kPi * azimuth_index / 72.0;
      for (double elevation : vertical_angles) {
        const double horizontal = std::cos(elevation);
        std::unordered_set<Cell, CellHash> ray_unknown;
        size_t unknown_layer = 0;
        for (double range = step; range <= sensor_range_; range += step) {
          const double x = candidate.x + range * horizontal * std::cos(azimuth);
          const double y = candidate.y + range * horizontal * std::sin(azimuth);
          const double z = candidate.z + range * std::sin(elevation);
          if (x < min_x_ || x > max_x_ || y < min_y_ || y > max_y_ ||
              z < min_z_ || z > max_z_) break;
          const auto* node = tree_->search(x, y, z);
          if (node && tree_->isNodeOccupied(node)) break;
          if (!node) {
            const Cell cell{gx(x), gy(y), gz(z)};
            if (!ray_unknown.insert(cell).second) continue;
            const double weight = std::exp(
                -room_unknown_decay_lambda_ * static_cast<double>(unknown_layer));
            const auto found = visible_unknown_weights.find(cell);
            if (found == visible_unknown_weights.end() || weight > found->second)
              visible_unknown_weights[cell] = weight;
            ++unknown_layer;
          }
        }
      }
    }
    double gain = 0.0;
    for (const auto& item : visible_unknown_weights) gain += item.second;
    return gain;
  }

  CandidateGainDebug traceInformationGain(const Candidate& candidate) const {
    // Diagnostic replay only.  Keep informationGain() above as the sole value
    // used by scoring and candidate selection.
    CandidateGainDebug debug;
    std::unordered_set<Cell, CellHash> all_unknown;
    std::unordered_set<Cell, CellHash> unknown_before_obstacle;
    std::unordered_set<Cell, CellHash> unknown_after_obstacle;
    std::unordered_set<Cell, CellHash> unknown_after_first_unknown;
    const double vertical_angles[] = {-0.35, -0.22, -0.10, 0.0, 0.10, 0.22, 0.35};
    const double step = resolution_ * 0.75;
    int ray_id = 0;
    for (int azimuth_index = 0; azimuth_index < 72; ++azimuth_index) {
      const double azimuth = candidate.yaw - kPi +
                             2.0 * kPi * azimuth_index / 72.0;
      for (double elevation : vertical_angles) {
        RayGainDebug ray;
        ray.ray_id = ray_id++;
        const double horizontal = std::cos(elevation);
        ray.direction_x = horizontal * std::cos(azimuth);
        ray.direction_y = horizontal * std::sin(azimuth);
        ray.direction_z = std::sin(elevation);
        std::unordered_set<Cell, CellHash> ray_unknown;
        std::unordered_set<Cell, CellHash> ray_free;
        std::unordered_set<Cell, CellHash> ray_occupied;
        std::unordered_set<Cell, CellHash> ray_unknown_after_first_unknown;
        bool saw_unknown = false;
        bool terminated_by_bounds = false;
        for (double range = step; range <= sensor_range_; range += step) {
          const double x = candidate.x + range * ray.direction_x;
          const double y = candidate.y + range * ray.direction_y;
          const double z = candidate.z + range * ray.direction_z;
          if (x < min_x_ || x > max_x_ || y < min_y_ || y > max_y_ ||
              z < min_z_ || z > max_z_) {
            terminated_by_bounds = true;
            break;
          }
          ray.ray_length = range;
          const Cell cell{gx(x), gy(y), gz(z)};
          const auto* node = tree_->search(x, y, z);
          if (node && tree_->isNodeOccupied(node)) {
            if (ray.first_occupied_distance < 0.0)
              ray.first_occupied_distance = range;
            ray_occupied.insert(cell);
            ray.stopped_by_obstacle = true;
            break;
          }
          if (!node) {
            if (ray.first_unknown_distance < 0.0)
              ray.first_unknown_distance = range;
            if (saw_unknown) {
              ray_unknown_after_first_unknown.insert(cell);
              unknown_after_first_unknown.insert(cell);
            }
            saw_unknown = true;
            ray_unknown.insert(cell);
            all_unknown.insert(cell);
            // The production ray stops only at occupied nodes, therefore all
            // unknown counted by it is before the first occupied node.
            unknown_before_obstacle.insert(cell);
          } else {
            ray_free.insert(cell);
          }
        }
        if (!ray.stopped_by_obstacle && !terminated_by_bounds) {
          ray.reached_max_range = true;
          ray.ray_length = sensor_range_;
        }
        ray.visible_unknown_cells = ray_unknown.size();
        ray.visible_free_cells = ray_free.size();
        ray.visible_occupied_cells = ray_occupied.size();
        ray.unknown_after_first_unknown = ray_unknown_after_first_unknown.size();
        debug.rays.push_back(ray);
      }
    }
    debug.total_unknown_cells = all_unknown.size();
    debug.unknown_before_first_obstacle = unknown_before_obstacle.size();
    debug.unknown_after_first_obstacle = unknown_after_obstacle.size();
    debug.unknown_after_first_unknown = unknown_after_first_unknown.size();
    return debug;
  }

  double localUnknownVolume(const Candidate& candidate) const {
    size_t count = 0;
    const int radius = std::max(1, static_cast<int>(std::ceil(2.5 / resolution_)));
    const int cx = gx(candidate.x), cy = gy(candidate.y), cz = gz(candidate.z);
    for (int dz = -std::max(1, radius / 2); dz <= std::max(1, radius / 2); ++dz)
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          if (dx * dx + dy * dy > radius * radius) continue;
          if (isUnknown(wx(cx + dx), wy(cy + dy), wz(cz + dz))) ++count;
        }
    return static_cast<double>(count) * resolution_ * resolution_ * resolution_;
  }

  void loadHistory() {
    history_.clear();
    if (history_file_.empty()) return;
    std::ifstream input(history_file_);
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream values(line);
      double x = 0.0, y = 0.0;
      std::string source{"legacy"};
      if (values >> x >> y && std::isfinite(x) && std::isfinite(y)) {
        values >> source;
        history_.push_back({x, y, source});
      }
    }
  }

  bool isExecutionHistorySource(const std::string& source) const {
    return source == "goal" || source == "failed_goal" ||
           source == "offline_execution";
  }

  void assignExecutionDuplicate(Candidate& candidate) const {
    const auto execution = executionPoint(candidate);
    candidate.execution_history_distance = std::numeric_limits<double>::infinity();
    for (const auto& point : history_) {
      if (!isExecutionHistorySource(point.source)) continue;
      candidate.execution_history_distance = std::min(
          candidate.execution_history_distance,
          std::hypot(execution.first - point.x, execution.second - point.y));
    }
    candidate.execution_duplicate =
        candidate.execution_history_distance < execution_duplicate_radius_;
  }

  void assignScores(std::vector<Candidate>& candidates) const {
    for (Candidate& candidate : candidates) {
      candidate.original_information_gain = informationGain(candidate);
      const bool use_room_gain =
          enable_room_information_gain_ && exploration_mode_ == "ROOM_EXPLORATION";
      candidate.room_information_gain =
          exploration_mode_ == "ROOM_EXPLORATION"
              ? roomInformationGain(candidate)
              : candidate.original_information_gain;
      candidate.gain = use_room_gain ? candidate.room_information_gain
                                     : candidate.original_information_gain;
      candidate.local_unknown_volume = localUnknownVolume(candidate);
      double history_distance = 4.0;
      // A doorway viewpoint can lie on a heavily visited corridor while its
      // associated frontier points into a completely new room. Use the
      // frontier direction/centre for novelty so opposite open doors are not
      // treated as the same revisited corridor location.
      for (const auto& point : history_)
        history_distance = std::min(history_distance,
                                    std::hypot(candidate.frontier_x - point.x,
                                               candidate.frontier_y - point.y));
      candidate.history_separation = history_distance;
      candidate.novelty_bonus = std::min(1.0, history_distance / 3.0);
      candidate.revisit_penalty = std::max(0.0, 1.0 - history_distance / 1.5);
      candidate.collision_cost = 1.0 / std::max(candidate.clearance, 0.05);
      candidate.score = alpha_ * candidate.gain - beta_ * candidate.path_distance -
                        gamma_ * candidate.collision_cost +
                        unknown_weight_ * candidate.local_unknown_volume +
                        cluster_weight_ * candidate.cluster_size +
                        novelty_weight_ * candidate.novelty_bonus -
                        revisit_weight_ * candidate.revisit_penalty;
      const auto execution = executionPoint(candidate);
      if (depth_breadth_phase_ == "LOCAL_DEPTH") {
        const double progress = std::min(
            3.0, std::hypot(execution.first - depth_breadth_anchor_x_,
                            execution.second - depth_breadth_anchor_y_));
        candidate.depth_breadth_bonus = depth_breadth_depth_weight_ * progress / 3.0;
      } else if (depth_breadth_phase_ == "LOCAL_BREADTH") {
        candidate.depth_breadth_bonus = depth_breadth_separation_weight_ *
                                        std::min(2.0, candidate.history_separation) / 2.0;
      }
      candidate.score += candidate.depth_breadth_bonus;
    }
  }

  void writeFrontiers(const std::vector<Cluster>& clusters) const {
    std::ofstream out(output_dir_ + "/frontiers.json");
    out << std::fixed << std::setprecision(6) << "[\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
      const Cluster& c = clusters[i];
      if (i) out << ",\n";
      out << "  {\"id\":" << c.id << ",\"center\":[" << c.center[0] << ','
          << c.center[1] << ',' << c.center[2] << "],\"size\":" << c.cells.size()
          << ",\"bounding_box\":{\"min\":[" << c.minimum[0] << ',' << c.minimum[1]
          << ',' << c.minimum[2] << "],\"max\":[" << c.maximum[0] << ','
          << c.maximum[1] << ',' << c.maximum[2] << "]},\"distance\":"
          << c.robot_distance << '}';
    }
    out << "\n]\n";
  }

  void writeCandidates(const std::vector<Candidate>& candidates) const {
    std::ofstream out(output_dir_ + "/candidate_viewpoints.json");
    out << std::fixed << std::setprecision(6) << "[\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate& c = candidates[i];
      const auto execution = executionPoint(c);
      const double execution_yaw = executionYaw(c);
      if (i) out << ",\n";
      out << "  {\"id\":" << c.id << ",\"frontier_id\":" << c.cluster_id
          << ",\"position\":[" << execution.first << ',' << execution.second << ',' << c.z
          << "],\"viewpoint_position\":[" << c.x << ',' << c.y << ',' << c.z
          << "],\"frontier_center\":[" << c.frontier_x << ',' << c.frontier_y
          << "],\"yaw\":" << execution_yaw << ",\"viewpoint_yaw\":" << c.yaw
          << ",\"distance_to_frontier\":"
          << c.frontier_distance << ",\"distance_to_robot\":"
          << std::hypot(execution.first - robot_x_, execution.second - robot_y_)
          << ",\"viewpoint_distance_to_robot\":" << c.robot_distance
          << ",\"execution_grid_clearance\":"
          << gridClearance(execution.first, execution.second)
          << ",\"path_distance\":" << c.path_distance << ",\"clearance\":"
          << c.clearance << ",\"adaptive_clearance\":"
          << (c.adaptive_clearance ? "true" : "false")
          << ",\"execution_history_distance\":";
      if (std::isfinite(c.execution_history_distance))
        out << c.execution_history_distance;
      else
        out << "null";
      out << ",\"execution_duplicate\":"
          << (c.execution_duplicate ? "true" : "false")
          << ",\"transit_revisit_fallback\":"
          << (c.transit_revisit_fallback ? "true" : "false")
          << ",\"reachable\":true}\n";
    }
    out << "]\n";
  }

  void writeScores(const std::vector<Candidate>& candidates, int selected,
                   size_t raw_frontiers, size_t raw_clusters,
                   size_t discarded_clusters) const {
    std::ofstream out(output_dir_ + "/exploration_score.json");
    out << std::fixed << std::setprecision(6)
        << "{\n  \"parameters\":{\"alpha\":" << alpha_ << ",\"beta\":" << beta_
        << ",\"gamma\":" << gamma_ << ",\"sensor_range\":" << sensor_range_
        << ",\"safety_clearance\":" << safety_clearance_
        << ",\"adaptive_clearance\":" << adaptive_clearance_
        << ",\"minimum_information_gain\":" << minimum_information_gain_
        << ",\"unknown_weight\":" << unknown_weight_
        << ",\"cluster_weight\":" << cluster_weight_
        << ",\"novelty_weight\":" << novelty_weight_
        << ",\"revisit_weight\":" << revisit_weight_
        << ",\"neighbor_connectivity\":26},\n"
        << "  \"counts\":{\"raw_frontier_voxels\":" << raw_frontiers
        << ",\"raw_clusters\":" << raw_clusters << ",\"valid_clusters\":"
        << (raw_clusters - discarded_clusters) << ",\"discarded_clusters\":"
        << discarded_clusters << ",\"candidates\":" << candidates.size() << "},\n"
        << "  \"selected_candidate_id\":" << selected << ",\n  \"scores\":[\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate& c = candidates[i];
      const auto execution = executionPoint(c);
      if (i) out << ",\n";
      out << "    {\"goal_id\":" << c.id << ",\"candidate_id\":" << c.id
          << ",\"frontier_id\":" << c.cluster_id
          << ",\"position\":[" << execution.first << ',' << execution.second << ',' << c.z << ']'
          << ",\"viewpoint_position\":[" << c.x << ',' << c.y << ',' << c.z << ']'
          << ",\"frontier_center\":[" << c.frontier_x << ',' << c.frontier_y << ']'
          << ",\"information_gain\":" << c.gain << ",\"distance_cost\":"
          << c.path_distance << ",\"collision_cost\":" << c.collision_cost
          << ",\"information_gain_contribution\":" << alpha_ * c.gain
          << ",\"local_unknown_contribution\":" << unknown_weight_ * c.local_unknown_volume
          << ",\"cluster_size_contribution\":" << cluster_weight_ * c.cluster_size
          << ",\"novelty_contribution\":" << novelty_weight_ * c.novelty_bonus
          << ",\"distance_contribution\":" << -beta_ * c.path_distance
          << ",\"revisit_contribution\":" << -revisit_weight_ * c.revisit_penalty
          << ",\"depth_breadth_contribution\":" << c.depth_breadth_bonus
          << ",\"collision_contribution\":" << -gamma_ * c.collision_cost
          << ",\"local_unknown_volume\":" << c.local_unknown_volume
          << ",\"cluster_size\":" << c.cluster_size
          << ",\"frontier_cluster_size\":" << c.cluster_size
          << ",\"novelty_bonus\":" << c.novelty_bonus
          << ",\"revisit_penalty\":" << c.revisit_penalty
          << ",\"execution_history_distance\":";
      if (std::isfinite(c.execution_history_distance))
        out << c.execution_history_distance;
      else
        out << "null";
      out << ",\"execution_duplicate\":"
          << (c.execution_duplicate ? "true" : "false")
          << ",\"transit_revisit_fallback\":"
          << (c.transit_revisit_fallback ? "true" : "false")
          << ",\"score\":" << c.score << ",\"final_score\":" << c.score << '}';
    }
    out << "\n  ]\n}\n";
  }

  void writeGainDebug(const std::vector<Candidate>& candidates,
                      int selected) const {
    std::vector<CandidateGainDebug> traces;
    traces.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
      traces.push_back(traceInformationGain(candidate));

    std::ofstream detail(output_dir_ + "/candidate_gain_debug.json");
    detail << std::fixed << std::setprecision(6)
           << "{\n  \"schema\":\"simenv_candidate_gain_debug_v1\","
           << "\n  \"selected_candidate_id\":" << selected
           << ",\n  \"candidates\":[\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate& candidate = candidates[i];
      const CandidateGainDebug& trace = traces[i];
      const auto execution = executionPoint(candidate);
      const double after_obstacle_ratio =
          trace.total_unknown_cells == 0 ? 0.0 :
          static_cast<double>(trace.unknown_after_first_obstacle) /
          static_cast<double>(trace.total_unknown_cells);
      const double after_first_unknown_ratio =
          trace.total_unknown_cells == 0 ? 0.0 :
          static_cast<double>(trace.unknown_after_first_unknown) /
          static_cast<double>(trace.total_unknown_cells);
      if (i) detail << ",\n";
      detail << "    {\"candidate_id\":" << candidate.id
             << ",\"position\":[" << candidate.x << ',' << candidate.y << ','
             << candidate.z << ']'
             << ",\"execution_position\":[" << execution.first << ','
             << execution.second << ',' << candidate.z << ']'
             << ",\"information_gain\":" << candidate.original_information_gain
             << ",\"selected_information_gain\":" << candidate.gain
             << ",\"distance_cost\":" << candidate.path_distance
             << ",\"collision_cost\":" << candidate.collision_cost
             << ",\"local_unknown_volume\":" << candidate.local_unknown_volume
             << ",\"novelty_bonus\":" << candidate.novelty_bonus
             << ",\"history_separation\":" << candidate.history_separation
             << ",\"final_score\":" << candidate.score
             << ",\"selected\":" << (candidate.id == selected ? "true" : "false")
             << ",\"total_unknown_cells\":" << trace.total_unknown_cells
             << ",\"unknown_before_first_obstacle\":"
             << trace.unknown_before_first_obstacle
             << ",\"unknown_after_first_obstacle\":"
             << trace.unknown_after_first_obstacle
             << ",\"unknown_ratio_after_obstacle\":" << after_obstacle_ratio
             << ",\"unknown_after_first_unknown\":"
             << trace.unknown_after_first_unknown
             << ",\"unknown_ratio_after_first_unknown\":"
             << after_first_unknown_ratio
             << ",\"trace_matches_information_gain\":"
             << (std::fabs(candidate.original_information_gain -
                           static_cast<double>(trace.total_unknown_cells)) < 0.5
                     ? "true" : "false")
             << ",\"rays\":[\n";
      for (size_t ray_index = 0; ray_index < trace.rays.size(); ++ray_index) {
        const RayGainDebug& ray = trace.rays[ray_index];
        if (ray_index) detail << ",\n";
        detail << "      {\"ray_id\":" << ray.ray_id
               << ",\"ray_direction\":[" << ray.direction_x << ','
               << ray.direction_y << ',' << ray.direction_z << ']'
               << ",\"ray_length\":" << ray.ray_length
               << ",\"first_unknown_distance\":";
        if (ray.first_unknown_distance >= 0.0)
          detail << ray.first_unknown_distance;
        else
          detail << "null";
        detail << ",\"first_occupied_distance\":";
        if (ray.first_occupied_distance >= 0.0)
          detail << ray.first_occupied_distance;
        else
          detail << "null";
        detail << ",\"visible_unknown_cells\":" << ray.visible_unknown_cells
               << ",\"visible_free_cells\":" << ray.visible_free_cells
               << ",\"visible_occupied_cells\":" << ray.visible_occupied_cells
               << ",\"unknown_after_first_unknown\":"
               << ray.unknown_after_first_unknown
               << ",\"ray_stopped_by_obstacle\":"
               << (ray.stopped_by_obstacle ? "true" : "false")
               << ",\"ray_reached_max_range\":"
               << (ray.reached_max_range ? "true" : "false") << '}';
      }
      detail << "\n    ]}";
    }
    detail << "\n  ]\n}\n";

    std::ofstream summary(output_dir_ + "/candidate_gain_summary.json");
    summary << std::fixed << std::setprecision(6)
            << "{\n  \"schema\":\"simenv_candidate_gain_summary_v1\","
            << "\n  \"selected_candidate_id\":" << selected
            << ",\n  \"candidates\":[\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate& candidate = candidates[i];
      const CandidateGainDebug& trace = traces[i];
      const double ratio = trace.total_unknown_cells == 0 ? 0.0 :
          static_cast<double>(trace.unknown_after_first_obstacle) /
          static_cast<double>(trace.total_unknown_cells);
      const double first_unknown_ratio = trace.total_unknown_cells == 0 ? 0.0 :
          static_cast<double>(trace.unknown_after_first_unknown) /
          static_cast<double>(trace.total_unknown_cells);
      if (i) summary << ",\n";
      summary << "    {\"candidate_id\":" << candidate.id
              << ",\"information_gain\":" << candidate.original_information_gain
              << ",\"selected_information_gain\":" << candidate.gain
              << ",\"total_unknown\":" << trace.total_unknown_cells
              << ",\"unknown_before_obstacle\":"
              << trace.unknown_before_first_obstacle
              << ",\"unknown_after_obstacle\":"
              << trace.unknown_after_first_obstacle
              << ",\"after_obstacle_ratio\":" << ratio
              << ",\"unknown_after_first_unknown\":"
              << trace.unknown_after_first_unknown
              << ",\"after_first_unknown_ratio\":" << first_unknown_ratio
              << ",\"selected\":"
              << (candidate.id == selected ? "true" : "false") << '}';
    }
    summary << "\n  ]\n}\n";
  }

  void writeRoomGainDebug(const std::vector<Candidate>& candidates,
                          int selected) const {
    std::ofstream out(output_dir_ + "/candidate_room_gain_debug.json");
    out << std::fixed << std::setprecision(6)
        << "{\n  \"schema\":\"simenv_candidate_room_gain_debug_v1\","
        << "\n  \"mode\":\"" << escapeJson(exploration_mode_) << '\"'
        << ",\n  \"enable_room_information_gain\":"
        << (enable_room_information_gain_ ? "true" : "false")
        << ",\n  \"room_unknown_decay_lambda\":"
        << room_unknown_decay_lambda_
        << ",\n  \"selected_candidate_id\":" << selected
        << ",\n  \"candidates\":[\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate& candidate = candidates[i];
      if (i) out << ",\n";
      out << "    {\"candidate_id\":" << candidate.id
          << ",\"original_information_gain\":"
          << candidate.original_information_gain
          << ",\"room_information_gain\":"
          << candidate.room_information_gain
          << ",\"selected_gain\":" << candidate.gain
          << ",\"mode\":\"" << escapeJson(exploration_mode_) << '\"'
          << ",\"selected\":"
          << (candidate.id == selected ? "true" : "false") << '}';
    }
    out << "\n  ]\n}\n";
  }

  void writeFilterDiagnostic(const std::vector<ClusterAudit>& audits,
                             size_t raw_frontiers,
                             const std::vector<Candidate>& candidates) const {
    std::map<std::string, size_t> reasons;
    size_t after_size = 0, after_height = 0, after_connectivity = 0;
    size_t before_safety = 0, after_safety = 0, after_gain = 0;
    size_t duplicate_candidates = 0, after_duplicate = 0;
    for (const auto& audit : audits) {
      ++reasons[audit.reason];
      if (audit.reason != "too_small") ++after_size;
      if (audit.reason != "too_small" && audit.reason != "wrong_height") ++after_height;
      if (audit.candidate_count_after_connectivity > 0) ++after_connectivity;
      before_safety += audit.candidate_count_before_safety;
      after_safety += audit.candidate_count_after_safety;
      after_gain += audit.candidate_count_after_gain;
      duplicate_candidates += audit.candidate_count_duplicate_or_blacklisted;
      after_duplicate += audit.candidate_count_before_duplicate_filter -
                         audit.candidate_count_duplicate_or_blacklisted;
    }
    std::ofstream out(output_dir_ + "/frontier_filter_diagnostic.json");
    out << std::fixed << std::setprecision(6)
        << "{\n  \"counts\":{\"raw_frontier_voxels\":" << raw_frontiers
        << ",\"raw_clusters\":" << audits.size()
        << ",\"clusters_after_size_filter\":" << after_size
        << ",\"clusters_after_height_filter\":" << after_height
        << ",\"clusters_after_connectivity_filter\":" << after_connectivity
        << ",\"candidate_count_before_safety\":" << before_safety
        << ",\"candidate_count_after_safety\":" << after_safety
        << ",\"candidate_count_after_gain\":" << after_gain
        << ",\"candidate_count_duplicate_or_blacklisted\":" << duplicate_candidates
        << ",\"candidate_count_after_duplicate_filter\":" << after_duplicate
        << ",\"final_valid_goal_count\":" << (candidates.empty() ? 0 : 1)
        << ",\"valid_candidates\":" << candidates.size();
    for (const std::string& name : {"too_small", "wrong_height", "not_connected",
                                   "inside_inflation", "no_free_viewpoint",
                                   "gain_too_low", "outside_floor_bounds",
                                   "duplicate_or_blacklisted"})
      if (!reasons.count(name)) out << ",\"" << name << "\":0";
    for (const auto& reason : reasons)
      out << ",\"" << escapeJson(reason.first) << "\":" << reason.second;
    out << "},\n  \"clusters\":[\n";
    for (size_t i = 0; i < audits.size(); ++i) {
      const auto& a = audits[i];
      if (i) out << ",\n";
      out << "    {\"id\":" << a.cluster.id << ",\"center\":["
          << a.cluster.center[0] << ',' << a.cluster.center[1] << ','
          << a.cluster.center[2] << "],\"size\":" << a.cluster.cells.size()
          << ",\"area_m2\":" << a.cluster.cells.size() * resolution_ * resolution_
          << ",\"reason\":\"" << escapeJson(a.reason)
          << "\",\"used_adaptive_clearance\":"
          << (a.used_adaptive_clearance ? "true" : "false")
          << ",\"candidate_count_before_safety\":" << a.candidate_count_before_safety
          << ",\"candidate_count_center_safe\":" << a.candidate_count_center_safe
          << ",\"candidate_count_after_safety\":" << a.candidate_count_after_safety
          << ",\"candidate_count_after_connectivity\":"
          << a.candidate_count_after_connectivity
          << ",\"candidate_count_after_gain\":" << a.candidate_count_after_gain
          << ",\"candidate_count_before_duplicate_filter\":"
          << a.candidate_count_before_duplicate_filter
          << ",\"candidate_count_duplicate_or_blacklisted\":"
          << a.candidate_count_duplicate_or_blacklisted << '}';
    }
    out << "\n  ]\n}\n";
  }

  void writeGoal(const Candidate& goal) const {
    const auto execution = executionPoint(goal);
    const double yaw = executionYaw(goal);
    const double qz = std::sin(yaw * 0.5), qw = std::cos(yaw * 0.5);
    std::ofstream out(output_dir_ + "/current_exploration_goal.json");
    out << std::fixed << std::setprecision(6)
        << "{\n  \"frame_id\":\"" << escapeJson(frame_id_) << "\",\n"
        << "  \"candidate_id\":" << goal.id << ",\n  \"frontier_id\":"
        << goal.cluster_id << ",\n  \"position\":[" << execution.first << ','
        << execution.second << ','
        << goal.z << "],\n  \"orientation\":[0.0,0.0," << qz << ',' << qw
        << "],\n  \"viewpoint_position\":[" << goal.x << ',' << goal.y << ',' << goal.z
        << "],\n  \"frontier_center\":[" << goal.frontier_x << ','
        << goal.frontier_y
        << "],\n  \"path_distance\":" << goal.path_distance
        << ",\n  \"yaw\":" << yaw << ",\n  \"viewpoint_yaw\":" << goal.yaw
        << ",\n  \"information_gain\":"
        << goal.gain << ",\n  \"score\":" << goal.score
        << ",\n  \"execution_grid_clearance\":"
        << gridClearance(execution.first, execution.second)
        << ",\n  \"execution_duplicate\":"
        << (goal.execution_duplicate ? "true" : "false")
        << ",\n  \"transit_revisit_fallback\":"
        << (goal.transit_revisit_fallback ? "true" : "false")
        << ",\n  \"reachable\":true,\n  \"free_space\":true,\n  \"clearance\":"
        << goal.clearance << "\n}\n";
  }

  void render(const std::vector<Cluster>& clusters,
              const std::vector<Candidate>& candidates, int selected,
              const std::vector<ClusterAudit>& audits) const {
    cv::Mat image(ny_, nx_, CV_8UC3, cv::Scalar(150, 150, 150));
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x) {
        const auto* node = tree_->search(wx(x), wy(y), robot_z_);
        if (!node) continue;
        image.at<cv::Vec3b>(ny_ - 1 - y, x) = tree_->isNodeOccupied(node)
            ? cv::Vec3b(25, 25, 25) : cv::Vec3b(235, 235, 235);
      }
    for (const Cluster& cluster : clusters)
      for (const Cell& cell : cluster.cells)
        if (inside2(cell.x, cell.y))
          image.at<cv::Vec3b>(ny_ - 1 - cell.y, cell.x) = cv::Vec3b(40, 40, 230);
    for (const Candidate& c : candidates) {
      const cv::Scalar color = c.id == selected ? cv::Scalar(30, 210, 30)
                                                : cv::Scalar(230, 120, 30);
      cv::circle(image, cv::Point(gx(c.x), ny_ - 1 - gy(c.y)),
                 c.id == selected ? 4 : 2, color, -1, cv::LINE_AA);
    }
    cv::circle(image, cv::Point(gx(robot_x_), ny_ - 1 - gy(robot_y_)), 4,
               cv::Scalar(20, 180, 230), -1, cv::LINE_AA);
    cv::Mat enlarged;
    cv::resize(image, enlarged, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
    cv::putText(enlarged, "gray=unknown white=free black=occupied red=frontier "
                          "blue=candidate green=goal yellow=robot",
                cv::Point(12, 25), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(enlarged, "gray=unknown white=free black=occupied red=frontier "
                          "blue=candidate green=goal yellow=robot",
                cv::Point(12, 25), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::imwrite(output_dir_ + "/exploration_debug.png", enlarged);
    cv::Mat diagnostic(ny_, nx_, CV_8UC3, cv::Scalar(150, 150, 150));
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x) {
        const auto* node = tree_->search(wx(x), wy(y), robot_z_);
        if (node) diagnostic.at<cv::Vec3b>(ny_ - 1 - y, x) =
            tree_->isNodeOccupied(node) ? cv::Vec3b(25,25,25) : cv::Vec3b(235,235,235);
      }
    const std::map<std::string, cv::Vec3b> colors = {
      {"valid", {40,190,40}}, {"too_small", {110,110,110}},
      {"no_free_viewpoint", {20,150,240}}, {"inside_inflation", {20,20,230}},
      {"not_connected", {210,80,180}}, {"gain_too_low", {220,150,30}}
    };
    for (const auto& audit : audits) {
      const auto found = colors.find(audit.reason);
      const cv::Vec3b color = found == colors.end() ? cv::Vec3b(0,0,0) : found->second;
      for (const auto& cell : audit.cluster.cells)
        if (inside2(cell.x, cell.y)) diagnostic.at<cv::Vec3b>(ny_ - 1 - cell.y, cell.x) = color;
    }
    for (const auto& c : candidates)
      cv::circle(diagnostic, {gx(c.x), ny_ - 1 - gy(c.y)}, 3,
                 c.adaptive_clearance ? cv::Scalar(255,0,255) : cv::Scalar(255,100,0), -1);
    cv::circle(diagnostic, {gx(robot_x_), ny_ - 1 - gy(robot_y_)}, 4,
               cv::Scalar(0,220,255), -1);
    cv::resize(diagnostic, diagnostic, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
    cv::putText(diagnostic, "green=valid red=inflation purple=disconnected orange=no-free gray=small",
                {12,25}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,255,255}, 1, cv::LINE_AA);
    cv::imwrite(output_dir_ + "/frontier_filter_diagnostic.png", diagnostic);
  }

  void publishGoal(const Candidate& goal) {
    const auto execution = executionPoint(goal);
    const double yaw = executionYaw(goal);
    geometry_msgs::PoseStamped message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = frame_id_;
    message.pose.position.x = execution.first;
    message.pose.position.y = execution.second;
    message.pose.position.z = goal.z;
    message.pose.orientation.z = std::sin(yaw * 0.5);
    message.pose.orientation.w = std::cos(yaw * 0.5);
    goal_pub_.publish(message);
  }

  void planOnce() {
    if (planned_) return;
    planned_ = true;
    if (!finitePose(robot_x_, robot_y_, robot_z_) || !loadMap()) {
      ROS_ERROR("FUEL-lite has no valid pose or map");
      ros::shutdown();
      return;
    }
    const std::vector<Cell> raw_frontiers = detectFrontiers();
    std::vector<Cluster> raw_clusters = clusterFrontiers(raw_frontiers);
    buildTraversability();
    loadHistory();
    std::vector<Cluster> valid_clusters;
    std::vector<Candidate> candidates;
    std::vector<Candidate> transit_revisit_candidates;
    std::vector<ClusterAudit> audits;
    size_t discarded = 0;
    for (Cluster& cluster : raw_clusters) {
      cluster.id = static_cast<int>(audits.size());
      ClusterAudit audit;
      audit.cluster = cluster;
      if (use_entry_halfspace_ &&
          (cluster.center[0] - entry_boundary_x_) * entry_forward_x_ +
          (cluster.center[1] - entry_boundary_y_) * entry_forward_y_ < 0.0) {
        audit.reason = "outside_floor_bounds";
        audits.push_back(audit);
        ++discarded;
        continue;
      }
      if (static_cast<int>(cluster.cells.size()) < minimum_cluster_size_) {
        audit.reason = "too_small";
        audits.push_back(audit);
        ++discarded;
        continue;
      }
      CandidateGeneration generated = generateCandidates(cluster, false);
      if (generated.connected.empty()) {
        CandidateGeneration adaptive = generateCandidates(cluster, true);
        generated.before_safety = std::max(generated.before_safety, adaptive.before_safety);
        generated.center_safe = std::max(generated.center_safe, adaptive.center_safe);
        generated.footprint_safe = std::max(generated.footprint_safe, adaptive.footprint_safe);
        generated.connected_count = adaptive.connected_count;
        generated.connected = std::move(adaptive.connected);
        generated.used_adaptive = true;
      }
      audit.candidate_count_before_safety = generated.before_safety;
      audit.candidate_count_center_safe = generated.center_safe;
      audit.candidate_count_after_safety = generated.footprint_safe;
      audit.candidate_count_after_connectivity = generated.connected_count;
      audit.used_adaptive_clearance = generated.used_adaptive;
      if (generated.center_safe == 0) audit.reason = "no_free_viewpoint";
      else if (generated.footprint_safe == 0) audit.reason = "inside_inflation";
      else if (generated.connected.empty()) audit.reason = "not_connected";
      if (generated.connected.empty()) {
        audits.push_back(audit);
        ++discarded;
        continue;
      }
      assignScores(generated.connected);
      if (enable_region_exclusion_ && excluded_region_radius_ > 0.0) {
        generated.connected.erase(
            std::remove_if(generated.connected.begin(), generated.connected.end(),
                           [&](const Candidate& candidate) {
                             const auto execution = executionPoint(candidate);
                             return std::hypot(execution.first - excluded_region_x_,
                                               execution.second - excluded_region_y_) <
                                    excluded_region_radius_;
                           }),
            generated.connected.end());
        if (generated.connected.empty()) {
          audit.reason = "depth_breadth_region_excluded";
          audits.push_back(audit);
          ++discarded;
          continue;
        }
      }
      generated.connected.erase(
          std::remove_if(generated.connected.begin(), generated.connected.end(),
                         [&](const Candidate& c) { return c.gain < minimum_information_gain_; }),
          generated.connected.end());
      audit.candidate_count_after_gain = generated.connected.size();
      if (generated.connected.empty()) {
        audit.reason = "gain_too_low";
        audits.push_back(audit);
        ++discarded;
        continue;
      }
      audit.candidate_count_before_duplicate_filter = generated.connected.size();
      for (Candidate& candidate : generated.connected)
        assignExecutionDuplicate(candidate);
      for (const Candidate& candidate : generated.connected) {
        const auto execution = executionPoint(candidate);
        if (candidate.execution_duplicate &&
            std::hypot(execution.first - robot_x_, execution.second - robot_y_) >=
                transit_revisit_min_progress_)
          transit_revisit_candidates.push_back(candidate);
      }
      generated.connected.erase(
          std::remove_if(generated.connected.begin(), generated.connected.end(),
                         [](const Candidate& candidate) {
                           return candidate.execution_duplicate;
                         }),
          generated.connected.end());
      audit.candidate_count_duplicate_or_blacklisted =
          audit.candidate_count_before_duplicate_filter - generated.connected.size();
      if (generated.connected.empty()) {
        audit.reason = "duplicate_or_blacklisted";
        audits.push_back(audit);
        ++discarded;
        continue;
      }
      audit.reason = "valid";
      audits.push_back(audit);
      valid_clusters.push_back(cluster);
      candidates.insert(candidates.end(), generated.connected.begin(), generated.connected.end());
    }
    // A position-only duplicate radius must not turn previously traversed
    // doorways and corridors into permanent no-go regions. Prefer novel
    // execution points globally; when none exist, publish one explicitly
    // labelled transit revisit that still makes measurable progress.
    if (allow_transit_revisit_fallback_ && candidates.empty() &&
        !transit_revisit_candidates.empty()) {
      auto best = std::max_element(
          transit_revisit_candidates.begin(), transit_revisit_candidates.end(),
          [](const Candidate& a, const Candidate& b) { return a.score < b.score; });
      best->transit_revisit_fallback = true;
      candidates.push_back(*best);
      for (ClusterAudit& audit : audits) {
        if (audit.cluster.id != best->cluster_id) continue;
        if (audit.reason == "duplicate_or_blacklisted" && discarded > 0) --discarded;
        audit.reason = "valid";
        valid_clusters.push_back(audit.cluster);
        break;
      }
    }
    for (size_t i = 0; i < candidates.size(); ++i) candidates[i].id = i;
    int selected = -1;
    for (size_t i = 0; i < candidates.size(); ++i)
      if (selected < 0 || candidates[i].score > candidates[selected].score) selected = i;
    if (selected >= 0 && candidates[selected].score < minimum_selected_score_)
      selected = -1;

    writeFrontiers(valid_clusters);
    writeCandidates(candidates);
    writeScores(candidates, selected, raw_frontiers.size(), raw_clusters.size(), discarded);
    // Baseline/GENERIC runs must not emit room-specific artifacts.  Keep the
    // diagnostic only for the explicitly enabled room exploration mode.
    if (enable_room_information_gain_ && exploration_mode_ == "ROOM_EXPLORATION")
      writeRoomGainDebug(candidates, selected);
    writeFilterDiagnostic(audits, raw_frontiers.size(), candidates);
    render(valid_clusters, candidates, selected, audits);
    if (selected < 0) {
      ROS_ERROR("FUEL-lite found no safe reachable candidate viewpoint");
      ros::shutdown();
      return;
    }
    writeGoal(candidates[selected]);
    publishGoal(candidates[selected]);
    ROS_INFO_STREAM("FUEL-lite selected candidate " << selected << " from "
                    << candidates.size() << " viewpoints and " << valid_clusters.size()
                    << " frontier clusters; gain=" << candidates[selected].gain
                    << " score=" << candidates[selected].score);
    if (exit_after_plan_) shutdown_timer_ = nh_.createTimer(
        ros::Duration(2.0), [](const ros::TimerEvent&) { ros::shutdown(); }, true);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber odom_sub_;
  ros::Publisher goal_pub_;
  ros::Timer plan_timer_, shutdown_timer_;
  std::unique_ptr<octomap::OcTree> tree_;
  bool use_pose_parameters_{false}, exit_after_plan_{false}, planned_{false};
  bool allow_transit_revisit_fallback_{true};
  bool use_entry_halfspace_{false};
  std::string map_file_, output_dir_, frame_id_, odom_topic_, history_file_;
  std::string exploration_mode_{"GENERIC"}, depth_breadth_phase_{"DISABLED"};
  double robot_x_{0}, robot_y_{0}, robot_z_{0.45};
  double slice_half_height_{0.30}, candidate_min_distance_{0.5};
  double candidate_max_distance_{2.0}, safety_clearance_{0.45}, sensor_range_{6.0};
  double rolling_goal_lookahead_{3.0};
  double entry_boundary_x_{0.0}, entry_boundary_y_{0.0};
  double entry_forward_x_{1.0}, entry_forward_y_{0.0};
  double alpha_{0.25}, beta_{0.3}, gamma_{1.0};
  double adaptive_clearance_{0.30}, minimum_information_gain_{5.0};
  double room_unknown_decay_lambda_{0.35};
  double unknown_weight_{5.0}, cluster_weight_{1.0};
  double novelty_weight_{600.0}, revisit_weight_{500.0};
  double execution_duplicate_radius_{0.35};
  double transit_revisit_min_progress_{0.40};
  double minimum_selected_score_{-std::numeric_limits<double>::infinity()};
  double excluded_region_x_{0.0}, excluded_region_y_{0.0};
  double excluded_region_radius_{0.0};
  double depth_breadth_anchor_x_{0.0}, depth_breadth_anchor_y_{0.0};
  double depth_breadth_depth_weight_{200.0}, depth_breadth_separation_weight_{250.0};
  bool enable_room_information_gain_{true}, enable_region_exclusion_{false};
  int minimum_cluster_size_{8}, maximum_clusters_{30}, candidates_per_cluster_{6};
  double frontier_partition_span_{3.0};
  double resolution_{0.15}, min_x_{0}, min_y_{0}, min_z_{0}, max_x_{0}, max_y_{0}, max_z_{0};
  int nx_{0}, ny_{0};
  cv::Mat clearance_pixels_;
  std::vector<bool> traversable_, adaptive_traversable_;
  std::vector<double> path_distance_, adaptive_path_distance_;
  std::vector<HistoryPoint> history_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "fuel_lite_planner");
  FuelLitePlanner planner;
  ros::spin();
  return 0;
}
