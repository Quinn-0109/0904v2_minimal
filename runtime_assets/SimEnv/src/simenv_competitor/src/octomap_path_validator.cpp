#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <octomap/OcTree.h>

struct Sample {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct Obstacle {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double half{0.0};
};

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "usage: octomap_path_validator MAP.bt PATH.csv Z CLEARANCE "
                 "MAX_SAMPLE_SPACING [INITIAL_UNKNOWN_GRACE]\n";
    return 2;
  }
  const std::string map_file = argv[1], path_file = argv[2];
  const double z = std::atof(argv[3]);
  const double required_clearance = std::atof(argv[4]);
  const double maximum_spacing = std::atof(argv[5]);
  const double initial_unknown_grace = argc == 7 ? std::atof(argv[6]) : 0.0;
  if (!(required_clearance >= 0.0) || !(maximum_spacing > 0.0) ||
      !(initial_unknown_grace >= 0.0)) return 2;

  std::ifstream input(path_file);
  if (!input) {
    std::cerr << "could not open path file: " << path_file << '\n';
    return 2;
  }
  std::vector<Sample> samples;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream values(line);
    Sample sample;
    if (values >> sample.x >> sample.y >> sample.yaw &&
        std::isfinite(sample.x) && std::isfinite(sample.y) &&
        std::isfinite(sample.yaw)) {
      samples.push_back(sample);
    }
  }
  if (samples.size() < 2) {
    std::cerr << "path must contain at least two finite samples\n";
    return 2;
  }

  octomap::OcTree tree(map_file);
  std::vector<Obstacle> obstacles;
  obstacles.reserve(tree.size() / 8 + 1);
  for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
    if (!tree.isNodeOccupied(*it)) continue;
    const double half = it.getSize() * 0.5;
    // Same body-height slice as the online 2-D safety layer.
    if (it.getZ() + half < z - 0.05 || it.getZ() - half > z + 1.20) continue;
    obstacles.push_back({it.getX(), it.getY(), it.getZ(), half});
  }

  int unknown = 0, ignored_initial_unknown = 0, occupied = 0;
  int clearance_failures = 0, spacing_failures = 0;
  double travelled = 0.0;
  double minimum_clearance = std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < samples.size(); ++index) {
    const Sample& sample = samples[index];
    if (index > 0) {
      const double step = std::hypot(sample.x - samples[index - 1].x,
                                     sample.y - samples[index - 1].y);
      travelled += step;
      if (step > maximum_spacing + 1e-6) ++spacing_failures;
    }
    // The online 2-D projection marks a column known when any voxel in the
    // robot-height slice has been observed.  Searching only at the floor z
    // falsely labelled valid FUEL paths unknown because most LiDAR rays pass
    // above that exact voxel.  Mirror the projection's vertical semantics.
    bool column_known = false;
    bool column_occupied = false;
    const double vertical_step = std::max(0.05, tree.getResolution());
    for (double probe_z = z - 0.05; probe_z <= z + 1.20 + 1e-9;
         probe_z += vertical_step) {
      const auto* node = tree.search(sample.x, sample.y, probe_z);
      if (!node) continue;
      column_known = true;
      column_occupied = column_occupied || tree.isNodeOccupied(node);
    }
    if (!column_known) {
      if (travelled <= initial_unknown_grace + 1e-9)
        ++ignored_initial_unknown;
      else
        ++unknown;
    } else if (column_occupied) {
      ++occupied;
    }
    double clearance = std::numeric_limits<double>::infinity();
    for (const Obstacle& obstacle : obstacles) {
      const double dx = std::max(0.0, std::abs(obstacle.x - sample.x) - obstacle.half);
      const double dy = std::max(0.0, std::abs(obstacle.y - sample.y) - obstacle.half);
      clearance = std::min(clearance, std::hypot(dx, dy));
    }
    minimum_clearance = std::min(minimum_clearance, clearance);
    if (clearance + 1e-6 < required_clearance) ++clearance_failures;
  }
  const bool safe = unknown == 0 && occupied == 0 && clearance_failures == 0 &&
                    spacing_failures == 0;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"safe\":" << (safe ? "true" : "false")
            << ",\"samples\":" << samples.size()
            << ",\"unknown_samples\":" << unknown
            << ",\"ignored_initial_unknown_samples\":" << ignored_initial_unknown
            << ",\"occupied_samples\":" << occupied
            << ",\"clearance_failures\":" << clearance_failures
            << ",\"spacing_failures\":" << spacing_failures
            << ",\"minimum_obstacle_clearance\":";
  if (std::isfinite(minimum_clearance))
    std::cout << minimum_clearance;
  else
    std::cout << "null";
  std::cout << ",\"required_clearance\":" << required_clearance
            << ",\"initial_unknown_grace\":" << initial_unknown_grace << "}\n";
  return safe ? 0 : 1;
}
