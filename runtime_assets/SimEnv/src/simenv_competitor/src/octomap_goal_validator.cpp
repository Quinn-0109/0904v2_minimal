#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <octomap/OcTree.h>

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: octomap_goal_validator MAP.bt X Y Z CLEARANCE\n";
    return 2;
  }
  const double x = std::atof(argv[2]), y = std::atof(argv[3]);
  const double z = std::atof(argv[4]), required = std::atof(argv[5]);
  octomap::OcTree tree(argv[1]);
  const auto* node = tree.search(x, y, z);
  const bool free = node && !tree.isNodeOccupied(node);
  double minimum = std::numeric_limits<double>::infinity();
  for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
    if (!tree.isNodeOccupied(*it)) continue;
    const double half = it.getSize() * 0.5;
    // Match the online traversability layer: floor and low leg/self returns
    // below the body must not masquerade as horizontal collision obstacles.
    if (it.getZ() + half < z - 0.05 || it.getZ() - half > z + 1.20) continue;
    const double dx = std::max(0.0, std::abs(it.getX() - x) - half);
    const double dy = std::max(0.0, std::abs(it.getY() - y) - half);
    minimum = std::min(minimum, std::hypot(dx, dy));
  }
  const bool safe = free && minimum + 1e-6 >= required;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"free_space\":" << (free ? "true" : "false")
            << ",\"minimum_obstacle_clearance\":" << minimum
            << ",\"required_clearance\":" << required
            << ",\"safe\":" << (safe ? "true" : "false") << "}\n";
  return safe ? 0 : 1;
}
