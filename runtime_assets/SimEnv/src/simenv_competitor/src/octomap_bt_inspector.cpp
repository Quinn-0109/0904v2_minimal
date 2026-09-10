#include <cmath>
#include <iostream>
#include <string>

#include <octomap/OcTree.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: octomap_bt_inspector MAP.bt\n";
    return 2;
  }
  // .bt is the compact binary occupancy format, whose header intentionally
  // differs from the full .ot AbstractOcTree format. OcTree's file
  // constructor is the supported loader for .bt.
  octomap::OcTree loaded(argv[1]);
  auto* tree = &loaded;
  size_t occupied = 0, free = 0;
  for (auto it = tree->begin_leafs(); it != tree->end_leafs(); ++it) {
    tree->isNodeOccupied(*it) ? ++occupied : ++free;
  }
  double min_x, min_y, min_z, max_x, max_y, max_z;
  tree->getMetricMin(min_x, min_y, min_z);
  tree->getMetricMax(max_x, max_y, max_z);
  const bool valid = tree->size() > 1 && occupied > 0 && free > 0 &&
                     std::isfinite(min_x) && std::isfinite(max_x);
  std::cout << "{\"reload_success\":" << (valid ? "true" : "false")
            << ",\"resolution\":" << tree->getResolution()
            << ",\"tree_node_count\":" << tree->size()
            << ",\"occupied_leaf_count\":" << occupied
            << ",\"free_leaf_count\":" << free
            << ",\"bounds_min\":[" << min_x << ',' << min_y << ',' << min_z << ']'
            << ",\"bounds_max\":[" << max_x << ',' << max_y << ',' << max_z << "]}\n";
  return valid ? 0 : 5;
}
