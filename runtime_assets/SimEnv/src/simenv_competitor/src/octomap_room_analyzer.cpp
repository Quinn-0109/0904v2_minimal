#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <octomap/OcTree.h>

namespace {

struct Region {
  std::string name;
  double min_x, min_y, max_x, max_y;
};

struct Counts {
  uint64_t free{0}, occupied{0}, unknown{0};
};

int classify(const octomap::OcTree& tree, double x, double y, double z) {
  const auto* node = tree.search(x, y, z);
  if (!node) return 0;
  return tree.isNodeOccupied(node) ? 2 : 1;
}

Counts countRegion(const octomap::OcTree& tree, const Region& region,
                   double min_z, double max_z) {
  Counts result;
  const double resolution = tree.getResolution();
  for (double x = region.min_x + resolution * 0.5; x < region.max_x; x += resolution)
    for (double y = region.min_y + resolution * 0.5; y < region.max_y; y += resolution)
      for (double z = min_z + resolution * 0.5; z < max_z; z += resolution) {
        const int state = classify(tree, x, y, z);
        if (state == 2) ++result.occupied;
        else if (state == 1) ++result.free;
        else ++result.unknown;
      }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 11 || (argc - 6) % 5 != 0) {
    std::cerr << "usage: octomap_room_analyzer MAP.bt OUTPUT.json GRID.csv "
                 "Z_MIN Z_MAX NAME X_MIN Y_MIN X_MAX Y_MAX [NAME ...]\n";
    return 2;
  }
  const std::string map_path = argv[1], json_path = argv[2], csv_path = argv[3];
  const double min_z = std::stod(argv[4]), max_z = std::stod(argv[5]);
  std::vector<Region> regions;
  for (int index = 6; index < argc; index += 5) {
    Region region{argv[index], std::stod(argv[index + 1]), std::stod(argv[index + 2]),
                  std::stod(argv[index + 3]), std::stod(argv[index + 4])};
    if (region.min_x > region.max_x) std::swap(region.min_x, region.max_x);
    if (region.min_y > region.max_y) std::swap(region.min_y, region.max_y);
    regions.push_back(region);
  }

  octomap::OcTree tree(map_path);
  if (tree.size() <= 1) {
    std::cerr << "OctoMap is empty\n";
    return 3;
  }
  double map_min_x, map_min_y, map_min_z, map_max_x, map_max_y, map_max_z;
  tree.getMetricMin(map_min_x, map_min_y, map_min_z);
  tree.getMetricMax(map_max_x, map_max_y, map_max_z);

  std::vector<Counts> counts;
  counts.reserve(regions.size());
  for (const Region& region : regions) counts.push_back(countRegion(tree, region, min_z, max_z));

  std::ofstream json(json_path);
  if (!json) return 4;
  json << std::fixed << std::setprecision(6)
       << "{\n  \"schema\": \"simenv_room_map_statistics_v1\",\n"
       << "  \"map_file\": \"" << map_path << "\",\n"
       << "  \"resolution\": " << tree.getResolution() << ",\n"
       << "  \"analysis_z_range\": [" << min_z << ", " << max_z << "],\n"
       << "  \"map_bounds\": {\"min\": [" << map_min_x << ',' << map_min_y << ','
       << map_min_z << "], \"max\": [" << map_max_x << ',' << map_max_y << ','
       << map_max_z << "]},\n  \"regions\": {\n";
  for (size_t index = 0; index < regions.size(); ++index) {
    const Region& region = regions[index];
    const Counts& value = counts[index];
    const uint64_t total = value.free + value.occupied + value.unknown;
    const double coverage = total ? static_cast<double>(value.free + value.occupied) / total : 0.0;
    const bool bounds_cover = region.min_x >= map_min_x && region.max_x <= map_max_x &&
                              region.min_y >= map_min_y && region.max_y <= map_max_y &&
                              min_z >= map_min_z && max_z <= map_max_z;
    json << "    \"" << region.name << "\": {\"bounds_map\": ["
         << region.min_x << ',' << region.min_y << ',' << region.max_x << ',' << region.max_y
         << "], \"free_voxel_count\": " << value.free
         << ", \"occupied_voxel_count\": " << value.occupied
         << ", \"unknown_voxel_count\": " << value.unknown
         << ", \"total_voxel_count\": " << total
         << ", \"coverage_ratio\": " << coverage
         << ", \"map_bounds_cover_region\": " << (bounds_cover ? "true" : "false") << '}';
    json << (index + 1 == regions.size() ? "\n" : ",\n");
  }
  json << "  }\n}\n";

  // A compact 2-D column classification for the diagnostic renderer.
  const Region& floor = regions.front();
  std::ofstream csv(csv_path);
  if (!csv) return 5;
  csv << "x,y,state\n";
  const double resolution = tree.getResolution();
  for (double y = floor.min_y + resolution * 0.5; y < floor.max_y; y += resolution)
    for (double x = floor.min_x + resolution * 0.5; x < floor.max_x; x += resolution) {
      bool occupied = false, free = false;
      for (double z = min_z + resolution * 0.5; z < max_z; z += resolution) {
        const int state = classify(tree, x, y, z);
        occupied = occupied || state == 2;
        free = free || state == 1;
      }
      csv << x << ',' << y << ',' << (occupied ? 2 : (free ? 1 : 0)) << '\n';
    }
  return 0;
}
