#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <octomap/OcTree.h>

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: octomap_floor_projection_exporter MAP.bt OUT.pgm "
                 "OUT.json Z_MIN Z_MAX\n";
    return 2;
  }
  const std::string map_path = argv[1], pgm_path = argv[2], json_path = argv[3];
  const double z_min = std::atof(argv[4]), z_max = std::atof(argv[5]);
  if (!std::isfinite(z_min) || !std::isfinite(z_max) || z_min >= z_max) {
    std::cerr << "invalid height interval\n";
    return 2;
  }

  octomap::OcTree tree(map_path);
  if (tree.size() <= 1) {
    std::cerr << "OctoMap is empty\n";
    return 3;
  }
  double min_x, min_y, min_z, max_x, max_y, max_z;
  tree.getMetricMin(min_x, min_y, min_z);
  tree.getMetricMax(max_x, max_y, max_z);
  const double resolution = tree.getResolution();
  min_x = std::floor((min_x - resolution) / resolution) * resolution;
  min_y = std::floor((min_y - resolution) / resolution) * resolution;
  max_x = std::ceil((max_x + resolution) / resolution) * resolution;
  max_y = std::ceil((max_y + resolution) / resolution) * resolution;
  const size_t width = std::max<size_t>(1, std::ceil((max_x - min_x) / resolution));
  const size_t height = std::max<size_t>(1, std::ceil((max_y - min_y) / resolution));
  if (width > 5000 || height > 5000 || width * height > 25000000) {
    std::cerr << "projection bounds are too large\n";
    return 4;
  }

  // PGM convention: 205 unknown, 254 observed free, 0 occupied. The array's
  // first row is minimum world y; the visualizer renders it with origin=lower.
  std::vector<uint8_t> pixels(width * height, 205);
  uint64_t free_cells = 0, occupied_cells = 0;
  for (int occupied_pass = 0; occupied_pass < 2; ++occupied_pass) {
    for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
      const bool occupied = tree.isNodeOccupied(*it);
      if (occupied != (occupied_pass == 1)) continue;
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
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          uint8_t& pixel = pixels[static_cast<size_t>(y) * width + x];
          if (occupied) {
            if (pixel == 254 && free_cells > 0) --free_cells;
            if (pixel != 0) ++occupied_cells;
            pixel = 0;
          } else if (pixel == 205) {
            pixel = 254;
            ++free_cells;
          }
        }
      }
    }
  }

  std::ofstream pgm(pgm_path, std::ios::binary);
  pgm << "P5\n" << width << ' ' << height << "\n255\n";
  pgm.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
  if (!pgm) {
    std::cerr << "could not write PGM\n";
    return 5;
  }
  std::ofstream metadata(json_path);
  metadata << std::fixed << std::setprecision(9)
           << "{\n  \"schema\": \"simenv_octomap_floor_projection_v1\",\n"
           << "  \"source_map\": \"" << map_path << "\",\n"
           << "  \"resolution\": " << resolution << ",\n"
           << "  \"origin_x\": " << min_x << ",\n"
           << "  \"origin_y\": " << min_y << ",\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"z_min\": " << z_min << ",\n"
           << "  \"z_max\": " << z_max << ",\n"
           << "  \"free_cells\": " << free_cells << ",\n"
           << "  \"occupied_cells\": " << occupied_cells << "\n}\n";
  if (!metadata) {
    std::cerr << "could not write projection metadata\n";
    return 5;
  }
  std::cout << "{\"success\":true,\"width\":" << width
            << ",\"height\":" << height << ",\"resolution\":"
            << resolution << "}\n";
  return 0;
}
