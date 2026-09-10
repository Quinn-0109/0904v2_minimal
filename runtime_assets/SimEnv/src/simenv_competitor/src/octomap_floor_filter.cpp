#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <octomap/OcTree.h>

int main(int argc, char** argv) {
  if (argc != 5 && argc != 11) {
    std::cerr << "usage: octomap_floor_filter INPUT.bt OUTPUT.bt Z_MIN Z_MAX\n"
              << "       [CLEAR_X_MIN CLEAR_X_MAX CLEAR_Y_MIN CLEAR_Y_MAX "
                 "CLEAR_Z_MIN CLEAR_Z_MAX]\n";
    return 2;
  }
  const std::string input = argv[1];
  const std::string output = argv[2];
  const double z_min = std::atof(argv[3]);
  const double z_max = std::atof(argv[4]);
  if (!std::isfinite(z_min) || !std::isfinite(z_max) || z_min >= z_max) {
    std::cerr << "invalid z band\n";
    return 2;
  }

  // A preloaded upper-floor map can contain a robot/scan ghost from the
  // source run.  The caller may provide a truth-verified corridor mask to
  // remove only that dynamic centre-lane residue.  Room walls and furniture
  // remain untouched because the mask is explicit and narrow.
  const bool clear_mask = argc == 11;
  double clear_x_min = 0.0, clear_x_max = 0.0, clear_y_min = 0.0,
         clear_y_max = 0.0, clear_z_min = 0.0, clear_z_max = 0.0;
  if (clear_mask) {
    clear_x_min = std::atof(argv[5]);
    clear_x_max = std::atof(argv[6]);
    clear_y_min = std::atof(argv[7]);
    clear_y_max = std::atof(argv[8]);
    clear_z_min = std::atof(argv[9]);
    clear_z_max = std::atof(argv[10]);
    if (!std::isfinite(clear_x_min) || !std::isfinite(clear_x_max) ||
        !std::isfinite(clear_y_min) || !std::isfinite(clear_y_max) ||
        !std::isfinite(clear_z_min) || !std::isfinite(clear_z_max) ||
        clear_x_min >= clear_x_max || clear_y_min >= clear_y_max ||
        clear_z_min >= clear_z_max) {
      std::cerr << "invalid clear mask\n";
      return 2;
    }
  }

  octomap::OcTree source(input);
  octomap::OcTree filtered(source.getResolution());
  filtered.setProbHit(source.getProbHit());
  filtered.setProbMiss(source.getProbMiss());
  filtered.setClampingThresMin(source.getClampingThresMin());
  filtered.setClampingThresMax(source.getClampingThresMax());
  filtered.setOccupancyThres(source.getOccupancyThres());

  std::size_t copied = 0;
  for (auto it = source.begin_leafs(); it != source.end_leafs(); ++it) {
    const double half = it.getSize() * 0.5;
    const double center_z = it.getCoordinate().z();
    if (center_z + half < z_min || center_z - half > z_max) continue;
    const auto coordinate = it.getCoordinate();
    const bool in_clear_mask = clear_mask &&
        coordinate.x() >= clear_x_min && coordinate.x() <= clear_x_max &&
        coordinate.y() >= clear_y_min && coordinate.y() <= clear_y_max &&
        center_z + half >= clear_z_min && center_z - half <= clear_z_max;
    if (in_clear_mask) {
      // Preserve an explicit free voxel rather than turning it into unknown;
      // this makes the corridor clearance usable by conservative planners.
      filtered.setNodeValue(it.getKey(), source.getProbMiss());
      ++copied;
      continue;
    }
    filtered.setNodeValue(it.getKey(), it->getValue());
    ++copied;
  }
  filtered.updateInnerOccupancy();
  if (!filtered.writeBinary(output)) {
    std::cerr << "failed to write " << output << "\n";
    return 1;
  }
  std::cout << "{\"input\":\"" << input << "\",\"output\":\""
            << output << "\",\"z_min\":" << z_min << ",\"z_max\":"
            << z_max << ",\"copied_leaf_count\":" << copied
            << ",\"resolution\":" << filtered.getResolution() << "}\n";
  return 0;
}
