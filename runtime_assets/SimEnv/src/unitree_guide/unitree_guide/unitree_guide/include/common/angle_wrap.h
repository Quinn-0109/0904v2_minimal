#ifndef UNITREE_ANGLE_WRAP_H
#define UNITREE_ANGLE_WRAP_H

#include <cmath>

// Wrap an angle to (-pi, pi].
inline float wrapToPi(float angle)
{
    if(!std::isfinite(angle)){
        return angle;
    }
    float result = std::fmod(angle + static_cast<float>(M_PI),
                             static_cast<float>(2.0 * M_PI));
    if(result < 0.0f){
        result += static_cast<float>(2.0 * M_PI);
    }
    return result - static_cast<float>(M_PI);
}

// Shortest signed angular difference from angleB to angleA, in (-pi, pi].
inline float shortestAngleDiff(float angleA, float angleB)
{
    return wrapToPi(angleA - angleB);
}

// Return the 2*pi-equivalent of `target` that is closest to `reference`.
// Gazebo may report joints (e.g. the four calves) as canonical + 2*pi, so
// commanding or interpolating toward the raw canonical value would take the
// long way around the circle.
inline float nearestEquivalentAngle(float reference, float target)
{
    return reference + shortestAngleDiff(target, reference);
}

#endif  // UNITREE_ANGLE_WRAP_H
