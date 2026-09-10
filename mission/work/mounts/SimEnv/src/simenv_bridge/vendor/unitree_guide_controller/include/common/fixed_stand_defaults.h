#ifndef UNITREE_FIXED_STAND_DEFAULTS_H
#define UNITREE_FIXED_STAND_DEFAULTS_H

// Single source of the hardcoded default FixedStand target pose.
// Joint order: FR/FL/RR/RL x (hip, thigh, calf), matching the FSM/URDF order.
// Gazebo joint_states is name-sorted (FL/FR/RL/RR), so captured snapshots must
// be re-indexed before use.  ROS-free so the deterministic regression test and
// the production State_FixedStand share the same definition.
inline void fixedStandDefaultTargetPos(float *targetPos)
{
    const float legDefault[3] = {0.0f, 0.9f, -1.8f};
    for(int leg = 0; leg < 4; ++leg){
        for(int j = 0; j < 3; ++j){
            targetPos[leg * 3 + j] = legDefault[j];
        }
    }
}

#endif  // UNITREE_FIXED_STAND_DEFAULTS_H
