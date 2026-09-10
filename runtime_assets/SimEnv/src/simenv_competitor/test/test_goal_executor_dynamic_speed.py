#!/usr/bin/env python3
"""Regression tests for role-specific executable speed floors."""

import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from goal_executor_core import bounded_dynamic_minimum_speed  # noqa: E402


def test_portal_floor_can_be_lower_than_corridor_cruise_floor():
    # Full-flow uses a 0.65 m/s corridor floor and explicitly requests 0.18
    # m/s for the final aligned jamb crossing.
    assert bounded_dynamic_minimum_speed(0.18, 0.15, 2.0) == pytest.approx(0.18)


def test_dynamic_floor_is_bounded_by_physical_floor_and_cap():
    assert bounded_dynamic_minimum_speed(0.01, 0.15, 2.0) == pytest.approx(0.15)
    assert bounded_dynamic_minimum_speed(3.0, 0.15, 2.0) == pytest.approx(2.0)


def test_dynamic_floor_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        bounded_dynamic_minimum_speed(math.nan, 0.15, 2.0)


def test_strict_fullflow_launch_disables_gazebo_fall_reset():
    launch_path = os.path.join(
        os.path.dirname(__file__), "..", "launch",
        "baseline_fastlio_exploration.launch")
    with open(launch_path, "r", encoding="utf-8") as stream:
        launch = stream.read()
    assert '<param name="use_gazebo_reset" value="false"/>' in launch
    assert '<param name="use_gazebo_reset" value="true"/>' not in launch


def test_fullflow_room_caps_stay_inside_measured_plane_gait_envelope():
    launch_path = os.path.join(
        os.path.dirname(__file__), "..", "launch",
        "fuel_semantic_fastlio_exploration.launch")
    root = ET.parse(launch_path).getroot()
    defaults = {
        node.attrib["name"]: float(node.attrib["default"])
        for node in root.findall("arg")
        if node.attrib.get("name") in {
            "first_floor_room_entry_speed",
            "first_floor_room_open_transition_speed",
            "first_floor_room_open_transition_lateral_speed",
            "first_floor_room_exit_speed",
            "goal_executor_acceleration",
            "goal_executor_deceleration",
        }
    }
    assert defaults["first_floor_room_entry_speed"] <= 1.20
    assert defaults["first_floor_room_open_transition_speed"] <= 1.20
    assert defaults["first_floor_room_open_transition_lateral_speed"] <= 0.30
    assert defaults["first_floor_room_exit_speed"] <= 0.95
    assert defaults["goal_executor_acceleration"] <= 1.00
    assert defaults["goal_executor_deceleration"] <= 1.20
