from typing import Optional

import tf2_ros
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
import rclpy
from rclpy.time import Duration, Time

from easy_handeye2.handeye_calibration import HandeyeCalibrationParameters
from easy_handeye2.handeye_dataset import (DiversityConfig, DiversityFilter, SampleDataset, SampleGate,
                                           SampleGateConfig)
from easy_handeye2_msgs.msg import Sample, SampleList


class HandeyeSampler:
    """
    Manages the samples acquired from tf.
    """

    def __init__(self, node: rclpy.node.Node, handeye_parameters: HandeyeCalibrationParameters):
        self.node = node
        self.handeye_parameters = handeye_parameters

        # tf structures
        self.tfBuffer: tf2_ros.Buffer = Buffer(cache_time=Duration(seconds=2), node=node)
        """
        used to get transforms to build each sample
        """
        self.tfListener: tf2_ros.TransformListener = TransformListener(self.tfBuffer, self.node, spin_thread=True)
        """
        used to get transforms to build each sample
        """
        self.tfBroadcaster: tf2_ros.TransformBroadcaster = TransformBroadcaster(self.node)
        """
        used to publish the calibration after saving it
        """

        gate_cfg = SampleGateConfig(
            min_translation=float(self.node.declare_parameter('auto_snapshot.translation_gate', 0.01).value),
            min_rotation_deg=float(self.node.declare_parameter('auto_snapshot.rotation_gate_deg', 5.0).value),
            min_interval_sec=float(self.node.declare_parameter('auto_snapshot.min_interval', 0.75).value),
        )
        diversity_cfg = DiversityConfig(
            min_translation=float(self.node.declare_parameter('diversity.min_translation', 0.02).value),
            min_rotation_deg=float(self.node.declare_parameter('diversity.min_rotation_deg', 7.5).value),
            max_samples=int(self.node.declare_parameter('diversity.max_samples', 250).value),
        )
        self.dataset = SampleDataset(self.node, handeye_parameters, SampleGate(gate_cfg), DiversityFilter(diversity_cfg))
        self.samples: SampleList = self.dataset.samples
        self.auto_snapshot_period = float(self.node.declare_parameter('auto_snapshot.period', 0.0).value)
        self.auto_snapshot_timer = None
        if self.auto_snapshot_period > 0.0:
            self.auto_snapshot_timer = self.node.create_timer(self.auto_snapshot_period, self._auto_snapshot_callback)

    def wait_for_tf_init(self) -> bool:
        """
        Waits until all needed frames are present in tf.
        """
        base_frame = self.handeye_parameters.robot_base_frame
        effector_frame = self.handeye_parameters.robot_effector_frame
        camera_frame = self.handeye_parameters.tracking_base_frame
        marker_frame = self.handeye_parameters.tracking_marker_frame
        self.node.get_logger().info('Checking that the expected transforms are available in tf')
        self.node.get_logger().info(f'Robot transform: {base_frame} -> {effector_frame}')
        self.node.get_logger().info(f'Tracking transform: {camera_frame} -> {marker_frame}')
        try:
            self.tfBuffer.lookup_transform(base_frame, effector_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the robot base and hand do not seem to be connected')
            self.node.get_logger().error('Run the following command and check its output:')
            self.node.get_logger().error(f'ros2 run tf2_ros tf2_echo {base_frame} {effector_frame}')
            self.node.get_logger().error(
                f'You may need to correct the base_frame or effector_frame argument passed to the easy_handeye2 launch file')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False

        try:
            self.tfBuffer.lookup_transform(camera_frame, marker_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the tracking system base/camera and marker do not seem to be connected')
            self.node.get_logger().error('Run the following command and check its output:')
            self.node.get_logger().error(f'ros2 run tf2_ros tf2_echo {camera_frame} {marker_frame}')
            self.node.get_logger().error(
                f'You may need to correct the base_frame or effector_frame argument passed to the easy_handeye2 launch file')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False

        self.node.get_logger().info('All expected transforms are available on tf; ready to take samples')
        return True

    def _get_transforms(self, time: Optional[rclpy.time.Time] = None) -> Sample | None:
        """
        Samples the transforms at the given time.
        """
        if time is None:
            time = self.node.get_clock().now() - rclpy.time.Duration(nanoseconds=200000000)

        # here we trick the library (it is actually made for eye_in_hand only). Trust me, I'm an engineer
        try:
            if self.handeye_parameters.calibration_type == 'eye_in_hand':
                robot = self.tfBuffer.lookup_transform(self.handeye_parameters.robot_base_frame,
                                                       self.handeye_parameters.robot_effector_frame, time,
                                                       Duration(seconds=1))
            else:
                robot = self.tfBuffer.lookup_transform(self.handeye_parameters.robot_effector_frame,
                                                       self.handeye_parameters.robot_base_frame, time,
                                                       Duration(seconds=1))
            tracking = self.tfBuffer.lookup_transform(self.handeye_parameters.tracking_base_frame,
                                                      self.handeye_parameters.tracking_marker_frame, time,
                                                      Duration(seconds=1))
        except tf2_ros.ExtrapolationException as e:
            self.node.get_logger().error(f'Failed to get the tracking transform: {e}')
            return None

        ret = Sample()
        ret.robot = robot.transform
        ret.tracking = tracking.transform
        return ret

    def current_transforms(self) -> Sample | None:
        return self._get_transforms()

    def take_sample(self) -> bool:
        """Samples the transformations and appends the sample to the dataset."""
        try:
            self.node.get_logger().info('Taking a sample...')
            self.node.get_logger().info('all frames: ' + self.tfBuffer.all_frames_as_string())
            sample = self._get_transforms()
            if sample is None:
                return False
            diff = self.dataset.difference_to_last(sample)
            accepted = self.dataset.add(sample)
            self.samples = self.dataset.samples
            if accepted:
                if diff:
                    self.node.get_logger().info(
                        f'Accepted sample #{len(self.samples.samples)} '
                        f'(Δt={diff.translation:.4f}m Δr={diff.rotation_deg:.2f}°)')
                else:
                    self.node.get_logger().info(f'Accepted first sample')
            else:
                self.node.get_logger().info('Sample rejected by auto gate')
            return accepted
        except Exception as exc:  # pragma: no cover - defensive
            self.node.get_logger().error(f'Failed to take sample: {exc}')
            return False

    def _auto_snapshot_callback(self):
        sample = self._get_transforms()
        if sample is None:
            return
        diff = self.dataset.difference_to_last(sample)
        if self.dataset.add(sample):
            self.samples = self.dataset.samples
            if diff:
                self.node.get_logger().info(
                    f'Auto snapshot accepted (Δt={diff.translation:.4f}m Δr={diff.rotation_deg:.2f}°)')
            else:
                self.node.get_logger().info('Auto snapshot seeded first sample')

    def remove_sample(self, index: int) -> int:
        """Removes a sample from the list and persists the change."""
        self.dataset.remove(index)
        self.samples = self.dataset.samples
        return len(self.samples.samples)

    def get_samples(self) -> easy_handeye2_msgs.msg.SampleList:
        """Returns the samples accumulated so far."""
        return self.dataset.samples

    def load_samples(self) -> bool:
        loaded = self.dataset.load()
        if loaded:
            self.samples = self.dataset.samples
        return loaded

    def save_samples(self) -> bool:
        return self.dataset.save()
