import colorsys
from typing import List, Sequence, Tuple

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class AprilTagHistoryPublisher(Node):
    """Publishes RViz markers for AprilTag detections."""

    def __init__(self) -> None:
        super().__init__('apriltag_history_publisher')
        det_topic = self.declare_parameter('detections_topic', '/tag_detections').get_parameter_value().string_value
        hist_topic = self.declare_parameter('history_topic', '/cal/extr/history').get_parameter_value().string_value
        scale_param = self.declare_parameter('marker_scale', [0.12, 0.02, 0.02]).get_parameter_value().double_array_value

        self._scale = self._parse_scale(scale_param)
        self._markers: List[Marker] = []
        self._next_id = 0
        self._warned_frame = False

        self._pub = self.create_publisher(MarkerArray, hist_topic, 10)
        self.create_subscription(AprilTagDetectionArray, det_topic, self._on_detections, 10)
        self.get_logger().info(f'Listening for AprilTag detections on {det_topic}')
        self.get_logger().info(f'Publishing history markers on {hist_topic}')

    def _parse_scale(self, scale: Sequence[float]) -> Tuple[float, float, float]:
        if len(scale) != 3:
            self.get_logger().warn('Expected marker_scale to have 3 values, using defaults')
            return 0.12, 0.02, 0.02
        sx, sy, sz = scale
        return float(sx), float(sy), float(sz)

    def _on_detections(self, msg: AprilTagDetectionArray) -> None:
        if not msg.detections:
            return

        stamp = self.get_clock().now().to_msg()
        new_markers = []
        for det in msg.detections:
            pose = det.pose.pose.pose
            frame = det.pose.header.frame_id or msg.header.frame_id
            if not frame:
                if not self._warned_frame:
                    self.get_logger().warn('Detections lack frame_id; defaulting to "map"')
                    self._warned_frame = True
                frame = 'map'
            marker = self._create_marker(pose, frame, stamp, det.id)
            new_markers.append(marker)

        self._markers.extend(new_markers)
        array = MarkerArray()
        array.markers = list(self._markers)
        self._pub.publish(array)

    def _create_marker(self, pose: Pose, frame: str, stamp, tag_id: Sequence[int]) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = 'apriltag_history'
        marker.id = self._next_id
        self._next_id += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x, marker.scale.y, marker.scale.z = self._scale
        marker.color = self._color(tag_id)
        marker.lifetime = Duration()
        return marker

    def _color(self, tag_id: Sequence[int]) -> ColorRGBA:
        color = ColorRGBA()
        tag = tag_id[0] if tag_id else -1
        hue = (tag * 0.61803398875) % 1.0 if tag >= 0 else 0.55
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = 0.85
        return color


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AprilTagHistoryPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
