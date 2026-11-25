#!/usr/bin/env python3
# coding=utf-8
"""
Conecta as saídas do IPMNode (frame base_footprint) + localização
ao canal /planning/world_state do TrajectoryPlanner.

Modos de pose (`pose_source`):
  - 'local'        : sem localização; planeja no próprio frame base_footprint (robô em 0,0,0)
  - 'tf'           : lê TF (field_frame -> robot_frame)
  - 'odom'         : assina nav_msgs/Odometry em `pose_topic`
  - 'pose_stamped' : assina geometry_msgs/PoseStamped em `pose_topic`
  - 'pose_cov'     : assina geometry_msgs/PoseWithCovarianceStamped em `pose_topic`
"""

import math
import time

import rclpy
import soccer_vision_3d_msgs.msg as sv3dm
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from .campoVetorial import calcular_alvo_chute


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def to_field(me, x: float, y: float):
    """Transforma ponto (x, y) de base_footprint para o frame do campo."""
    c, s = math.cos(me[2]), math.sin(me[2])
    return me[0] + c * x - s * y, me[1] + s * x + c * y


def valid_xy(x: float, y: float, max_range: float) -> bool:
    """Descarta NaN, infinito e projeções IPM muito distantes (imprecisas perto do horizonte)."""
    return math.isfinite(x) and math.isfinite(y) and math.hypot(x, y) <= max_range


def extract_conf(item) -> float:
    """Extrai o float de confiança (suporta Confidence.confidence ou float direto)."""
    return float(getattr(item.confidence, 'confidence', item.confidence))


class WorldSubscriber(Node):

    def __init__(self):
        super().__init__('world_subscriber')
        p = self.declare_parameter

        # Fonte da pose global do robô
        self.pose_source = str(p('pose_source', 'local').value)
        if self.pose_source not in ('local', 'tf', 'odom', 'pose_stamped', 'pose_cov'):
            raise ValueError(
                "pose_source deve ser 'local', 'tf', 'odom', 'pose_stamped' ou 'pose_cov'."
            )
        self.pose_topic = str(p('pose_topic', '/odom').value)
        self.field_frame = str(p('field_frame', 'map').value)
        self.robot_frame = str(p('robot_frame', 'base_footprint').value)

        # Nomes dos tópicos publicados pelo IPMNode
        balls_topic = str(p('ipm.balls_topic', 'balls_relative').value)
        posts_topic = str(p('ipm.posts_topic', 'goal_posts_relative').value)
        robots_topic = str(p('ipm.robots_topic', 'robots_relative').value)

        # Limites do campo CBR 2026 (9x6 m + faixa lateral de 80 cm -> 11x8 m)
        self.field_bounds = (
            float(p('field.min_x', -5.2).value),
            float(p('field.max_x', 5.2).value),
            float(p('field.min_y', -3.7).value),
            float(p('field.max_y', 3.7).value),
        )

        # Filtros e raios de obstáculos baseados nas saídas do IPMNode
        self.max_range = float(p('ipm.max_range', 5.0).value)
        self.min_confidence = float(p('ipm.min_confidence', 0.0).value)
        self.robot_min_radius = float(p('obstacles.robot_radius', 0.25).value)
        self.post_min_radius = float(p('obstacles.post_radius', 0.08).value)
        self.min_self_dist = float(p('obstacles.min_self_dist', 0.30).value)

        # Gol adversário e aproximação para chute (usados apenas com localização ativa)
        self.opp_goal = (
            float(p('opponent_goal.x', 4.5).value),
            float(p('opponent_goal.y', 0.0).value),
        )
        self.kick_offset = float(p('kick_approach_dist', 0.25).value)

        # TTLs alinhados ao OBJECT_TTL_S = 0.5 s do IPMNode
        self.pose_ttl = float(p('pose_ttl', 1.0).value)
        self.ball_ttl = float(p('ball_ttl', 1.0).value)
        self.obstacle_ttl = float(p('obstacle_ttl', 0.5).value)

        # Estado em memória
        self.me = None              # (monotonic_t, (x, y, yaw))
        self.ball = None            # (monotonic_t, (x, y))
        self.robots = (0.0, [])     # (monotonic_t, [(x, y, r), ...])
        self.posts = (0.0, [])      # (monotonic_t, [(x, y, r), ...])

        # Configura a fonte de pose escolhida
        self.tf_buffer = None
        self.tf_listener = None
        if self.pose_source == 'tf':
            self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)
        elif self.pose_source == 'odom':
            self.create_subscription(Odometry, self.pose_topic, self._odom_cb, 10)
        elif self.pose_source == 'pose_stamped':
            self.create_subscription(PoseStamped, self.pose_topic, self._pose_stamped_cb, 10)
        elif self.pose_source == 'pose_cov':
            self.create_subscription(
                PoseWithCovarianceStamped, self.pose_topic, self._pose_cov_cb, 10
            )

        # Assina exatamente os tópicos publicados pelo IPMNode (QoS 1 igual ao publisher)
        self.create_subscription(sv3dm.BallArray, balls_topic, self.balls_callback, 1)
        self.create_subscription(sv3dm.GoalpostArray, posts_topic, self.posts_callback, 1)
        self.create_subscription(sv3dm.RobotArray, robots_topic, self.robots_callback, 1)

        self.pub = self.create_publisher(Float64MultiArray, '/planning/world_state', 10)
        self.create_timer(0.05, self.publish_state)  # 20 Hz

        self.get_logger().info(
            f'WorldSubscriber pronto | pose_source={self.pose_source} '
            f'(frame_robo={self.robot_frame}) | '
            f'topicos IPM=({balls_topic}, {posts_topic}, {robots_topic})'
        )

    # ---------------- pose do robô ----------------
    def _store_pose(self, pose_msg):
        x = pose_msg.position.x
        y = pose_msg.position.y
        q = pose_msg.orientation
        if not all(math.isfinite(v) for v in (x, y, q.x, q.y, q.z, q.w)):
            return
        self.me = (time.monotonic(), (x, y, quat_to_yaw(q)))

    def _odom_cb(self, msg: Odometry):
        self._store_pose(msg.pose.pose)

    def _pose_stamped_cb(self, msg: PoseStamped):
        self._store_pose(msg.pose)

    def _pose_cov_cb(self, msg: PoseWithCovarianceStamped):
        self._store_pose(msg.pose.pose)

    def _lookup_tf_pose(self, stamp=None):
        """Busca transformada field_frame -> robot_frame (tenta no stamp da câmera e cai para latest)."""
        if self.tf_buffer is None:
            return None
        query_time = Time.from_msg(stamp) if stamp is not None else Time()
        try:
            tf = self.tf_buffer.lookup_transform(self.field_frame, self.robot_frame, query_time)
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform(self.field_frame, self.robot_frame, Time())
            except TransformException:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        if not all(math.isfinite(v) for v in (t.x, t.y, q.x, q.y, q.z, q.w)):
            return None
        return (t.x, t.y, quat_to_yaw(q))

    def _get_pose(self, stamp=None):
        if self.pose_source == 'local':
            return (0.0, 0.0, 0.0)
        if self.pose_source == 'tf':
            return self._lookup_tf_pose(stamp)
        if self.me is None:
            return None
        t, pose = self.me
        if (time.monotonic() - t) > self.pose_ttl:
            return None
        return pose

    def _check_frame(self, header):
        if header.frame_id and header.frame_id != self.robot_frame:
            self.get_logger().warn(
                f"Frame do IPM '{header.frame_id}' difere de robot_frame='{self.robot_frame}'.",
                throttle_duration_sec=5.0,
            )

    def _active_bounds(self):
        if self.pose_source == 'local':
            r = self.max_range + 0.5
            return (-r, r, -r, r)
        return self.field_bounds

    @staticmethod
    def _clamp_target(target, bounds, margin=0.25):
        min_x, max_x, min_y, max_y = bounds
        tx = max(min_x + margin, min(max_x - margin, target[0]))
        ty = max(min_y + margin, min(max_y - margin, target[1]))
        return (tx, ty)

    # ---------------- callbacks do IPMNode ----------------
    def balls_callback(self, msg: sv3dm.BallArray):
        self._check_frame(msg.header)
        if not msg.balls:
            return
        pose = self._get_pose(msg.header.stamp)
        if pose is None:
            return

        valid_balls = [
            b for b in msg.balls
            if valid_xy(b.center.x, b.center.y, self.max_range)
            and extract_conf(b) >= self.min_confidence
        ]
        if not valid_balls:
            return

        best = max(valid_balls, key=extract_conf)
        self.ball = (time.monotonic(), to_field(pose, best.center.x, best.center.y))

    def robots_callback(self, msg: sv3dm.RobotArray):
        self._check_frame(msg.header)
        if not msg.robots:
            # Não zera na hora para evitar piscada de 1 frame do YOEO; expira via obstacle_ttl
            return
        pose = self._get_pose(msg.header.stamp)
        if pose is None:
            return

        pts = []
        for r in msg.robots:
            rx = r.bb.center.position.x
            ry = r.bb.center.position.y
            if not valid_xy(rx, ry, self.max_range):
                continue
            if extract_conf(r) < self.min_confidence:
                continue
            if math.hypot(rx, ry) <= self.min_self_dist:
                continue
            bb_radius = max(r.bb.size.x, r.bb.size.y) / 2.0
            radius = max(self.robot_min_radius, bb_radius)
            fx, fy = to_field(pose, rx, ry)
            pts.append((fx, fy, radius))

        if pts:
            self.robots = (time.monotonic(), pts)

    def posts_callback(self, msg: sv3dm.GoalpostArray):
        self._check_frame(msg.header)
        if not msg.posts:
            return
        pose = self._get_pose(msg.header.stamp)
        if pose is None:
            return

        pts = []
        for g in msg.posts:
            gx = g.bb.center.position.x
            gy = g.bb.center.position.y
            if not valid_xy(gx, gy, self.max_range):
                continue
            if extract_conf(g) < self.min_confidence:
                continue
            bb_radius = max(g.bb.size.x, g.bb.size.y) / 2.0
            radius = max(self.post_min_radius, bb_radius)
            fx, fy = to_field(pose, gx, gy)
            pts.append((fx, fy, radius))

        if pts:
            self.posts = (time.monotonic(), pts)

    # ---------------- publicação para o TrajectoryPlanner ----------------
    def publish_state(self):
        pose = self._get_pose()
        if pose is None:
            self.get_logger().warn(
                f'Sem pose recente do robô (pose_source={self.pose_source}).',
                throttle_duration_sec=2.0,
            )
            return

        now = time.monotonic()
        ball = self.ball[1] if self.ball and (now - self.ball[0]) < self.ball_ttl else None

        # Sem bola válida: para de publicar para o TrajectoryPlanner/WalkingRoutine entrar em SEARCH
        if ball is None:
            return

        bounds = self._active_bounds()
        min_x, max_x, min_y, max_y = bounds
        if not (min_x < pose[0] < max_x and min_y < pose[1] < max_y):
            self.get_logger().warn(
                f'Pose ({pose[0]:.2f}, {pose[1]:.2f}) fora dos limites {bounds}!',
                throttle_duration_sec=2.0,
            )
            return

        if self.pose_source != 'local' and self.kick_offset > 0.0:
            raw_target, _ = calcular_alvo_chute(ball, self.opp_goal, self.kick_offset)
            target = self._clamp_target(raw_target, bounds)
        else:
            target = self._clamp_target(ball, bounds)

        data = [*bounds, *pose, *target, *ball]
        for t, pts in (self.robots, self.posts):
            if (now - t) < self.obstacle_ttl:
                for x, y, radius in pts:
                    data.extend((x, y, radius))

        msg = Float64MultiArray()
        msg.data = [float(v) for v in data]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WorldSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
