#!/usr/bin/env python3
# coding=utf-8
"""Recebe saídas do IPM e pose da localização e publica /planning/world_state."""

import math
import time

import rclpy
import soccer_vision_3d_msgs.msg as sv3dm
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .campoVetorial import calcular_alvo_chute


def to_field(me, x, y):
    """Converte ponto relativo ao robô (base_footprint) para o frame do campo."""
    c, s = math.cos(me[2]), math.sin(me[2])
    return me[0] + c * x - s * y, me[1] + s * x + c * y


def valid_xy(x, y, max_dist=12.0):
    """Filtra NaN, infinito e projeções absurdas do IPM."""
    return math.isfinite(x) and math.isfinite(y) and math.hypot(x, y) <= max_dist


def extract_conf(ball):
    """Extrai confiança numérica (suporta submensagem Confidence ou float direto)."""
    return float(getattr(ball.confidence, 'confidence', ball.confidence))


class WorldSubscriber(Node):

    def __init__(self):
        super().__init__('world_subscriber')
        p = self.declare_parameter

        # Campo CBR 2026: 9x6 m + faixa lateral de 80 cm (base 11x8 m)
        self.bounds = (
            float(p('field.min_x', -5.2).value),
            float(p('field.max_x', 5.2).value),
            float(p('field.min_y', -3.7).value),
            float(p('field.max_y', 3.7).value),
        )
        self.robot_radius = float(p('obstacles.robot_radius', 0.25).value)
        self.post_radius = float(p('obstacles.post_radius', 0.08).value)
        self.min_self_dist = float(p('obstacles.min_self_dist', 0.30).value)

        # Posição do gol adversário e distância de aproximação atrás da bola (0.0 = ir direto na bola)
        self.opp_goal = (
            float(p('opponent_goal.x', 4.5).value),
            float(p('opponent_goal.y', 0.0).value),
        )
        self.kick_offset = float(p('kick_approach_dist', 0.25).value)

        # Validade temporal (TTL) das leituras
        self.pose_ttl = float(p('pose_ttl', 1.0).value)
        self.ball_ttl = float(p('ball_ttl', 1.0).value)
        self.obstacle_ttl = float(p('obstacle_ttl', 0.5).value)
        pose_topic = str(p('pose_topic', '/odom').value)

        self.me = None             # (instante, (x, y, yaw)) no campo
        self.ball = None           # (instante, (x, y)) no campo
        self.robots = (0.0, [])    # (instante, [(x, y)]) no campo
        self.posts = (0.0, [])     # (instante, [(x, y)]) no campo

        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)
        self.create_subscription(sv3dm.BallArray, 'balls_relative', self.balls_callback, 1)
        self.create_subscription(sv3dm.GoalpostArray, 'goal_posts_relative', self.posts_callback, 1)
        self.create_subscription(sv3dm.RobotArray, 'robots_relative', self.robots_callback, 1)

        self.pub = self.create_publisher(Float64MultiArray, '/planning/world_state', 10)
        self.create_timer(0.05, self.publish_state)  # 20 Hz

        self.get_logger().info(
            f'WorldSubscriber ativo (pose={pose_topic}, gol_adv={self.opp_goal}, '
            f'kick_offset={self.kick_offset}m)'
        )

    def _current_pose(self):
        """Retorna (x, y, yaw) apenas se a pose da localização estiver recente."""
        if self.me is None:
            return None
        t, pose = self.me
        if time.monotonic() - t > self.pose_ttl:
            return None
        return pose

    def _clamp_target(self, target, margin=0.25):
        """Evita que o ponto de aproximação atrás da bola caia fora da grade perto das bordas."""
        min_x, max_x, min_y, max_y = self.bounds
        tx = max(min_x + margin, min(max_x - margin, target[0]))
        ty = max(min_y + margin, min(max_y - margin, target[1]))
        return (tx, ty)

    # ---------------- entradas ----------------
    def pose_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        if not all(math.isfinite(v) for v in (x, y, q.x, q.y, q.z, q.w)):
            return
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.me = (time.monotonic(), (x, y, yaw))

    def balls_callback(self, msg: sv3dm.BallArray):
        pose = self._current_pose()
        if pose is None or not msg.balls:
            return
        valid_balls = [b for b in msg.balls if valid_xy(b.center.x, b.center.y)]
        if not valid_balls:
            return
        best = max(valid_balls, key=extract_conf)
        self.ball = (time.monotonic(), to_field(pose, best.center.x, best.center.y))

    def robots_callback(self, msg: sv3dm.RobotArray):
        pose = self._current_pose()
        if pose is None:
            return
        pts = []
        for r in msg.robots:
            rx, ry = r.bb.center.position.x, r.bb.center.position.y
            #sem olhar pontos em cima do robo
            if valid_xy(rx, ry) and math.hypot(rx, ry) > self.min_self_dist:
                pts.append(to_field(pose, rx, ry))
        self.robots = (time.monotonic(), pts)

    def posts_callback(self, msg: sv3dm.GoalpostArray):
        pose = self._current_pose()
        if pose is None:
            return
        pts = [
            to_field(pose, g.bb.center.position.x, g.bb.center.position.y)
            for g in msg.posts
            if valid_xy(g.bb.center.position.x, g.bb.center.position.y)
        ]
        self.posts = (time.monotonic(), pts)

    # ---------------- saída ----------------
    def publish_state(self):
        pose = self._current_pose()
        if pose is None:
            self.get_logger().warn(
                'Sem pose recente do robô (esperando Odometry).',
                throttle_duration_sec=2.0,
            )
            return

        now = time.monotonic()
        ball = self.ball[1] if self.ball and (now - self.ball[0]) < self.ball_ttl else None

        # Sem bola válida: não publica destino falso
        if ball is None:
            return

        if self.kick_offset > 0.0:
            raw_target, _ = calcular_alvo_chute(ball, self.opp_goal, self.kick_offset)
            target = self._clamp_target(raw_target)
        else:
            target = self._clamp_target(ball)

        data = [*self.bounds, *pose, *target, *ball]
        for (t, pts), radius in ((self.robots, self.robot_radius), (self.posts, self.post_radius)):
            if (now - t) < self.obstacle_ttl:
                for x, y in pts:
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
