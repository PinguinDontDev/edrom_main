#!/usr/bin/env python3
# coding=utf-8

import time
import rclpy
from rclpy.node import Node
from modularized_bhv_msgs.msg import CurrentStateMsg
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Path
from edrom_msgs.msg import VisionData


class WalkingRoutine(Node):
    """
    Nó Gerente da Rotina de Caminhada, Busca e Alinhamento.
    Usa o Path Planning por padrão; se ficar sem receber dados válidos por
    `planner_fallback_timeout` segundos (padrão 3.0 s), alterna para o walk antigo.
    """
    def __init__(self):
        super().__init__('walking_routine_node')

        # --- Variáveis de Controle ---
        self.is_active = False
        self.current_fsm_state = ""
        self.planned_positions = []
        self.planned_velocities = []
        self.using_legacy_walk = False

        # Controle de validade (timestamps monotônicos)
        self.last_plan_time = 0.0
        self.walk_active_since = 0.0
        self.last_ball_seen_time = 0.0

        # --- Parâmetros ---
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_walk')
        self.declare_parameter('walk_speed_x', 0.1)
        self.declare_parameter('search_angular_speed', 0.4)      # Giro do corpo ao buscar (rad/s)
        self.declare_parameter('plan_timeout', 1.0)              # Tempo máx. para usar o último plano (s)
        self.declare_parameter('planner_fallback_timeout', 3.0)  # Tempo sem dados para cair no walk antigo (s)
        self.declare_parameter('ball_validation_timeout', 5.0)   # Tempo máx. sem validar a bola na visão (s)
        self.declare_parameter('kp_body_align', 0.8)
        self.declare_parameter('kp_foot_strafe', 0.001)
        self.declare_parameter('kp_foot_forward', 0.001)
        self.declare_parameter('kicking_hotspot_x', 320.0)
        self.declare_parameter('kicking_hotspot_y', 400.0)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.walk_speed_x = self.get_parameter('walk_speed_x').get_parameter_value().double_value
        self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value
        self.plan_timeout = self.get_parameter('plan_timeout').get_parameter_value().double_value
        self.planner_fallback_timeout = self.get_parameter('planner_fallback_timeout').get_parameter_value().double_value
        self.ball_validation_timeout = self.get_parameter('ball_validation_timeout').get_parameter_value().double_value
        self.kp_body_align = self.get_parameter('kp_body_align').get_parameter_value().double_value
        self.kp_foot_strafe = self.get_parameter('kp_foot_strafe').get_parameter_value().double_value
        self.kp_foot_forward = self.get_parameter('kp_foot_forward').get_parameter_value().double_value
        self.kicking_hotspot_x = self.get_parameter('kicking_hotspot_x').get_parameter_value().double_value
        self.kicking_hotspot_y = self.get_parameter('kicking_hotspot_y').get_parameter_value().double_value

        # --- Variáveis de Sensores ---
        self.head_pan_angle = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0

        self.timer_period = 0.05  # 20 Hz

        # --- Subscribers ---
        self.pos_array_sub = self.create_subscription(
            Path, '/planning/positions_array', self.positions_callback, 10)
        self.vel_array_sub = self.create_subscription(
            Float64MultiArray, '/planning/velocities_array', self.velocities_callback, 10)
        self.state_sub = self.create_subscription(
            CurrentStateMsg, '/transitions_and_states/state_machine', self.state_callback, 10)
        self.head_feedback_sub = self.create_subscription(
            JointState, '/goal_joint_states', self.head_feedback_callback, 10)
        self.vision_sub = self.create_subscription(
            VisionData, 'vision2BhvTopic', self.vision_callback, 10)

        # --- Publishers ---
        self.head_control_pub = self.create_publisher(String, '/head_control/state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # --- Timer ---
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info(
            f"Rotina de Caminhada pronta (cmd_vel={cmd_vel_topic}, "
            f"fallback={self.planner_fallback_timeout:.1f}s)."
        )

    def positions_callback(self, msg: Path):
        """Armazena as posições planejadas."""
        self.planned_positions = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def velocities_callback(self, msg: Float64MultiArray):
        """Armazena as velocidades planejadas (dx, dy, psi) e atualiza o timestamp."""
        data = msg.data
        if len(data) >= 3:
            self.planned_velocities = [
                (data[i], data[i + 1], data[i + 2]) for i in range(0, len(data) - 2, 3)
            ]
            self.last_plan_time = time.monotonic()
        else:
            self.planned_velocities = []

    def state_callback(self, msg: CurrentStateMsg):
        """Ativa ou desativa a rotina e armazena o estado atual."""
        prev_state = self.current_fsm_state
        self.current_fsm_state = msg.current_state

        if self.current_fsm_state in ['walking', 'aligning_body', 'aligning_foot']:
            if not self.is_active:
                self.get_logger().info(f"Ordem '{self.current_fsm_state}' recebida! Ativando rotina.")
            if self.current_fsm_state == 'walking' and prev_state != 'walking':
                # Inicia a contagem dos 3s a partir do momento em que começa a andar
                self.walk_active_since = time.monotonic()
            self.is_active = True
        else:
            if self.is_active:
                self.get_logger().info(f"Ordem '{self.current_fsm_state}' recebida. Desativando rotina.")
                self.cmd_vel_pub.publish(Twist())  # Para o robô uma única vez ao sair
            self.is_active = False
            self.using_legacy_walk = False

    def head_feedback_callback(self, msg: JointState):
        """Armazena o ângulo de pan da cabeça."""
        try:
            index = msg.name.index('head_pan')
            self.head_pan_angle = msg.position[index]
        except (ValueError, IndexError):
            pass

    def vision_callback(self, msg: VisionData):
        """Armazena a posição da bola na imagem e registra o instante da validação."""
        if msg.ball.found:
            self.ball_x = float(msg.ball.x)
            self.ball_y = float(msg.ball.y)
            self.last_ball_seen_time = time.monotonic()

    def publish_head_state(self, state_name: str):
        """Publica o modo desejado para o especialista da cabeça."""
        head_command = String()
        head_command.data = state_name
        self.head_control_pub.publish(head_command)

    def control_loop(self):
        """Se a rotina estiver ativa, executa a sub-rotina correspondente."""
        if not self.is_active:
            return

        if self.current_fsm_state == 'walking':
            self.execute_walk()
        elif self.current_fsm_state == 'aligning_body':
            self.publish_head_state("TRACKING")
            self.execute_body_alignment()
        elif self.current_fsm_state == 'aligning_foot':
            self.publish_head_state("TRACKING")
            self.execute_foot_alignment()

    def execute_walk(self):
        """
        Decide entre o walk novo (Path Planning) e o walk antigo (fallback):
        - Se passar `planner_fallback_timeout` (3s) sem receber dados do Path Planning,
          alterna para o walk antigo.
        - Se os dados do Path Planning voltarem, retoma automaticamente o walk novo.
        """
        now = time.monotonic()
        time_without_plan = now - max(self.last_plan_time, self.walk_active_since)

        if time_without_plan >= self.planner_fallback_timeout:
            if not self.using_legacy_walk:
                self.using_legacy_walk = True
                self.get_logger().warn(
                    f"Path planning há {time_without_plan:.1f}s sem enviar dados. "
                    "Alternando para o walk antigo (fallback)."
                )
            self.execute_legacy_walk_forward()
            return

        if self.using_legacy_walk:
            self.using_legacy_walk = False
            self.get_logger().info("Dados do path planning retomados! Voltando ao walk novo.")

        self.execute_planned_walk(now)

    def execute_planned_walk(self, now: float):
        """Walk novo: segue (dx, dy, psi) do Path Planning ou gira procurando a bola."""
        has_valid_plan = (
            bool(self.planned_velocities)
            and (now - self.last_plan_time) <= self.plan_timeout
        )
        ball_validated = (now - self.last_ball_seen_time) <= self.ball_validation_timeout

        self.publish_head_state("SEARCHING")

        twist_command = Twist()

        if not has_valid_plan or not ball_validated:
            twist_command.linear.x = 0.0
            twist_command.linear.y = 0.0
            twist_command.angular.z = self.search_angular_speed

            motivo = "sem path planning válido" if not has_valid_plan else "tempo limite sem validar a bola"
            self.get_logger().info(
                f"Modo SEARCH (girando corpo): {motivo}.", throttle_duration_sec=1.0
            )
        else:
            dx, dy, psi = self.planned_velocities[0]
            twist_command.linear.x = float(dx)
            twist_command.linear.y = float(dy)
            twist_command.angular.z = float(psi)

        self.cmd_vel_pub.publish(twist_command)

    def execute_legacy_walk_forward(self):
        """Walk antigo (fallback): cabeça em TRACKING e corpo andando para frente."""
        self.publish_head_state("TRACKING")

        twist_command = Twist()
        twist_command.linear.x = self.walk_speed_x
        self.cmd_vel_pub.publish(twist_command)
        self.get_logger().info(
            "Fallback (walk antigo): Andar para frente.", throttle_duration_sec=1.0
        )

    def execute_body_alignment(self):
        """Comanda o robô para girar no eixo até alinhar o corpo com a cabeça."""
        twist_command = Twist()
        twist_command.angular.z = self.kp_body_align * self.head_pan_angle
        self.cmd_vel_pub.publish(twist_command)
        self.get_logger().info(
            f"Executando: Alinhar corpo (Ângulo da cabeça: {self.head_pan_angle:.2f})",
            throttle_duration_sec=1.0
        )

    def execute_foot_alignment(self):
        """Comanda micro-passos para alinhar o pé com a bola."""
        error_x = self.kicking_hotspot_x - self.ball_x
        error_y = self.kicking_hotspot_y - self.ball_y

        twist_command = Twist()
        twist_command.linear.y = self.kp_foot_strafe * error_x
        twist_command.linear.x = self.kp_foot_forward * error_y

        self.cmd_vel_pub.publish(twist_command)
        self.get_logger().info(
            f"Executando: Alinhar pé (Erro X: {error_x:.1f}, Erro Y: {error_y:.1f})",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    walking_routine = WalkingRoutine()
    try:
        rclpy.spin(walking_routine)
    except KeyboardInterrupt:
        pass
    finally:
        walking_routine.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
