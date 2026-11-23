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
    Ativado pelos estados 'walking', 'aligning_body' e 'aligning_foot'.
    """
    def __init__(self):
        super().__init__('walking_routine_node')

        # --- Variáveis de Controle ---
        self.is_active = False
        self.current_fsm_state = ""
        self.planned_positions = []
        self.planned_velocities = []

        # Controle de validade (timestamps monotônicos)
        self.last_plan_time = 0.0
        self.last_ball_seen_time = 0.0

        # --- Parâmetros ---
        self.declare_parameter('walk_speed_x', 0.1)
        self.declare_parameter('search_angular_speed', 0.4)     # Velocidade de giro do corpo ao buscar (rad/s)
        self.declare_parameter('plan_timeout', 1.0)             # Tempo máx. sem receber path planning (s)
        self.declare_parameter('ball_validation_timeout', 5.0)  # Tempo máx. sem validar a bola na visão (s)
        self.declare_parameter('kp_body_align', 0.8)
        self.declare_parameter('kp_foot_strafe', 0.001)
        self.declare_parameter('kp_foot_forward', 0.001)
        self.declare_parameter('kicking_hotspot_x', 320.0)
        self.declare_parameter('kicking_hotspot_y', 400.0)

        self.walk_speed_x = self.get_parameter('walk_speed_x').get_parameter_value().double_value
        self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value
        self.plan_timeout = self.get_parameter('plan_timeout').get_parameter_value().double_value
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
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_walk', 10)

        # --- Timer ---
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info("Rotina de Caminhada e Alinhamento pronta e ouvindo ordens.")

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
        self.current_fsm_state = msg.current_state

        if self.current_fsm_state in ['walking', 'aligning_body', 'aligning_foot']:
            if not self.is_active:
                self.get_logger().info(f"Ordem '{self.current_fsm_state}' recebida! Ativando rotina.")
            self.is_active = True
        else:
            if self.is_active:
                self.get_logger().info(f"Ordem '{self.current_fsm_state}' recebida. Desativando rotina.")
            self.is_active = False

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
            self.execute_walk_forward()
        elif self.current_fsm_state == 'aligning_body':
            self.publish_head_state("TRACKING")
            self.execute_body_alignment()
        elif self.current_fsm_state == 'aligning_foot':
            self.publish_head_state("TRACKING")
            self.execute_foot_alignment()

    def execute_walk_forward(self):
        """
        Chaveamento da caminhada:
        - Mantém a cabeça em 'SEARCHING' em ambos os casos.
        - Se houver path planning recente E a bola tiver sido validada dentro do timeout:
          segue as velocidades (dx, dy, psi) do path planning.
        - Caso contrário (sem path planning ou muito tempo sem validar a bola):
          volta ao modo de busca girando o corpo no próprio eixo.
        """
        now = time.monotonic()
        has_valid_plan = (
            bool(self.planned_velocities)
            and (now - self.last_plan_time) <= self.plan_timeout
        )
        ball_validated = (now - self.last_ball_seen_time) <= self.ball_validation_timeout

        #cabeca permanece buscando referências/bola durante toda a navegação
        self.publish_head_state("SEARCHING")

        twist_command = Twist()

        if not has_valid_plan or not ball_validated: #gira e procura a bola
            twist_command.linear.x = 0.0
            twist_command.linear.y = 0.0
            twist_command.angular.z = self.search_angular_speed

            motivo = "sem path planning válido" if not has_valid_plan else "tempo limite sem validar a bola"
            self.get_logger().info(
                f"Modo SEARCH (girando corpo): {motivo}.", throttle_duration_sec=1.0
            )
        else:
            # Modo navegação: segue o que o path planning mandou
            dx, dy, psi = self.planned_velocities[0]
            twist_command.linear.x = float(dx)
            twist_command.linear.y = float(dy)
            twist_command.angular.z = float(psi)

        self.cmd_vel_pub.publish(twist_command)

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
