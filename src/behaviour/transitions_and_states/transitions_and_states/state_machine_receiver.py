#!/usr/bin/env python3
# coding=utf-8
"""
Node que recebe variáveis da máquina de estados, sinal de término de chute
e dados do GameController, reduzindo o GC para 3 estados: PLAY, STOP ou PENALIZED.
"""

import rclpy
from rclpy.node import Node
from modularized_bhv_msgs.msg import StateMachineMsg, CurrentStateMsg, GameControllerMsg
from std_msgs.msg import Bool
from .state_machine import StateMachine
from .behaviour_parameters import BehaviourParameters


class StateMachineReceiver(Node):

    def __init__(self):
        super().__init__('state_machine_receiver')

        self.parameters = BehaviourParameters()
        self.state_machine = StateMachine()

        # Variáveis internas
        self.last_state_machine_msg = None
        self.kick_done = False

        # Dados brutos do GameController
        self.gc_connected = False
        self.gc_state = 'PLAYING'
        self.gc_penalty = 0
        self.gc_player_found = False
        self.gc_stopped = False

        # Manda para a maquina de estadados apenas 'PLAY', 'STOP' ou 'PENALIZED'
        self.treated_gc_state = 'PLAY'

        # Publisher do estado atual
        self.state_publisher = self.create_publisher(
            CurrentStateMsg, '/transitions_and_states/state_machine', 10
        )

        # Subscribers
        self.create_subscription(
            StateMachineMsg, self.parameters.stateMachineTopic, self.state_machine_callback, 10
        )
        self.create_subscription(
            Bool, '/kick_done', self.kick_done_callback, 10
        )
        self.create_subscription(
            GameControllerMsg, '/gc/state', self.game_controller_callback, 10
        )

        self.get_logger().info('StateMachineReceiver iniciado e aguardando mensagens...')
        self.create_timer(0.1, self.update_state)  # 10 Hz

    def treat_game_controller(self) -> str:
        """Converte todos os dados do GameController em apenas 3 estados: PLAY, STOP ou PENALIZED."""
        # Sem conexão com o GC -> continua jogando (regra 4.4 b)
        if not self.gc_connected:
            return 'PLAY'

        # Jogador não encontrado na lista do time -> para por segurança
        if not self.gc_player_found:
            return 'STOP'

        # Penalizado -> congelamento total (sem andar, levantar ou mexer a cabeça)
        if self.gc_penalty != 0:
            return 'PENALIZED'

        # Jogo pausado pelo árbitro (stopped) ou qualquer estado diferente de PLAYING
        if self.gc_stopped or self.gc_state != 'PLAYING':
            return 'STOP'

        return 'PLAY'

    def game_controller_callback(self, msg: GameControllerMsg):
        self.gc_connected = msg.connected
        self.gc_state = msg.state
        self.gc_penalty = msg.penalty
        self.gc_player_found = msg.player_found
        self.gc_stopped = bool(getattr(msg, 'stopped', False))

        new_treated = self.treat_game_controller()
        if new_treated != self.treated_gc_state:
            self.get_logger().info(f'GC tratado: {self.treated_gc_state} -> {new_treated}')
            self.treated_gc_state = new_treated

        self.update_state()

    def state_machine_callback(self, msg: StateMachineMsg):
        self.last_state_machine_msg = msg
        self.update_state()

    def kick_done_callback(self, msg: Bool):
        self.kick_done = msg.data

    def update_state(self):
        if self.last_state_machine_msg is None:
            return

        stateMachineVars = self.last_state_machine_msg
        self.treated_gc_state = self.treat_game_controller()

        # Envia apenas o estado já tratado ('PLAY', 'STOP' ou 'PENALIZED')
        state_msg = self.state_machine.request_state_machine_update(
            stateMachineVars.ball_position,
            stateMachineVars.ball_close,
            stateMachineVars.ball_found,
            stateMachineVars.fall_state,
            stateMachineVars.hor_motor_out_of_center,
            stateMachineVars.head_kick_check,
            self.kick_done,
            gc_state=self.treated_gc_state,
        )

        self.state_publisher.publish(state_msg)
        self.get_logger().debug(f'Estado publicado: {state_msg.current_state}')


def main(args=None):
    rclpy.init(args=args)
    receiver = StateMachineReceiver()
    try:
        rclpy.spin(receiver)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
