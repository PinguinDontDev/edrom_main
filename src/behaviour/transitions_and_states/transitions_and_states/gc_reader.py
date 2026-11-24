#!/usr/bin/env python3
# coding=utf-8
"""Receive GameController UDP broadcasts, send unicast return packets, and publish /gc/state."""

import socket
import struct
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from modularized_bhv_msgs.msg import (
    GameControllerMsg,
    GameControllerPlayer,
    GameControllerTeam,
)

from .robocup_game_control_data import (
    GAMECONTROLLER_DATA_PORT,
    GAMECONTROLLER_RETURN_PORT,
    RoboCupGameControlData,
)

# Cabeçalho oficial de retorno da Humanoid League (RoboCupGameControlData.h)
GAMECONTROLLER_RETURN_STRUCT_HEADER = b'RGrt'
GAMECONTROLLER_RETURN_STRUCT_VERSION = 2
GAMECONTROLLER_RETURN_MSG_ALIVE = 2


def to_message(parsed, team_num, player_num, has_localization=False):
    """Converte RoboCupGameControlData para GameControllerMsg sem serialização JSON."""
    msg = GameControllerMsg(team_number=team_num, player_number=player_num)
    if parsed is None:
        msg.connected = False
        return msg

    msg.connected = True
    msg.packet = int(parsed.packetNumber)
    msg.players_per_team = int(parsed.playersPerTeam)
    msg.competition = int(parsed.competitionType)
    msg.stopped = bool(parsed.stopped)
    msg.phase = parsed.phase_name
    msg.set_play = parsed.set_play_name
    msg.first_half = bool(parsed.firstHalf)
    msg.secs_remaining = int(parsed.secsRemaining)
    msg.secondary_time = int(parsed.secondaryTime)
    msg.kicking_team = -1 if parsed.kickingTeam == 255 else int(parsed.kickingTeam)

    for team in parsed.teams:
        players = [
            GameControllerPlayer(
                number=int(p.number),
                penalty=int(p.penalty),
                penalty_name=p.penalty_name,
                secs_till_unpenalised=int(p.secsTillUnpenalised),
                cautions=int(p.cautions),
            )
            for p in team.players
        ]
        msg.teams.append(
            GameControllerTeam(
                team_number=int(team.teamNumber),
                field_colour=team.field_colour_name,
                goalkeeper_colour=team.goalkeeper_colour_name,
                goalkeeper=int(team.goalkeeper),
                score=int(team.score),
                penalty_shot=int(team.penaltyShot),
                single_shots=int(team.singleShots),
                message_budget=int(team.messageBudget),
                players=players,
            )
        )

    own = next((t for t in parsed.teams if t.teamNumber == team_num), None)
    msg.team_found = own is not None
    msg.player_found = own is not None and 1 <= player_num <= len(own.players)

    has_kickoff = False
    if own is not None:
        opponent = next((t for t in parsed.teams if t is not own), None)
        has_kickoff = (msg.kicking_team == team_num)
        msg.score = int(own.score)
        msg.opponent_score = int(opponent.score) if opponent is not None else 0
        msg.has_kickoff = has_kickoff
        msg.is_goalkeeper = (own.goalkeeper == player_num)
        msg.message_budget = int(own.messageBudget)

        if msg.player_found:
            me = own.players[player_num - 1]
            msg.penalty = int(me.penalty)
            msg.penalty_name = me.penalty_name
            msg.secs_till_unpenalised = int(me.secsTillUnpenalised)
            msg.cautions = int(me.cautions)

    # Sem localização: segura em SET durante 'stopped' ou nos 10s do kick-off adversário
    effective_state = parsed.state_name
    kickoff_wait = (
        not has_localization
        and effective_state == 'PLAYING'
        and msg.kicking_team != -1
        and not has_kickoff
        and parsed.secondaryTime > 0
    )
    if (parsed.stopped and effective_state in ('READY', 'PLAYING')) or kickoff_wait:
        effective_state = 'SET'

    msg.state = effective_state
    return msg


class GcReader(Node):
    def __init__(self, **kwargs):
        super().__init__('gc_reader', **kwargs)
        self.team_num = int(self.declare_parameter('team_num', 0).value)
        self.player_num = int(self.declare_parameter('player_num', 1).value)
        port = int(self.declare_parameter('port', GAMECONTROLLER_DATA_PORT).value)
        self.return_port = int(
            self.declare_parameter('return_port', GAMECONTROLLER_RETURN_PORT).value
        )
        self.send_return = bool(self.declare_parameter('send_return_packet', True).value)
        self.has_localization = bool(self.declare_parameter('has_localization', False).value)

        if not 0 <= self.team_num <= 254 or not 1 <= self.player_num <= 20:
            raise ValueError('team_num deve estar entre 0 e 254; player_num entre 1 e 20.')
        if not 0 <= port <= 65535 or not 0 <= self.return_port <= 65535:
            raise ValueError('Porta UDP inválida.')

        self.pub = self.create_publisher(GameControllerMsg, '/gc/state', 10)
        self.last_rx = None

        # Socket de recepção (broadcast UDP 3838)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Socket separado para retorno UNICAST ao IP do PC do GameController (porta 3939)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rx.bind(('', port))
            self.rx.setblocking(False)
        except OSError:
            self.rx.close()
            self.tx.close()
            raise

        # Wall time é obrigatório mesmo quando o /clock da simulação está pausado
        clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.02, self.receive, clock=clock)
        self.create_timer(0.5, self.check_connection, clock=clock)

        self.get_logger().info(
            f'GameController UDP {self.rx.getsockname()[1]} -> /gc/state '
            f'(retorno unicast porta {self.return_port}); '
            f'time={self.team_num}, jogador={self.player_num}, '
            f'localizacao={self.has_localization}'
        )

    def _send_alive_packet(self, gc_ip: str):
        """Envia pacote RGrt em unicast para o PC do GameController (não requer pose)."""
        if not self.send_return or not gc_ip:
            return
        packet = struct.pack(
            '<4sBBBB',
            GAMECONTROLLER_RETURN_STRUCT_HEADER,
            GAMECONTROLLER_RETURN_STRUCT_VERSION,
            self.team_num & 0xFF,
            self.player_num & 0xFF,
            GAMECONTROLLER_RETURN_MSG_ALIVE,
        )
        try:
            self.tx.sendto(packet, (gc_ip, self.return_port))
        except OSError as exc:
            self.get_logger().warn(
                f'Falha enviando retorno unicast ao GC ({gc_ip}:{self.return_port}): {exc}',
                throttle_duration_sec=2.0,
            )

    def receive(self):
        # Limita pacotes por callback para tráfego malformado não travar o executor
        for _ in range(64):
            try:
                data, addr = self.rx.recvfrom(4096)
            except BlockingIOError:
                return

            try:
                parsed = RoboCupGameControlData.parse(data)
            except ValueError as exc:
                self.get_logger().warn(
                    f'Datagrama GC ignorado: {exc}', throttle_duration_sec=2.0
                )
                continue

            self.last_rx = time.monotonic()
            self._send_alive_packet(addr[0])
            self.pub.publish(
                to_message(parsed, self.team_num, self.player_num, self.has_localization)
            )

    def check_connection(self):
        if self.last_rx is None or time.monotonic() - self.last_rx >= 3.0:
            self.pub.publish(to_message(None, self.team_num, self.player_num))

    def destroy_node(self):
        self.rx.close()
        self.tx.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GcReader()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
