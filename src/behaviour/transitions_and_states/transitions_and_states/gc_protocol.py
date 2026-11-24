    #!/usr/bin/env python3
# coding=utf-8
"""Decode GameController v20 (RGme, 158 bytes) Espero que nao seja eu a consertar caso de erro"""

from dataclasses import dataclass
import struct
from typing import List

GAMECONTROLLER_DATA_PORT = 3838
GAMECONTROLLER_RETURN_PORT = 3939

GAMECONTROLLER_STRUCT_HEADER = b'RGme'
GAMECONTROLLER_STRUCT_VERSION = 20

HEADER = struct.Struct('<4sBBBBBBBBBBhh')
TEAM = struct.Struct('<BBBBBBHH' + 'BBB' * 20)
EXPECTED_PACKET_SIZE = HEADER.size + 2 * TEAM.size  # 158 bytes

STATES = ['INITIAL', 'READY', 'SET', 'PLAYING', 'FINISHED']
PHASES = ['NORMAL', 'PENALTY_SHOOT_OUT', 'EXTRA_TIME', 'TIMEOUT']
SET_PLAYS = [
    'NONE', 'DIRECT_FREE_KICK', 'INDIRECT_FREE_KICK',
    'PENALTY_KICK', 'THROW_IN', 'GOAL_KICK', 'CORNER_KICK'
]
PENALTIES = [
    'NONE', 'ILLEGAL_POSITIONING', 'MOTION_IN_SET', 'MOTION_IN_STOP',
    'LOCAL_GAME_STUCK', 'INCAPABLE_ROBOT', 'PICK_UP', 'BALL_HOLDING',
    'LEAVING_THE_FIELD', 'PLAYING_WITH_ARMS_HANDS', 'PUSHING',
    'CAUTIONED', 'SENT_OFF', 'SUBSTITUTE'
]
COLOURS = [
    'BLUE', 'RED', 'YELLOW', 'BLACK', 'WHITE', 'GREEN',
    'ORANGE', 'PURPLE', 'BROWN', 'GRAY'
]


def name(values: List[str], index: int) -> str:
    return values[index] if 0 <= index < len(values) else str(index)


@dataclass
class RobotInfo:
    number: int
    penalty: int
    secsTillUnpenalised: int
    cautions: int

    @property
    def penalty_name(self) -> str:
        return name(PENALTIES, self.penalty)


@dataclass
class TeamInfo:
    teamNumber: int
    fieldPlayerColour: int
    goalkeeperColour: int
    goalkeeper: int
    score: int
    penaltyShot: int
    singleShots: int
    messageBudget: int
    players: List[RobotInfo]

    @property
    def field_colour_name(self) -> str:
        return name(COLOURS, self.fieldPlayerColour)

    @property
    def goalkeeper_colour_name(self) -> str:
        return name(COLOURS, self.goalkeeperColour)


@dataclass
class RoboCupGameControlData:
    version: int
    packetNumber: int
    playersPerTeam: int
    competitionType: int
    stopped: bool
    gamePhase: int
    state: int
    setPlay: int
    firstHalf: int
    kickingTeam: int
    secsRemaining: int
    secondaryTime: int
    teams: List[TeamInfo]

    @property
    def state_name(self) -> str:
        return name(STATES, self.state)

    @property
    def phase_name(self) -> str:
        return name(PHASES, self.gamePhase)

    @property
    def set_play_name(self) -> str:
        return name(SET_PLAYS, self.setPlay)

    @classmethod
    def parse(cls, data: bytes) -> 'RoboCupGameControlData':
        """Decodifica um datagrama RGme v20 (158 bytes) ou lança ValueError."""
        if len(data) != EXPECTED_PACKET_SIZE:
            raise ValueError(
                f'Tamanho de pacote inválido: {len(data)} bytes (esperado {EXPECTED_PACKET_SIZE}).'
            )
        if data[:4] != GAMECONTROLLER_STRUCT_HEADER:
            raise ValueError(f'Cabeçalho inválido: {data[:4]!r} (esperado {GAMECONTROLLER_STRUCT_HEADER!r}).')

        (
            _, version, packet, per_team, comp, stopped, phase, state,
            set_play, first_half, kicking, secs, secondary
        ) = HEADER.unpack_from(data)

        if version != GAMECONTROLLER_STRUCT_VERSION:
            raise ValueError(
                f'Versão do GameController incompatível: v{version} '
                f'(esperado v{GAMECONTROLLER_STRUCT_VERSION}).'
            )
        if not 1 <= per_team <= 20:
            raise ValueError(f'Número de jogadores por time inválido: {per_team}.')

        teams: List[TeamInfo] = []
        for i in range(2):
            v = TEAM.unpack_from(data, HEADER.size + TEAM.size * i)
            players = [
                RobotInfo(
                    number=p + 1,
                    penalty=v[8 + 3 * p],
                    secsTillUnpenalised=v[9 + 3 * p],
                    cautions=v[10 + 3 * p],
                )
                for p in range(per_team)
            ]
            teams.append(
                TeamInfo(
                    teamNumber=v[0],
                    fieldPlayerColour=v[1],
                    goalkeeperColour=v[2],
                    goalkeeper=v[3],
                    score=v[4],
                    penaltyShot=v[5],
                    singleShots=v[6],
                    messageBudget=v[7],
                    players=players,
                )
            )

        return cls(
            version=version,
            packetNumber=packet,
            playersPerTeam=per_team,
            competitionType=comp,
            stopped=bool(stopped),
            gamePhase=phase,
            state=state,
            setPlay=set_play,
            firstHalf=first_half,
            kickingTeam=kicking,  # 255 quando nenhum time tem o kick-off
            secsRemaining=secs,
            secondaryTime=secondary,
            teams=teams,
        )
