"""ViZDoom Deathmatch controls and a visible-enemy teacher."""

from pathlib import Path

import vizdoom as vzd

ACTIONS = ("advance", "turn_right", "aim_left", "aim_right", "advance_fire", "fire")
# Available buttons are explicitly reduced to ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT.
BUTTONS = {
    "advance": [0, 1, 0, 0],
    "turn_right": [0, 0, 0, 1],
    "aim_left": [1, 0, 1, 0],
    "aim_right": [1, 0, 0, 1],
    "advance_fire": [1, 1, 0, 0],
    "fire": [1, 0, 0, 0],
}
DECISION_TICS = 5
ENEMIES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon",
           "Spectre", "LostSoul", "Cacodemon", "HellKnight", "BaronOfHell",
           "Arachnotron", "Revenant", "Mancubus", "PainElemental"}


def open_game(seed: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "deathmatch.cfg"))
    game.set_doom_skill(1)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_available_buttons([vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD,
                                vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT])
    game.set_available_game_variables([vzd.GameVariable.HEALTH,
                                       vzd.GameVariable.SELECTED_WEAPON_AMMO,
                                       vzd.GameVariable.KILLCOUNT])
    game.set_seed(seed)
    game.init()
    game.new_episode()
    return game


def target(state):
    visible = [label for label in state.labels if label.object_name in ENEMIES]
    return max(visible, key=lambda label: label.width * label.height, default=None)


def wall_near(state) -> bool:
    depth = state.depth_buffer
    if depth is None:
        return False
    return int(depth[depth.shape[0] // 2, depth.shape[1] // 2]) <= 4


def describe(state) -> str:
    enemy = target(state)
    health, ammo, _ = map(int, state.game_variables)
    status = (f"[health {'low' if health < 40 else 'ok'}] "
              f"[ammo {'empty' if ammo == 0 else 'ready'}] "
              f"[wall {'near' if wall_near(state) else 'far'}]")
    if enemy is None:
        return f"[enemy hidden] {status}"
    center = enemy.x + enemy.width / 2
    offset = center - state.screen_buffer.shape[1] / 2
    aim = "left" if offset < -18 else "right" if offset > 18 else "center"
    size = "near" if enemy.width > 65 else "mid" if enemy.width > 30 else "far"
    return f"[enemy visible] [aim {aim}] [size {size}] {status}"


def teacher(state) -> str:
    enemy = target(state)
    if enemy is None or state.game_variables[1] <= 0:
        return "turn_right" if wall_near(state) else "advance"
    center = enemy.x + enemy.width / 2
    offset = center - state.screen_buffer.shape[1] / 2
    if offset < -18:
        return "aim_left"
    if offset > 18:
        return "aim_right"
    return "fire" if enemy.width > 65 else "advance_fire"
