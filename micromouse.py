import math


class MouseState:
    def __init__(self, position=(0, 0), direction=0, momentum=0, run_time_ms=0, reached_goal=False):
        self.position = position  # (x, y)
        self.direction = direction  # 0=up, 1=right, 2=down, 3=left
        self.momentum = momentum  # integer momentum value
        self.run_time_ms = run_time_ms  # accumulated run time in milliseconds
        self.reached_goal = reached_goal

    @classmethod
    def from_dict(cls, d):
        return cls(
            position=tuple(d.get('position', (0, 0))),
            direction=d.get('direction', 0),
            momentum=d.get('momentum', 0),
            run_time_ms=d.get('run_time_ms', 0),
            reached_goal=d.get('reached_goal', False)
        )

    def to_dict(self):
        return {
            'position': self.position,
            'direction': self.direction,
            'momentum': self.momentum,
            'run_time_ms': self.run_time_ms,
            'reached_goal': self.reached_goal
        }


# Momentum reduction table: maps current momentum to new momentum after reduction
# For example purposes, assume momentum reduction is as follows:
# If momentum is 0 or 1, it remains 0
# If momentum is 2 or 3, it reduces by 1
# If momentum >=4, reduces by 2
def momentum_reduction(m):
    if m <= 1:
        return 0
    elif m <= 3:
        return m - 1
    else:
        return m - 2


def apply_token(state, token):
    """
    Applies a movement token to the given MouseState.
    Supported tokens:
    - F0, F1, F2: forward with speed 0,1,2
    - V0, V1, V2: velocity changes (treat like forward with speed 0,1,2)
    - BB: brake (momentum reset)
    - L, R: in-place rotation (left or right turn)
    - Moving rotations: F1L, F0R, etc. (forward + turn)
    - Corner turns: F1LT, F2RW (forward + left turn + turn type)
    """

    # Helper functions
    def rotate_dir(d, turn):
        # turn: 'L' or 'R'
        if turn == 'L':
            return (d - 1) % 4
        elif turn == 'R':
            return (d + 1) % 4
        else:
            return d

    def move_forward(pos, direction, steps=1):
        x, y = pos
        if direction == 0:
            return (x, y + steps)
        elif direction == 1:
            return (x + steps, y)
        elif direction == 2:
            return (x, y - steps)
        elif direction == 3:
            return (x - steps, y)
        else:
            return pos

    # Parse token
    token = token.upper()

    # Momentum before move
    m = state.momentum

    # Initialize run_time increment
    run_time_inc = 0

    # Handle brake token
    if token == 'BB':
        # Brake: momentum reset to zero, small time penalty
        state.momentum = 0
        state.run_time_ms += 10
        return state

    # Handle in-place rotations (L or R only)
    if token in ['L', 'R']:
        # In-place rotation takes fixed time, momentum unchanged
        state.direction = rotate_dir(state.direction, token)
        state.run_time_ms += 5
        return state

    # Handle forward or velocity tokens without turn
    if token in ['F0', 'F1', 'F2', 'V0', 'V1', 'V2']:
        # Extract speed
        speed = int(token[1])
        # Momentum must be updated accordingly
        # Moving forward increases momentum by speed (max 5)
        new_momentum = min(m + speed, 5)
        # Movement only allowed if new_momentum <= 5 (always true here)
        # Move forward one step
        state.position = move_forward(state.position, state.direction, 1)
        state.momentum = new_momentum
        # Time cost depends on speed and momentum reduction
        run_time_inc = 10 + speed * 5
        state.run_time_ms += run_time_inc
        return state

    # Handle moving rotations like F1L, F0R
    # Format: F{speed}{turn}
    if len(token) == 3 and token[0] in ['F', 'V'] and token[2] in ['L', 'R']:
        speed = int(token[1])
        turn = token[2]
        # Check effective momentum m_eff = current momentum + speed
        m_eff = m + speed
        if m_eff > 1:
            raise ValueError(f"Moving rotation {token} not allowed: momentum {m_eff} > 1")
        # Move forward one step
        state.position = move_forward(state.position, state.direction, 1)
        # Rotate direction
        state.direction = rotate_dir(state.direction, turn)
        # Update momentum
        state.momentum = m_eff
        # Time cost: base + speed*5 + rotation 5
        run_time_inc = 10 + speed * 5 + 5
        state.run_time_ms += run_time_inc
        return state

    # Handle corner turns like F1LT, F2RW
    # Format: F{speed}{turn}{turn_type}
    # turn_type: T or W (turn type, no effect on logic here)
    if len(token) == 4 and token[0] in ['F', 'V'] and token[2] in ['L', 'R'] and token[3] in ['T', 'W']:
        speed = int(token[1])
        turn = token[2]
        # turn_type = token[3]  # Not used in logic here
        # Check effective momentum
        m_eff = m + speed
        # For corner turns, m_eff <= 1 or <= 2 allowed
        if m_eff > 2:
            raise ValueError(f"Corner turn {token} not allowed: momentum {m_eff} > 2")
        # Move forward one step
        state.position = move_forward(state.position, state.direction, 1)
        # Rotate direction
        state.direction = rotate_dir(state.direction, turn)
        # Update momentum
        state.momentum = m_eff
        # Time cost: base + speed*5 + corner turn 10
        run_time_inc = 10 + speed * 5 + 10
        state.run_time_ms += run_time_inc
        return state

    # If token not recognized
    raise ValueError(f"Unsupported token: {token}")
