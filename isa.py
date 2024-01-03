from __future__ import annotations

from enum import Enum


class Opcode(Enum):
    HALT = "halt"

    LOAD = "ld"
    STORE = "st"

    CALL = "call"
    RETURN = "ret"

    PUSH = "push"
    POP = "pop"
    POPN = "popn"

    COMPARE = "cmp"
    BRANCH_ZERO = "brz"
    BRANCH_ANY = "br"

    INCREMENT = "inc"
    MODULO = "mod"
    MULTIPLY = "mul"
    DIVIDE = "div"
    SUBTRACT = "sub"


class AddressType(Enum):
    EXACT = "#"
    ABSOLUTE = "*"
    RELATIVE_IPR = "*ipr"
    RELATIVE_SPR = "*spr"
    RELATIVE_INDIRECT_SPR = "**spr"


class Address:
    def __init__(self, kind: AddressType, val: int):
        self.kind = kind
        self.val = val

    def __str__(self):
        val_str: str = hex(self.val)
        if self.kind in (AddressType.RELATIVE_IPR, AddressType.RELATIVE_SPR, AddressType.RELATIVE_INDIRECT_SPR):
            if self.val > 0:
                val_str = "+" + val_str
            elif self.val == 0:
                val_str = ""
        return f"{self.kind.value}{val_str}"


class Term:
    def __init__(self, op: Opcode, arg: str | Address | None = None):
        self.op = op
        self.arg = arg

    def __str__(self):
        if self.arg is None:
            return f"{self.op.value}"
        return f"{self.op.value} {self.arg}"
