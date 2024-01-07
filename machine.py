from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import AnyStr, BinaryIO

import isa
import translator
from isa import Address, AddressType, Opcode, Term


class Code:
    def __init__(self, data: list[int] | None = None, instructions: list[Term] | None = None):
        self.data = [] if data is None else data
        self.instructions = [] if instructions is None else instructions


def bytes_to_int(b: AnyStr) -> int:
    val = 0
    for byte in b:
        val = (val << 8) + byte
    return val


def read_data(f: BinaryIO) -> list[int]:
    data_length = bytes_to_int(f.read(2))
    assert data_length % 8 == 0, f"unexpected data length, got {data_length}"
    data = []
    while data_length > 0:
        data.append(bytes_to_int(f.read(8)))
        data_length -= 8
    return data


def parse_instruction(instr: int) -> Term:
    assert instr < (1 << 32), f"instruction mus be less than {1 << 32}, got {instr}"
    assert instr >= 0, f"instruction must be non-negative, got {instr}"
    opcode_val = (instr >> 24) & 0xFF
    assert opcode_val in isa.opcode_by_code, f"unexpected opcode, got {opcode_val}"
    op = isa.opcode_by_code[opcode_val]
    if op in isa.no_arg_ops:
        return Term(op)
    adr_type_val = (instr >> 20) & 0xF
    assert adr_type_val in isa.address_by_code, f"unexpected address type, got {adr_type_val}"
    adr_type = isa.address_by_code[adr_type_val]
    adr_val = (instr >> 0) & 0xFFFFF
    return Term(op, Address(adr_type, adr_val))


def read_instr(f: BinaryIO) -> list[Term]:
    instr_length = bytes_to_int(f.read(2))
    assert instr_length % 4 == 0, f"unexpected instr length, got {instr_length}"
    instructions = []
    while instr_length > 0:
        instructions.append(parse_instruction(bytes_to_int(f.read(4))))
        instr_length -= 4
    return instructions


def read_code(src: str) -> Code:
    with open(src, "rb") as f:
        data_offset = bytes_to_int(f.read(2))
        instr_offset = bytes_to_int(f.read(2))
        magic = bytes_to_int(f.read(2))
        assert magic == 0xC0DE, f"Wrong magic, got {magic}"
        f.seek(data_offset)
        data = read_data(f)
        f.seek(instr_offset)
        instr = read_instr(f)
        return Code(data, instr)


class Reg(Enum):
    ACR, IPR, INR = "acr", "ipr", "inr"
    BUR, ADR, DAR, SPR = "bur", "adr", "dar", "spr"


class AluOp(Enum):
    SUM = "sum"
    MUL = "mul"


extend_20 = 0x01
plus_1 = 0x02
set_flags = 0x04
inv_left = 0x08
inv_right = 0x10
inv_out = 0x20


def extend_bits(val: int, count: int) -> int:
    bit = (val >> (count - 1)) & 0x1
    if bit == 0:
        return val & ((1 << count) - 1)
    return val | ((-1) << count)


mask_64 = (1 << 64) - 1


class DataPath:
    def __init__(self, data_memory_size: int, data_memory: list[int], input_tokens: list[str]):
        assert data_memory_size >= len(data_memory), "data_memory_size must be greater than data_memory size"
        assert all(0 <= word < (64 << 1) for word in data_memory), "all words in data memory must be uint64"
        self.data_memory_size = data_memory_size
        self.data_memory = data_memory.copy()
        self.data_memory.extend([0] * (data_memory_size - len(data_memory)))
        self.input_tokens = input_tokens
        self.output_tokens = []
        # registers
        self.acr, self.ipr, self.inr = 0, 0, 0
        self.bur, self.adr, self.dar, self.spr = 0, 0, 0, data_memory_size
        self.flr = 0

    def signal_read_data_memory(self):
        assert self.adr < self.data_memory_size, f"adr ({self.adr}) out of data_memory_size ({self.data_memory_size})"
        if self.adr == 5555:
            if len(self.input_tokens) == 0:
                raise EOFError()
            self.dar = ord(self.input_tokens.pop(0))
        else:
            self.dar = self.data_memory[self.adr]

    def signal_write_data_memory(self):
        if self.adr == 5556:
            assert 0 <= self.dar <= 0x10FFFF, f"dar contains unknown symbol, got {self.dar}"
            self.output_tokens.append(chr(self.dar))
        else:
            self.data_memory[self.adr] = self.dar

    def signal_set_inr(self, instr: Term):
        self.inr = bytes_to_int(translator.term_to_binary(instr))

    def get_ipr(self):
        return self.ipr

    def zero(self) -> bool:
        return self.flr & 0x4 != 0

    def left_alu_val(self, left: Reg | None) -> int:
        if left is None:
            return 0
        assert left in (Reg.ACR, Reg.IPR, Reg.INR), f"incorrect left register, got {left}"
        return getattr(self, left.value)

    def right_alu_val(self, right: Reg | None) -> int:
        if right is None:
            return 0
        assert right in (Reg.BUR, Reg.ADR, Reg.DAR, Reg.SPR), f"incorrect right register, got {right}"
        return getattr(self, right.value)

    def alu(self, left: int, right: int, op: AluOp, opts: int) -> int:
        output = 0
        left = ~left if opts & inv_left != 0 else left
        right = ~right if opts & inv_right != 0 else right

        left = extend_bits(left, 20) if opts & extend_20 != 0 else left

        if op == AluOp.SUM:
            output = left + right
            output = output + 1 if opts & plus_1 != 0 else output
        elif op == AluOp.MUL:
            output = left * right

        if opts & set_flags != 0:
            if output & mask_64 == 0:
                self.flr |= 0x4
            else:
                self.flr &= ~0x4

        output = ~output if opts & inv_out != 0 else output

        return output & mask_64

    def set_regs(self, alu_out: int, regs: list[Reg]):
        for reg in regs:
            assert reg in (Reg.ACR, Reg.IPR, Reg.INR, Reg.BUR, Reg.ADR, Reg.DAR, Reg.SPR), f"unsupported register {reg}"
            setattr(self, reg.value, alu_out)

    def signal_alu(
        self,
        left: Reg | None = None,
        right: Reg | None = None,
        alu_op: AluOp = AluOp.SUM,
        set_regs: list[Reg] | None = None,
        opts: int = 0,
    ):
        if set_regs is None:
            set_regs = []
        left_val = self.left_alu_val(left)
        right_val = self.right_alu_val(right)
        output = self.alu(left_val, right_val, alu_op, opts)
        self.set_regs(output, set_regs)


class ControlUnit:
    def __init__(self, program: list[Term], data_path: DataPath):
        self.program = program
        self.data_path = data_path

    def execute_call_control_instruction(self, instr: Term):
        if instr.op == Opcode.CALL:
            assert instr.arg.kind == AddressType.ABSOLUTE, f"unsupported addressing for call, got {instr}"
            self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.ADR, Reg.SPR], opts=inv_right | inv_out | plus_1)
            self.data_path.signal_alu(left=Reg.IPR, set_regs=[Reg.DAR], opts=plus_1)
            self.data_path.signal_write_data_memory()
            self.data_path.signal_alu(left=Reg.INR, set_regs=[Reg.IPR], opts=extend_20)
            return
        # RETURN
        self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.ADR])
        self.data_path.signal_read_data_memory()
        self.data_path.signal_alu(right=Reg.DAR, set_regs=[Reg.IPR])
        self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.SPR], opts=plus_1)

    def execute_branch_control_instruction(self, instr: Term):
        if instr.op == Opcode.BRANCH_ZERO:
            assert instr.arg.kind == AddressType.RELATIVE_IPR, f"unsupported addressing for brz, got {instr}"
            if self.data_path.zero():
                self.data_path.signal_alu(left=Reg.IPR, set_regs=[Reg.BUR])
                self.data_path.signal_alu(left=Reg.INR, right=Reg.BUR, set_regs=[Reg.IPR], opts=extend_20)
            else:
                self.data_path.signal_alu(left=Reg.IPR, set_regs=[Reg.IPR], opts=plus_1)
            return
        # BRANCH_ANY
        assert instr.arg.kind in (
            AddressType.ABSOLUTE,
            AddressType.RELATIVE_IPR,
        ), f"unsupported addressing for br, got {instr}"
        if instr.arg.kind == AddressType.ABSOLUTE:
            self.data_path.signal_alu(left=Reg.INR, set_regs=[Reg.IPR], opts=extend_20)
        else:
            self.data_path.signal_alu(left=Reg.IPR, set_regs=[Reg.BUR])
            self.data_path.signal_alu(left=Reg.INR, right=Reg.BUR, set_regs=[Reg.IPR], opts=extend_20)

    def execute_control_instruction(self, instr: Term) -> bool:
        opcode = instr.op
        if opcode == Opcode.HALT:
            raise StopIteration()
        if opcode in (Opcode.CALL, Opcode.RETURN):
            self.execute_call_control_instruction(instr)
            return True
        if opcode in (Opcode.BRANCH_ZERO, Opcode.BRANCH_ANY):
            self.execute_branch_control_instruction(instr)
            return True
        return False

    def address_decode(self, instr: Term):
        addr = instr.arg
        if addr.kind == AddressType.EXACT:
            pass
        elif addr.kind == AddressType.ABSOLUTE:
            self.data_path.signal_alu(left=Reg.INR, set_regs=[Reg.ADR], opts=extend_20)
        elif addr.kind == AddressType.RELATIVE_SPR:
            self.data_path.signal_alu(left=Reg.INR, right=Reg.SPR, set_regs=[Reg.ADR], opts=extend_20)
        elif addr.kind == AddressType.RELATIVE_INDIRECT_SPR:
            self.data_path.signal_alu(left=Reg.INR, right=Reg.SPR, set_regs=[Reg.ADR], opts=extend_20)
            self.data_path.signal_read_data_memory()
            self.data_path.signal_alu(right=Reg.DAR, set_regs=[Reg.ADR])
        else:
            raise NotImplementedError(f"unsupported address type for address decoding, got {instr}")

    def value_fetch(self, instr: Term):
        addr = instr.arg
        if addr.kind == AddressType.EXACT:
            self.data_path.signal_alu(left=Reg.INR, set_regs=[Reg.DAR], opts=extend_20)
        elif addr.kind in (AddressType.ABSOLUTE, AddressType.RELATIVE_SPR, AddressType.RELATIVE_INDIRECT_SPR):
            self.data_path.signal_read_data_memory()
        else:
            raise NotImplementedError(f"unsupported address type for value fetching, got {instr}")

    def execute_load_store(self, instr: Term):
        if instr.op == Opcode.LOAD:
            self.data_path.signal_alu(right=Reg.DAR, set_regs=[Reg.ACR])
            return
        # STORE
        self.data_path.signal_alu(left=Reg.ACR, set_regs=[Reg.DAR])
        self.data_path.signal_write_data_memory()

    def execute_push_pop(self, instr: Term):
        if instr.op == Opcode.PUSH:
            self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.ADR, Reg.SPR], opts=inv_right | inv_out | plus_1)
            self.data_path.signal_alu(left=Reg.ACR, set_regs=[Reg.DAR])
            self.data_path.signal_write_data_memory()
            return
        if instr.op == Opcode.POP:
            self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.ADR])
            self.data_path.signal_read_data_memory()
            self.data_path.signal_alu(right=Reg.DAR, set_regs=[Reg.ACR])
            self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.SPR], opts=plus_1)
            return
        # POPN
        self.data_path.signal_alu(right=Reg.SPR, set_regs=[Reg.SPR], opts=plus_1)

    def execute_math(self, instr: Term):
        if instr.op == Opcode.MODULO:
            return
        if instr.op == Opcode.MULTIPLY:
            return
        if instr.op == Opcode.DIVIDE:
            return
        # SUBTRACT
        self.data_path.signal_alu(left=Reg.ACR, right=Reg.DAR, set_regs=[Reg.ACR], opts=inv_right | plus_1)

    def execute(self, instr: Term):
        opcode = instr.op
        if opcode == Opcode.NOOP:
            pass
        elif opcode in (Opcode.LOAD, Opcode.STORE):
            self.execute_load_store(instr)
        elif opcode in (Opcode.PUSH, Opcode.POP, Opcode.POPN):
            self.execute_push_pop(instr)
        elif opcode == Opcode.COMPARE:
            self.data_path.signal_alu(left=Reg.ACR, right=Reg.DAR, opts=inv_right | plus_1 | set_flags)
        elif opcode == Opcode.INCREMENT:
            self.data_path.signal_read_data_memory()
            self.data_path.signal_alu(right=Reg.DAR, set_regs=[Reg.DAR], opts=plus_1)
            self.data_path.signal_write_data_memory()
        elif opcode in (Opcode.MODULO, Opcode.MULTIPLY, Opcode.DIVIDE, Opcode.SUBTRACT):
            self.execute_math(instr)
        else:
            raise NotImplementedError(f"unsupported opcode for instruction executing, got {instr}")

    def finalize(self):
        self.data_path.signal_alu(left=Reg.IPR, set_regs=[Reg.IPR], opts=plus_1)

    def execute_ordinary_instruction(self, instr: Term):
        if instr.op in isa.addr_ops or instr.op in isa.value_ops:
            self.address_decode(instr)

        if instr.op in isa.value_ops:
            self.value_fetch(instr)

        self.execute(instr)

        self.finalize()

    def execute_next_instruction(self):
        instruction = self.program[self.data_path.get_ipr()]

        self.data_path.signal_set_inr(instruction)

        if self.execute_control_instruction(instruction):
            return

        self.execute_ordinary_instruction(instruction)


def simulation(code: Code, input_tokens: list[str], data_memory_size: int = 10_000, limit: int = 1_000):
    data_path = DataPath(data_memory_size, code.data, input_tokens)
    control_unit = ControlUnit(code.instructions, data_path)
    instruction_proceed = 0

    try:
        while instruction_proceed < limit:
            control_unit.execute_next_instruction()
            instruction_proceed += 1
    except EOFError:
        logging.warning("Input buffer is empty")
    except StopIteration:
        pass

    if instruction_proceed >= limit:
        logging.warning(f"Limit {limit} exceeded")

    return "".join(data_path.output_tokens)


def main(src: str, in_file: str):
    logging.debug(f"Start reading from {src}")
    code = read_code(src)
    logging.debug("Read code from file")

    input_tokens = []
    logging.debug(f"Start reading input tokens {in_file}")
    with open(in_file) as f:
        input_text = f.read()
        for char in input_text:
            input_tokens.append(char)
    logging.debug(f"Successfully read {len(input_tokens)} input tokens")

    logging.debug("Simulation start")
    output = simulation(code, input_tokens)
    logging.debug("Simulation end")

    print(output)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Usage: machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
