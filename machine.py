from __future__ import annotations

import logging
import sys
from typing import AnyStr, BinaryIO

import isa
from isa import Address, Term


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
        self.acr, self.bur, self.inr = 0, 0, 0
        self.adr, self.dar, self.spr = 0, 0, 0
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


def simulation(code, input_tokens):
    pass


def main(src: str, input_file: str):
    logging.debug(f"Start reading from {src}")
    code = read_code(src)
    logging.debug("Read code from file")

    input_tokens = []
    logging.debug(f"Start reading input tokens {input_file}")
    with open(input_file) as f:
        input_text = f.read()
        for char in input_text:
            input_tokens.append(char)
    logging.debug(f"Successfully read {len(input_tokens)} input tokens")

    logging.debug("Simulation start")
    simulation(code, input_tokens)
    logging.debug("Simulation end")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Usage: machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
