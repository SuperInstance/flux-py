"""Tests for FluxVM core execution engine."""

import struct
import pytest
from flux_vm import FluxVM, OPCODES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def movi(reg, value):
    """Build MOVI instruction bytes (opcode + reg + i16)."""
    return bytes([0x2B, reg]) + struct.pack('<h', value)


def halt():
    """Build HALT instruction byte."""
    return bytes([0x80])


def iadd(d, a, b):
    """Build IADD instruction bytes."""
    return bytes([0x08, d, a, b])


def imul(d, a, b):
    """Build IMUL instruction bytes."""
    return bytes([0x0A, d, a, b])


def idiv(d, a, b):
    """Build IDIV instruction bytes."""
    return bytes([0x0B, d, a, b])


def dec(reg):
    """Build DEC instruction bytes."""
    return bytes([0x0F, reg])


def jnz(reg, offset):
    """Build JNZ instruction bytes (opcode + reg + i16 offset)."""
    return bytes([0x06, reg]) + struct.pack('<h', offset)


# ─── Basic Execution ─────────────────────────────────────────────────────────

class TestBasicExecution:

    def test_mov_immediate_and_halt(self):
        """MOVI loads a value into a register; HALT stops execution."""
        bc = movi(0, 42) + halt()
        vm = FluxVM(bc).execute()
        assert vm.halted
        assert vm.error is None
        assert vm.result(0) == 42

    def test_multiple_mov_immediates(self):
        """Multiple MOVI instructions load different registers."""
        bc = movi(0, 10) + movi(1, 20) + movi(2, 30) + halt()
        vm = FluxVM(bc).execute()
        assert vm.reg(0) == 10
        assert vm.reg(1) == 20
        assert vm.reg(2) == 30

    def test_empty_bytecode(self):
        """Empty bytecode: VM should stop immediately (no instructions)."""
        vm = FluxVM(b'').execute()
        assert not vm.halted
        assert vm.error is None
        assert vm.cycles == 0

    def test_only_halt(self):
        """A single HALT instruction should work correctly."""
        bc = halt()
        vm = FluxVM(bc).execute()
        assert vm.halted
        assert vm.cycles == 1

    def test_max_cycles_limit(self):
        """VM should stop after max_cycles even if not halted."""
        # MOVI R0, 1 — no HALT, so it would loop forever
        bc = movi(0, 1)
        vm = FluxVM(bc, max_cycles=5).execute()
        assert not vm.halted
        assert vm.cycles == 1  # only 1 instruction in bytecode


# ─── Arithmetic ───────────────────────────────────────────────────────────────

class TestArithmetic:

    def test_iadd(self):
        """IADD: R0 = R1 + R2."""
        bc = movi(1, 7) + movi(2, 5) + iadd(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 12

    def test_iadd_negative_values(self):
        """IADD with negative immediate values."""
        bc = movi(1, -10) + movi(2, 3) + iadd(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == -7

    def test_imul(self):
        """IMUL: R0 = R1 * R2."""
        bc = movi(1, 6) + movi(2, 7) + imul(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 42

    def test_imul_zero(self):
        """IMUL by zero."""
        bc = movi(1, 100) + movi(2, 0) + imul(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 0

    def test_idiv(self):
        """IDIV: R0 = R1 / R2 (integer division)."""
        bc = movi(1, 20) + movi(2, 4) + idiv(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 5

    def test_idiv_truncates(self):
        """IDIV truncates toward zero (integer division)."""
        bc = movi(1, 7) + movi(2, 3) + idiv(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 2

    def test_idiv_by_zero(self):
        """Division by zero should set error and not halt."""
        bc = movi(1, 10) + movi(2, 0) + idiv(0, 1, 2) + halt()
        vm = FluxVM(bc).execute()
        assert not vm.halted
        assert vm.error == "Division by zero"
        assert vm.result(0) is None

    def test_dec(self):
        """DEC decrements a register."""
        bc = movi(0, 5) + dec(0) + dec(0) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 3

    def test_dec_negative(self):
        """DEC past zero goes negative."""
        bc = movi(0, 1) + dec(0) + dec(0) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == -1


# ─── Control Flow ─────────────────────────────────────────────────────────────

class TestControlFlow:

    def test_jnz_taken(self):
        """JNZ jumps when register is non-zero."""
        # MOVI R0, 1; JNZ R0, 4; MOVI R0, 99; HALT
        # After reading JNZ (4 bytes), PC=8. Offset 4 → PC=12=HALT, skips MOVI R0,99
        bc = movi(0, 1) + jnz(0, 4) + movi(0, 99) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 1  # JNZ was taken, R0 stays 1

    def test_jnz_not_taken(self):
        """JNZ does NOT jump when register is zero."""
        # MOVI R0, 0; JNZ R0, 4; MOVI R0, 42; HALT
        # JNZ not taken, so MOVI R0,42 executes
        bc = movi(0, 0) + jnz(0, 4) + movi(0, 42) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 42

    def test_loop_with_dec_and_jnz(self):
        """Loop: sum 3+2+1 = 6 using DEC + JNZ."""
        # MOVI R0, 3; MOVI R1, 0; IADD R1, R1, R0; DEC R0; JNZ R0, -10; HALT
        # Loop body = IADD(4) + DEC(2) + JNZ(4) = 10 bytes
        # After reading JNZ, PC points to HALT. Offset -10 goes back to IADD.
        bc = (movi(0, 3) + movi(1, 0) + iadd(1, 1, 0) + dec(0)
              + jnz(0, -10) + halt())
        vm = FluxVM(bc).execute()
        assert vm.result(1) == 6

    def test_factorial_loop(self):
        """Factorial of 5 = 120 using DEC + JNZ + IMUL."""
        # MOVI R0, 5; MOVI R1, 1;
        # loop: IMUL R1, R1, R0; DEC R0; JNZ R0, -10; HALT
        # Loop body = IMUL(4) + DEC(2) + JNZ(4) = 10 bytes
        bc = (movi(0, 5) + movi(1, 1) + imul(1, 1, 0)
              + dec(0) + jnz(0, -10) + halt())
        vm = FluxVM(bc).execute()
        assert vm.result(1) == 120

    def test_backwards_jnz_loop_10(self):
        """Sum 1..10 = 55."""
        # MOVI R0, 10; MOVI R1, 0;
        # loop: IADD R1, R1, R0; DEC R0; JNZ R0, -10; HALT
        # Loop body = IADD(4) + DEC(2) + JNZ(4) = 10 bytes
        bc = (movi(0, 10) + movi(1, 0) + iadd(1, 1, 0)
              + dec(0) + jnz(0, -10) + halt())
        vm = FluxVM(bc).execute()
        assert vm.result(1) == 55


# ─── Error Handling ───────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_unknown_opcode(self):
        """Unknown opcode sets error."""
        bc = bytes([0x00])  # not a valid opcode
        vm = FluxVM(bc).execute()
        assert not vm.halted
        assert vm.error is not None
        assert "Unknown opcode" in vm.error

    def test_truncated_bytecode_movi(self):
        """MOVI without full operands should raise IndexError on read."""
        bc = bytes([0x2B])  # MOVI but no register or value
        vm = FluxVM(bc).execute()
        assert vm.error is not None

    def test_truncated_bytecode_iadd(self):
        """IADD without full operands should raise error."""
        bc = bytes([0x08, 0, 0])  # IADD but only 2 operands
        vm = FluxVM(bc).execute()
        assert vm.error is not None

    def test_result_on_error_returns_none(self):
        """result() should return None when there's an error."""
        bc = movi(1, 0) + idiv(0, 1, 1) + halt()  # R1=0, div by zero
        vm = FluxVM(bc).execute()
        assert vm.result(0) is None

    def test_result_on_not_halted_returns_none(self):
        """result() should return None when VM hasn't halted."""
        bc = movi(0, 42)  # no halt
        vm = FluxVM(bc).execute()
        assert vm.result(0) is None


# ─── Register Access ──────────────────────────────────────────────────────────

class TestRegisterAccess:

    def test_all_16_registers_default_zero(self):
        """All 16 registers should start at 0."""
        vm = FluxVM(halt()).execute()
        for i in range(16):
            assert vm.reg(i) == 0

    def test_register_out_of_bounds(self):
        """Register index >= 16 should return 0."""
        vm = FluxVM(halt()).execute()
        assert vm.reg(16) == 0
        assert vm.reg(100) == 0
        assert vm.reg(-1) == 0

    def test_result_returns_correct_register(self):
        """result(reg) returns the specified register."""
        bc = movi(5, 99) + halt()
        vm = FluxVM(bc).execute()
        assert vm.result(5) == 99
        assert vm.result(0) == 0


# ─── Opcodes Constant ────────────────────────────────────────────────────────

class TestOpcodesConstant:

    def test_opcode_values(self):
        """Verify opcode constant values."""
        assert OPCODES['MOVI'] == 0x2B
        assert OPCODES['IADD'] == 0x08
        assert OPCODES['IMUL'] == 0x0A
        assert OPCODES['IDIV'] == 0x0B
        assert OPCODES['DEC'] == 0x0F
        assert OPCODES['JNZ'] == 0x06
        assert OPCODES['HALT'] == 0x80
