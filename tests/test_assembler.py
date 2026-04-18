"""Tests for the Assembler and Disassembler."""

import struct
import pytest
from flux_vm import Assembler, Disassembler, OPCODES


# ─── Assembler Tests ─────────────────────────────────────────────────────────

class TestAssemblerBasic:

    def setup_method(self):
        self.asm = Assembler()

    def test_assemble_halt(self):
        """HALT assembles to single byte 0x80."""
        bc = self.asm.assemble("HALT")
        assert bc == bytes([0x80])

    def test_assemble_movi(self):
        """MOVI R0, 42 assembles correctly."""
        bc = self.asm.assemble("MOVI R0, 42")
        assert bc[0] == 0x2B
        assert bc[1] == 0  # R0
        val = struct.unpack('<h', bc[2:4])[0]
        assert val == 42

    def test_assemble_negative_immediate(self):
        """MOVI with negative immediate value."""
        bc = self.asm.assemble("MOVI R0, -10")
        val = struct.unpack('<h', bc[2:4])[0]
        assert val == -10

    def test_assemble_iadd(self):
        """IADD R0, R1, R2 assembles correctly."""
        bc = self.asm.assemble("IADD R0, R1, R2")
        assert bc == bytes([0x08, 0, 1, 2])

    def test_assemble_imul(self):
        """IMUL R0, R1, R2 assembles correctly."""
        bc = self.asm.assemble("IMUL R0, R1, R2")
        assert bc == bytes([0x0A, 0, 1, 2])

    def test_assemble_idiv(self):
        """IDIV R0, R1, R2 assembles correctly."""
        bc = self.asm.assemble("IDIV R0, R1, R2")
        assert bc == bytes([0x0B, 0, 1, 2])

    def test_assemble_dec(self):
        """DEC R0 assembles correctly."""
        bc = self.asm.assemble("DEC R0")
        assert bc == bytes([0x0F, 0])

    def test_assemble_jnz_with_offset(self):
        """JNZ R0, offset assembles correctly."""
        bc = self.asm.assemble("JNZ R0, 10")
        assert bc[0] == 0x06
        assert bc[1] == 0  # R0
        val = struct.unpack('<h', bc[2:4])[0]
        assert val == 10


class TestAssemblerLabels:

    def setup_method(self):
        self.asm = Assembler()

    def test_label_forward_jump(self):
        """Forward label resolution for JNZ."""
        text = "MOVI R0, 1\nJNZ R0, end\nMOVI R0, 99\nend: HALT"
        bc = self.asm.assemble(text)
        # Verify the JNZ offset points past MOVI R0,99 (4 bytes) to HALT
        # JNZ is at byte 5 (after MOVI R0,1 which is 4 bytes)
        # After reading JNZ (4 bytes), PC = 9
        # HALT is at byte 9 (after MOVI R0,99 which is 4 more bytes)
        # Wait: MOVI R0,1 = 4 bytes, JNZ = 4 bytes, MOVI R0,99 = 4 bytes, HALT = 1 byte
        # JNZ at offset 4, after reading JNZ PC=8, HALT at offset 12
        # So offset should be 12-8 = 4
        # Let's verify by disassembly
        dis = Disassembler()
        disasm = dis.disassemble(bc)
        # The key check: running this should give R0=1 (jump taken, skips MOVI R0,99)
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 1

    def test_label_backward_jump(self):
        """Backward label resolution for loop."""
        text = "MOVI R0, 3\nloop: DEC R0\nJNZ R0, loop\nHALT"
        bc = self.asm.assemble(text)
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.halted
        assert vm.reg(0) == 0  # decremented to 0

    def test_label_on_own_line(self):
        """Label can be on its own line."""
        text = "MOVI R0, 1\nstart:\nHALT"
        bc = self.asm.assemble(text)
        assert len(bc) == 5  # MOVI(4) + HALT(1)

    def test_label_not_found(self):
        """Unresolvable label raises ValueError."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            self.asm.assemble("JNZ R0, nonexistent")


class TestAssemblerComments:

    def setup_method(self):
        self.asm = Assembler()

    def test_hash_comment(self):
        """Lines starting with # are ignored."""
        bc = self.asm.assemble("# this is a comment\nHALT")
        assert bc == bytes([0x80])

    def test_semicolon_comment(self):
        """Lines starting with ; are ignored."""
        bc = self.asm.assemble("; this is a comment\nHALT")
        assert bc == bytes([0x80])

    def test_comment_between_instructions(self):
        """Comments between instructions are ignored."""
        bc = self.asm.assemble("MOVI R0, 5\n# comment\nHALT")
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 5


class TestAssemblerErrors:

    def setup_method(self):
        self.asm = Assembler()

    def test_unknown_mnemonic(self):
        """Unknown instruction mnemonic raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mnemonic"):
            self.asm.assemble("FOOBAR R0, 1")


class TestAssemblerMultiInstruction:

    def setup_method(self):
        self.asm = Assembler()

    def test_multi_instruction_program(self):
        """Complex program: (3 + 4) * 2 = 14."""
        text = "MOVI R0, 3\nMOVI R1, 4\nIADD R0, R0, R1\nMOVI R1, 2\nIMUL R0, R0, R1\nHALT"
        bc = self.asm.assemble(text)
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 14

    def test_factorial_program(self):
        """Factorial of 5 = 120."""
        text = """
            MOVI R0, 5
            MOVI R1, 1
            loop: IMUL R1, R1, R0
            DEC R0
            JNZ R0, loop
            HALT
        """
        bc = self.asm.assemble(text)
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(1) == 120


# ─── Assembler → VM Round-Trip ───────────────────────────────────────────────

class TestAssemblerVMRoundTrip:

    def setup_method(self):
        self.asm = Assembler()

    def test_round_trip_addition(self):
        """Assemble → execute: 10 + 20 = 30."""
        bc = self.asm.assemble("MOVI R0, 10\nMOVI R1, 20\nIADD R0, R0, R1\nHALT")
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 30

    def test_round_trip_division(self):
        """Assemble → execute: 100 / 7 = 14."""
        bc = self.asm.assemble("MOVI R0, 100\nMOVI R1, 7\nIDIV R0, R0, R1\nHALT")
        from flux_vm import FluxVM
        vm = FluxVM(bc).execute()
        assert vm.result(0) == 14


# ─── Disassembler Tests ──────────────────────────────────────────────────────

class TestDisassembler:

    def setup_method(self):
        self.dis = Disassembler()

    def test_disassemble_halt(self):
        """HALT disassembles correctly."""
        bc = bytes([0x80])
        text = self.dis.disassemble(bc)
        assert "HALT" in text

    def test_disassemble_movi(self):
        """MOVI R0, 42 disassembles correctly."""
        bc = bytes([0x2B, 0]) + struct.pack('<h', 42)
        text = self.dis.disassemble(bc)
        assert "MOVI R0, 42" in text

    def test_disassemble_iadd(self):
        """IADD R0, R1, R2 disassembles correctly."""
        bc = bytes([0x08, 0, 1, 2])
        text = self.dis.disassemble(bc)
        assert "IADD R0, R1, R2" in text

    def test_disassemble_imul(self):
        """IMUL R3, R4, R5 disassembles correctly."""
        bc = bytes([0x0A, 3, 4, 5])
        text = self.dis.disassemble(bc)
        assert "IMUL R3, R4, R5" in text

    def test_disassemble_idiv(self):
        """IDIV R0, R1, R2 disassembles correctly."""
        bc = bytes([0x0B, 0, 1, 2])
        text = self.dis.disassemble(bc)
        assert "IDIV R0, R1, R2" in text

    def test_disassemble_dec(self):
        """DEC R0 disassembles correctly."""
        bc = bytes([0x0F, 0])
        text = self.dis.disassemble(bc)
        assert "DEC R0" in text

    def test_disassemble_jnz(self):
        """JNZ R0, offset disassembles correctly."""
        bc = bytes([0x06, 0]) + struct.pack('<h', -6)
        text = self.dis.disassemble(bc)
        assert "JNZ R0, -6" in text

    def test_disassemble_unknown_opcode(self):
        """Unknown opcode shows hex value."""
        bc = bytes([0xFF])
        text = self.dis.disassemble(bc)
        assert "0xFF" in text

    def test_disassemble_has_address_prefixes(self):
        """Disassembly lines start with hex addresses."""
        bc = bytes([0x80, 0x80])
        text = self.dis.disassemble(bc)
        assert "0000:" in text
        assert "0001:" in text

    def test_disassemble_empty_bytecode(self):
        """Empty bytecode gives empty string."""
        text = self.dis.disassemble(b'')
        assert text == ''


# ─── Assemble → Disassemble Round-Trip ───────────────────────────────────────

class TestAssembleDisassembleRoundTrip:

    def test_round_trip_simple(self):
        """Assemble → disassemble → verify expected mnemonics."""
        asm = Assembler()
        dis = Disassembler()
        bc = asm.assemble("MOVI R0, 42\nHALT")
        text = dis.disassemble(bc)
        assert "MOVI R0, 42" in text
        assert "HALT" in text

    def test_round_trip_arithmetic(self):
        """Assemble → disassemble arithmetic program."""
        asm = Assembler()
        dis = Disassembler()
        bc = asm.assemble("MOVI R0, 10\nMOVI R1, 20\nIADD R2, R0, R1\nHALT")
        text = dis.disassemble(bc)
        assert "MOVI R0, 10" in text
        assert "MOVI R1, 20" in text
        assert "IADD R2, R0, R1" in text
        assert "HALT" in text
