"""
FLUX Minimal Python VM — bytecode interpreter with natural language interface.

Features:
  - 16 general-purpose registers (R0-R15)
  - Opcodes: MOVI=0x2B, MOV=0x2C, IADD=0x08, ISUB=0x09, IMUL=0x0A, IDIV=0x0B,
             INC=0x0E, DEC=0x0F, CMP=0x0D, JZ=0x07, JNZ=0x06, JMP=0x05,
             PUSH=0x01, POP=0x02, HALT=0x80
  - Assembler class: text → bytecode with label support
  - Disassembler class: bytecode → text
  - Vocabulary class: natural language patterns → assembly
  - Interpreter: natural language → result
"""

import re
import struct
from typing import Optional, Dict


# ─── Opcodes ────────────────────────────────────────────────────────────────

OPCODES = {
    'MOVI': 0x2B, 'MOV':  0x2C,
    'IADD': 0x08, 'ISUB': 0x09, 'IMUL': 0x0A, 'IDIV': 0x0B,
    'INC':  0x0E, 'DEC':  0x0F, 'CMP':  0x0D,
    'JZ':   0x07, 'JNZ':  0x06, 'JMP':  0x05,
    'PUSH': 0x01, 'POP':  0x02,
    'HALT': 0x80,
}


# ─── VM ─────────────────────────────────────────────────────────────────────

class FluxVM:
    """FLUX bytecode virtual machine with 16 registers and a stack."""

    MAX_CYCLES = 10_000_000

    def __init__(self, bytecode: bytes, max_cycles: int = None):
        self.bc = bytecode
        self.gp = [0] * 16
        self.pc = 0
        self.halted = False
        self.cycles = 0
        self.max_cycles = max_cycles or self.MAX_CYCLES
        self.error = None
        self.stack = []

    def reg(self, idx: int) -> int:
        """Read a register."""
        return self.gp[idx] if 0 <= idx < 16 else 0

    def _u8(self) -> int:
        v = self.bc[self.pc]
        self.pc += 1
        return v

    def _i16(self) -> int:
        lo = self.bc[self.pc]
        hi = self.bc[self.pc + 1]
        self.pc += 2
        val = lo | (hi << 8)
        return val - 65536 if val >= 32768 else val

    def execute(self) -> 'FluxVM':
        """Run until HALT or max_cycles. Returns self for chaining."""
        self.halted = False
        self.cycles = 0
        self.error = None
        try:
            while not self.halted and self.pc < len(self.bc) and self.cycles < self.max_cycles:
                op = self._u8()
                self.cycles += 1

                if   op == 0x80: self.halted = True
                elif op == 0x2B: d = self._u8(); self.gp[d] = self._i16()
                elif op == 0x2C: d = self._u8(); a = self._u8(); self.gp[d] = self.gp[a]
                elif op == 0x08: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] + self.gp[b]
                elif op == 0x09: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] - self.gp[b]
                elif op == 0x0A: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] * self.gp[b]
                elif op == 0x0B:
                    d, a, b = self._u8(), self._u8(), self._u8()
                    if self.gp[b] == 0: raise RuntimeError("Division by zero")
                    self.gp[d] = int(self.gp[a] / self.gp[b])
                elif op == 0x0E: self.gp[self._u8()] += 1
                elif op == 0x0F: self.gp[self._u8()] -= 1
                elif op == 0x0D:
                    a, b = self._u8(), self._u8()
                    if   self.gp[a] < self.gp[b]: self.gp[13] = -1
                    elif self.gp[a] > self.gp[b]: self.gp[13] = 1
                    else: self.gp[13] = 0
                elif op == 0x07:
                    d = self._u8(); off = self._i16()
                    if self.gp[d] == 0: self.pc += off
                elif op == 0x06:
                    d = self._u8(); off = self._i16()
                    if self.gp[d] != 0: self.pc += off
                elif op == 0x05:
                    off = self._i16()
                    self.pc += off
                elif op == 0x01:
                    self.stack.append(self.gp[self._u8()])
                elif op == 0x02:
                    if not self.stack: raise RuntimeError("Stack underflow")
                    self.gp[self._u8()] = self.stack.pop()
                else:
                    raise ValueError(f"Unknown opcode: 0x{op:02X} at PC={self.pc - 1}")
        except Exception as e:
            self.error = str(e)
        return self

    def result(self, reg: int = 0) -> Optional[int]:
        """Get register value if execution succeeded."""
        return self.gp[reg] if self.halted and self.error is None else None


# ─── Assembler ───────────────────────────────────────────────────────────────

class Assembler:
    """Text assembly → bytecode converter with label support."""

    def __init__(self):
        self.opcodes = OPCODES
        self.labels = {}
        self.lines = []

    def assemble(self, text: str) -> bytes:
        """Assemble FLUX assembly text into bytecode."""
        self.labels = {}
        self.lines = []

        # First pass: collect labels, strip comments
        pc = 0
        for raw_line in text.split('\n'):
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            # Check for label
            if ':' in line:
                label_part, _, rest = line.partition(':')
                self.labels[label_part.strip()] = pc
                line = rest.strip()
                if not line:
                    continue
            self.lines.append(line)
            # Calculate instruction size
            mn = line.replace(',', ' ').split()[0].upper()
            if mn == 'HALT':
                pc += 1
            elif mn in ('DEC', 'INC', 'PUSH', 'POP'):
                pc += 2
            elif mn in ('IADD', 'ISUB', 'IMUL', 'IDIV'):
                pc += 4
            elif mn in ('MOV', 'CMP'):
                pc += 3
            elif mn in ('MOVI', 'JNZ', 'JZ'):
                pc += 4
            elif mn == 'JMP':
                pc += 3

        # Second pass: emit bytecode
        bc = bytearray()
        for line in self.lines:
            parts = line.replace(',', ' ').split()
            mn = parts[0].upper()
            if mn not in self.opcodes:
                raise ValueError(f"Unknown mnemonic: {mn}")
            op = self.opcodes[mn]

            if mn == 'HALT':
                bc.append(op)
            elif mn in ('DEC', 'INC', 'PUSH', 'POP'):
                bc.append(op)
                bc.append(int(parts[1][1:]))  # Rn
            elif mn in ('IADD', 'ISUB', 'IMUL', 'IDIV'):
                bc.append(op)
                bc.append(int(parts[1][1:]))  # Rd
                bc.append(int(parts[2][1:]))  # Ra
                bc.append(int(parts[3][1:]))  # Rb
            elif mn in ('MOV', 'CMP'):
                bc.append(op)
                bc.append(int(parts[1][1:]))  # Rd/Ra
                bc.append(int(parts[2][1:]))  # Ra/Rb
            elif mn == 'MOVI':
                bc.append(op)
                bc.append(int(parts[1][1:]))  # Rd
                val = self._resolve_value(parts[2], len(bc) + 2)
                bc.extend(struct.pack('<h', val))
            elif mn in ('JNZ', 'JZ'):
                bc.append(op)
                bc.append(int(parts[1][1:]))  # Rd
                val = self._resolve_value(parts[2], len(bc) + 2)
                bc.extend(struct.pack('<h', val))
            elif mn == 'JMP':
                bc.append(op)
                val = self._resolve_value(parts[1], len(bc) + 2)
                bc.extend(struct.pack('<h', val))

        return bytes(bc)

    def _resolve_value(self, token: str, current_pc: int) -> int:
        """Resolve a value — could be a number, label, or relative offset."""
        token = token.strip()
        # Try as integer
        try:
            return int(token)
        except ValueError:
            pass
        # Try as label
        if token in self.labels:
            # Convert to relative offset from current PC
            return self.labels[token] - current_pc
        raise ValueError(f"Cannot resolve: {token}")


# ─── Disassembler ────────────────────────────────────────────────────────────

class Disassembler:
    """Bytecode → text converter."""

    def __init__(self):
        self.mnemonics = {v: k for k, v in OPCODES.items()}

    def disassemble(self, bytecode: bytes) -> str:
        """Disassemble bytecode into human-readable text."""
        lines = []
        pc = 0
        while pc < len(bytecode):
            addr = pc
            op = bytecode[pc]
            pc += 1
            mn = self.mnemonics.get(op, f'??? (0x{op:02X})')

            if op == 0x80:  # HALT
                lines.append(f'{addr:04X}: {mn}')
            elif op in (0x0F, 0x0E, 0x01, 0x02):  # DEC, INC, PUSH, POP
                rd = bytecode[pc]; pc += 1
                lines.append(f'{addr:04X}: {mn} R{rd}')
            elif op in (0x08, 0x09, 0x0A, 0x0B):  # IADD, ISUB, IMUL, IDIV
                rd, ra, rb = bytecode[pc], bytecode[pc+1], bytecode[pc+2]; pc += 3
                lines.append(f'{addr:04X}: {mn} R{rd}, R{ra}, R{rb}')
            elif op in (0x2C, 0x0D):  # MOV, CMP
                ra, rb = bytecode[pc], bytecode[pc+1]; pc += 2
                lines.append(f'{addr:04X}: {mn} R{ra}, R{rb}')
            elif op in (0x2B, 0x06, 0x07):  # MOVI, JNZ, JZ
                rd = bytecode[pc]; pc += 1
                val = bytecode[pc] | (bytecode[pc+1] << 8); pc += 2
                if val >= 32768: val -= 65536
                lines.append(f'{addr:04X}: {mn} R{rd}, {val}')
            elif op == 0x05:  # JMP
                val = bytecode[pc] | (bytecode[pc+1] << 8); pc += 2
                if val >= 32768: val -= 65536
                lines.append(f'{addr:04X}: {mn} {val}')
            else:
                lines.append(f'{addr:04X}: {mn}')

        return '\n'.join(lines)


# ─── Vocabulary ──────────────────────────────────────────────────────────────

class Vocabulary:
    """Natural language pattern → bytecode assembly mapping."""

    def __init__(self):
        self.patterns = []

    def add(self, pattern: str, assembly: str, result_reg: int = 0, description: str = ""):
        """Add a vocabulary entry."""
        self.patterns.append({
            'pattern': pattern,
            'assembly': assembly,
            'result_reg': result_reg,
            'description': description,
            'regex': self._compile_regex(pattern)
        })

    def _compile_regex(self, pattern: str) -> re.Pattern:
        """Compile pattern string into regex."""
        parts = re.split(r'(\$\w+)', pattern)
        regex_parts = []
        for p in parts:
            if p.startswith('$'):
                regex_parts.append(f'(?P<{p[1:]}>\\d+)')
            else:
                regex_parts.append(re.escape(p))
        return re.compile(''.join(regex_parts), re.IGNORECASE)

    def match(self, text: str) -> Optional[dict]:
        """Match text against vocabulary patterns."""
        for entry in self.patterns:
            m = entry['regex'].search(text)
            if m:
                return {
                    'assembly': entry['assembly'],
                    'result_reg': entry['result_reg'],
                    'description': entry['description'],
                    'groups': m.groupdict()
                }
        return None

    def get_builtin(self) -> 'Vocabulary':
        """Load built-in vocabulary patterns."""
        self.add("compute $a + $b",
                 "MOVI R0, ${a}\nMOVI R1, ${b}\nIADD R0, R0, R1\nHALT",
                 result_reg=0, description="Add two numbers")
        self.add("compute $a - $b",
                 "MOVI R0, ${a}\nMOVI R1, ${b}\nISUB R0, R0, R1\nHALT",
                 result_reg=0, description="Subtract two numbers")
        self.add("compute $a * $b",
                 "MOVI R0, ${a}\nMOVI R1, ${b}\nIMUL R0, R0, R1\nHALT",
                 result_reg=0, description="Multiply two numbers")
        self.add("compute $a / $b",
                 "MOVI R0, ${a}\nMOVI R1, ${b}\nIDIV R0, R0, R1\nHALT",
                 result_reg=0, description="Divide two numbers")
        self.add("factorial of $n",
                 "MOVI R0, ${n}\nMOVI R1, 1\nloop: IMUL R1, R1, R0\nDEC R0\nJNZ R0, loop\nHALT",
                 result_reg=1, description="Compute n!")
        self.add("fibonacci of $n",
                 "MOVI R0, ${n}\nMOVI R1, 0\nMOVI R2, 1\nDEC R0\nloop: MOV R3, R2\nIADD R2, R2, R1\nMOV R1, R3\nDEC R0\nJNZ R0, loop\nHALT",
                 result_reg=2, description="Compute F(n)")
        self.add("double $n",
                 "MOVI R0, ${n}\nIADD R0, R0, R0\nHALT",
                 result_reg=0, description="Double a number")
        self.add("square $n",
                 "MOVI R0, ${n}\nIMUL R0, R0, R0\nHALT",
                 result_reg=0, description="Square a number")
        self.add("sum $a to $b",
                 "MOVI R0, ${a}\nMOVI R1, ${b}\nMOVI R2, 0\nloop: IADD R2, R2, R0\nCMP R0, R1\nJZ R13, done\nINC R0\nJNZ R13, loop\ndone: HALT",
                 result_reg=2, description="Sum from a to b inclusive")
        self.add("power of $base to $exp",
                 "MOVI R0, ${base}\nMOVI R1, ${exp}\nMOVI R2, 1\nloop: IMUL R2, R2, R0\nDEC R1\nJNZ R1, loop\nHALT",
                 result_reg=2, description="Compute base^exp")
        self.add("hello",
                 "MOVI R0, 42\nHALT",
                 result_reg=0, description="Returns 42")
        return self


# ─── Interpreter ─────────────────────────────────────────────────────────────

class Interpreter:
    """Natural language → FLUX bytecode → result."""

    def __init__(self, vocab: Vocabulary = None):
        self.vocab = vocab or Vocabulary().get_builtin()
        self.assembler = Assembler()

    def run(self, text: str) -> tuple[Optional[int], str]:
        """
        Interpret natural language text.
        Returns (result, status_message).
        """
        # Try vocabulary match
        match = self.vocab.match(text)
        if match:
            asm = match['assembly']
            for k, v in match['groups'].items():
                asm = asm.replace('${' + k + '}', str(v))
            try:
                bc = self.assembler.assemble(asm)
                vm = FluxVM(bc)
                vm.execute()
                if vm.halted:
                    return vm.reg(match['result_reg']), f"OK ({vm.cycles} cycles)"
                return None, f"VM error: {vm.error}"
            except Exception as e:
                return None, f"Assembly error: {e}"

        return None, f"No match for: {text[:80]}"


# ─── Convenience ─────────────────────────────────────────────────────────────

def assemble(text: str) -> bytes:
    """Convenience function: assemble text to bytecode."""
    return Assembler().assemble(text)


# ─── Tests ───────────────────────────────────────────────────────────────────

def run_tests():
    """Run comprehensive tests."""
    asm = Assembler()
    dis = Disassembler()
    interp = Interpreter()

    print("=" * 50)
    print("FLUX VM Test Suite")
    print("=" * 50)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Simple arithmetic (IADD)
    print("\n1. Test IADD: 3 + 4 = 7")
    bc = asm.assemble("MOVI R0, 3\nMOVI R1, 4\nIADD R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 7:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 7, got {result}")
        tests_failed += 1

    # Test 2: Multiplication (IMUL)
    print("\n2. Test IMUL: 5 * 6 = 30")
    bc = asm.assemble("MOVI R0, 5\nMOVI R1, 6\nIMUL R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 30:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 30, got {result}")
        tests_failed += 1

    # Test 3: Division (IDIV)
    print("\n3. Test IDIV: 20 / 4 = 5")
    bc = asm.assemble("MOVI R0, 20\nMOVI R1, 4\nIDIV R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 5:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 5, got {result}")
        tests_failed += 1

    # Test 4: Loop with label (JNZ, DEC)
    print("\n4. Test loop with label: 3 + 2 + 1 = 6")
    bc = asm.assemble("MOVI R0, 3\nMOVI R1, 0\nloop: IADD R1, R1, R0\nDEC R0\nJNZ R0, loop\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(1)
    if result == 6:
        print(f"   ✓ PASS: R1 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 6, got {result}")
        tests_failed += 1

    # Test 5: Disassembler
    print("\n5. Test disassembler")
    bc = asm.assemble("MOVI R0, 42\nHALT")
    disasm = dis.disassemble(bc)
    if "MOVI R0, 42" in disasm and "HALT" in disasm:
        print(f"   ✓ PASS:\n{disasm}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Unexpected disassembly:\n{disasm}")
        tests_failed += 1

    # Test 6: Vocabulary - compute A + B
    print("\n6. Test vocabulary: 'compute 7 + 5'")
    result, msg = interp.run("compute 7 + 5")
    if result == 12:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 12, got {result}")
        tests_failed += 1

    # Test 7: Vocabulary - compute A * B
    print("\n7. Test vocabulary: 'compute 8 * 9'")
    result, msg = interp.run("compute 8 * 9")
    if result == 72:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 72, got {result}")
        tests_failed += 1

    # Test 8: Vocabulary - factorial
    print("\n8. Test vocabulary: 'factorial of 5'")
    result, msg = interp.run("factorial of 5")
    if result == 120:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 120, got {result}")
        tests_failed += 1

    # Test 9: Vocabulary - double
    print("\n9. Test vocabulary: 'double 21'")
    result, msg = interp.run("double 21")
    if result == 42:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 42, got {result}")
        tests_failed += 1

    # Test 10: Vocabulary - square
    print("\n10. Test vocabulary: 'square 7'")
    result, msg = interp.run("square 7")
    if result == 49:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 49, got {result}")
        tests_failed += 1

    # ─── New opcode tests ───────────────────────────────────────────────

    # Test 11: ISUB
    print("\n11. Test ISUB: 10 - 3 = 7")
    bc = asm.assemble("MOVI R0, 10\nMOVI R1, 3\nISUB R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 7:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 7, got {result}")
        tests_failed += 1

    # Test 12: INC
    print("\n12. Test INC: R0 starts at 5, INC → 6")
    bc = asm.assemble("MOVI R0, 5\nINC R0\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 6:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 6, got {result}")
        tests_failed += 1

    # Test 13: MOV
    print("\n13. Test MOV: copy R1=99 into R0")
    bc = asm.assemble("MOVI R1, 99\nMOV R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 99:
        print(f"   ✓ PASS: R0 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 99, got {result}")
        tests_failed += 1

    # Test 14: CMP (less than)
    print("\n14. Test CMP: 3 < 5 → R13 = -1")
    bc = asm.assemble("MOVI R0, 3\nMOVI R1, 5\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(13)
    if result == -1:
        print(f"   ✓ PASS: R13 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected -1, got {result}")
        tests_failed += 1

    # Test 15: CMP (equal)
    print("\n15. Test CMP: 7 == 7 → R13 = 0")
    bc = asm.assemble("MOVI R0, 7\nMOVI R1, 7\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(13)
    if result == 0:
        print(f"   ✓ PASS: R13 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 0, got {result}")
        tests_failed += 1

    # Test 16: CMP (greater than)
    print("\n16. Test CMP: 9 > 4 → R13 = 1")
    bc = asm.assemble("MOVI R0, 9\nMOVI R1, 4\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(13)
    if result == 1:
        print(f"   ✓ PASS: R13 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 1, got {result}")
        tests_failed += 1

    # Test 17: JZ (jump when zero)
    print("\n17. Test JZ: R0=0 should jump, R1 stays 0")
    bc = asm.assemble("MOVI R0, 0\nMOVI R1, 1\nJZ R0, skip\nMOVI R1, 2\nskip: HALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(1)
    if result == 1:
        print(f"   ✓ PASS: R1 = {result} (jumped over MOVI R1, 2)")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 1, got {result}")
        tests_failed += 1

    # Test 18: JZ (no jump when non-zero)
    print("\n18. Test JZ: R0=5 should not jump, R1 becomes 2")
    bc = asm.assemble("MOVI R0, 5\nMOVI R1, 1\nJZ R0, 1\nMOVI R1, 2\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(1)
    if result == 2:
        print(f"   ✓ PASS: R1 = {result} (did not jump)")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 2, got {result}")
        tests_failed += 1

    # Test 19: JMP (unconditional jump)
    print("\n19. Test JMP: skip over instruction")
    bc = asm.assemble("MOVI R0, 1\nJMP skip\nMOVI R0, 2\nskip: HALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(0)
    if result == 1:
        print(f"   ✓ PASS: R0 = {result} (jumped over MOVI R0, 2)")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 1, got {result}")
        tests_failed += 1

    # Test 20: JMP with label (loop)
    print("\n20. Test JMP with label: count down from 3 to 0")
    bc = asm.assemble("MOVI R0, 3\nMOVI R1, 0\nloop: INC R1\nDEC R0\nJNZ R0, loop\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    result = vm.result(1)
    if result == 3:
        print(f"   ✓ PASS: R1 = {result}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 3, got {result}")
        tests_failed += 1

    # Test 21: PUSH and POP
    print("\n21. Test PUSH/POP: push 42, pop into R2")
    bc = asm.assemble("MOVI R0, 42\nPUSH R0\nMOVI R1, 0\nPOP R2\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    r2 = vm.result(2)
    if r2 == 42:
        print(f"   ✓ PASS: R2 = {r2} (pushed and popped 42)")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 42, got {r2}")
        tests_failed += 1

    # Test 22: PUSH/POP preserve order (LIFO)
    print("\n22. Test PUSH/POP LIFO order: push 1,2 then pop → 2,1")
    bc = asm.assemble("MOVI R0, 1\nMOVI R1, 2\nPUSH R0\nPUSH R1\nPOP R2\nPOP R3\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    r2, r3 = vm.result(2), vm.result(3)
    if r2 == 2 and r3 == 1:
        print(f"   ✓ PASS: R2={r2}, R3={r3} (LIFO order)")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected R2=2, R3=1, got R2={r2}, R3={r3}")
        tests_failed += 1

    # Test 23: POP on empty stack
    print("\n23. Test POP on empty stack → error")
    bc = asm.assemble("POP R0\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    if vm.error is not None and "Stack underflow" in vm.error:
        print(f"   ✓ PASS: Got expected error: {vm.error}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected stack underflow error, got error={vm.error}")
        tests_failed += 1

    # ─── New vocabulary tests ───────────────────────────────────────────

    # Test 24: Vocabulary - compute A - B
    print("\n24. Test vocabulary: 'compute 20 - 8'")
    result, msg = interp.run("compute 20 - 8")
    if result == 12:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 12, got {result}")
        tests_failed += 1

    # Test 25: Vocabulary - compute A / B
    print("\n25. Test vocabulary: 'compute 100 / 4'")
    result, msg = interp.run("compute 100 / 4")
    if result == 25:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 25, got {result}")
        tests_failed += 1

    # Test 26: Vocabulary - fibonacci of 7
    print("\n26. Test vocabulary: 'fibonacci of 7'")
    result, msg = interp.run("fibonacci of 7")
    if result == 13:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 13, got {result}")
        tests_failed += 1

    # Test 27: Vocabulary - sum 1 to 100
    print("\n27. Test vocabulary: 'sum 1 to 100'")
    result, msg = interp.run("sum 1 to 100")
    if result == 5050:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 5050, got {result}")
        tests_failed += 1

    # Test 28: Vocabulary - power of 2 to 10
    print("\n28. Test vocabulary: 'power of 2 to 10'")
    result, msg = interp.run("power of 2 to 10")
    if result == 1024:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 1024, got {result}")
        tests_failed += 1

    # Test 29: Vocabulary - hello
    print("\n29. Test vocabulary: 'hello'")
    result, msg = interp.run("hello")
    if result == 42:
        print(f"   ✓ PASS: Result = {result} ({msg})")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: Expected 42, got {result}")
        tests_failed += 1

    # ─── Disassembler tests for new opcodes ─────────────────────────────

    # Test 30: Disassemble ISUB
    print("\n30. Test disassemble ISUB")
    bc = asm.assemble("MOVI R0, 10\nMOVI R1, 3\nISUB R0, R0, R1\nHALT")
    disasm = dis.disassemble(bc)
    if "ISUB" in disasm:
        print(f"   ✓ PASS:\n{disasm}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: ISUB not found in disassembly:\n{disasm}")
        tests_failed += 1

    # Test 31: Disassemble MOV
    print("\n31. Test disassemble MOV")
    bc = asm.assemble("MOVI R1, 7\nMOV R0, R1\nHALT")
    disasm = dis.disassemble(bc)
    if "MOV R0, R1" in disasm:
        print(f"   ✓ PASS:\n{disasm}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: MOV not found in disassembly:\n{disasm}")
        tests_failed += 1

    # Test 32: Disassemble PUSH/POP
    print("\n32. Test disassemble PUSH/POP")
    bc = asm.assemble("MOVI R0, 5\nPUSH R0\nPOP R1\nHALT")
    disasm = dis.disassemble(bc)
    if "PUSH R0" in disasm and "POP R1" in disasm:
        print(f"   ✓ PASS:\n{disasm}")
        tests_passed += 1
    else:
        print(f"   ✗ FAIL: PUSH/POP not found in disassembly:\n{disasm}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 50)

    return tests_failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
