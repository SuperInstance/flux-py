"""
FLUX Clean Python VM — Self-contained, zero-dependency bytecode runtime.

Features:
  - 16 general-purpose registers (R0-R15)
  - Arithmetic: IADD, ISUB, IMUL, IDIV
  - Control flow: JNZ, JZ, JMP, CMP, HALT
  - Memory: MOV, MOVI, INC, DEC, PUSH, POP
  - A2A Agent system for multi-agent coordination
  - Assembler: text assembly → bytecode
  - Vocabulary: natural language → bytecode

Usage:
  from flux_vm import FluxVM, assemble
  
  bc = assemble('''
      MOVI R0, 7
      MOVI R1, 1
  loop:
      IMUL R1, R1, R0
      DEC R0
      JNZ R0, -10
      HALT
  ''')
  vm = FluxVM(bc)
  vm.execute()
  print(vm.reg(1))  # 5040
"""

import struct
import re
import time
from typing import Optional, List, Dict, Tuple


# ─── Opcodes ────────────────────────────────────────────────────────────────

OPCODES = {
    'NOP':  0x00, 'MOV':  0x01, 'JMP':  0x07,
    'IADD': 0x08, 'ISUB': 0x09, 'IMUL': 0x0A, 'IDIV': 0x0B,
    'INC':  0x0E, 'DEC':  0x0F,
    'PUSH': 0x10, 'POP':  0x11,
    'JNZ':  0x06, 'JZ':   0x2E,
    'MOVI': 0x2B, 'CMP':  0x2D,
    'HALT': 0x80,
}


# ─── VM ─────────────────────────────────────────────────────────────────────

class FluxVM:
    """FLUX bytecode virtual machine with 16 registers."""

    MAX_CYCLES = 10_000_000

    def __init__(self, bytecode: bytes, max_cycles: int = None):
        self.bc = bytecode
        self.gp = [0] * 16
        self.pc = 0
        self.halted = False
        self.cycles = 0
        self.stack = []
        self.max_cycles = max_cycles or self.MAX_CYCLES
        self.error = None

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
                elif op == 0x00: pass  # NOP
                elif op == 0x01: d, s = self._u8(), self._u8(); self.gp[d] = self.gp[s]
                elif op == 0x2B: d = self._u8(); self.gp[d] = self._i16()
                elif op == 0x08: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] + self.gp[b]
                elif op == 0x09: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] - self.gp[b]
                elif op == 0x0A: d, a, b = self._u8(), self._u8(), self._u8(); self.gp[d] = self.gp[a] * self.gp[b]
                elif op == 0x0B:
                    d, a, b = self._u8(), self._u8(), self._u8()
                    if self.gp[b] == 0: raise RuntimeError("Division by zero")
                    self.gp[d] = int(self.gp[a] / self.gp[b])
                elif op == 0x0E: self.gp[self._u8()] += 1
                elif op == 0x0F: self.gp[self._u8()] -= 1
                elif op == 0x10: self.stack.append(self.gp[self._u8()])
                elif op == 0x11:
                    if self.stack: self.gp[self._u8()] = self.stack.pop()
                elif op == 0x06:
                    d = self._u8(); off = self._i16()
                    if self.gp[d] != 0: self.pc += off
                elif op == 0x2E:
                    d = self._u8(); off = self._i16()
                    if self.gp[d] == 0: self.pc += off
                elif op == 0x07: self.pc += self._i16()
                elif op == 0x2D:
                    a, b = self._u8(), self._u8()
                    self.gp[13] = (self.gp[a] > self.gp[b]) - (self.gp[a] < self.gp[b])
                else:
                    raise ValueError(f"Unknown opcode: 0x{op:02X} at PC={self.pc - 1}")
        except Exception as e:
            self.error = str(e)
        return self

    def result(self, reg: int = 0) -> Optional[int]:
        """Get register value if execution succeeded."""
        return self.gp[reg] if self.halted and self.error is None else None

    def dump(self) -> str:
        """Pretty-print register state."""
        regs = ' '.join(f'R{i}={v}' for i, v in enumerate(self.gp) if v != 0)
        return f"PC={self.pc} cycles={self.cycles} halted={self.halted} [{regs}]"


# ─── Assembler ──────────────────────────────────────────────────────────────

def assemble(text: str) -> bytes:
    """
    Assemble FLUX assembly text into bytecode.
    Supports labels (label:) and comments (# or ;).
    
    Format E (4 bytes): op, rd, rs1, rs2  — IADD, ISUB, IMUL, IDIV, CMP
    Format C (3 bytes): op, rd, rs        — MOV
    Format I (4 bytes): op, rd, imm16     — MOVI, JNZ, JZ
    Format J (3 bytes): op, offset16      — JMP
    Format S (2 bytes): op, rd            — INC, DEC, PUSH, POP
    Format X (1 byte):  op                — HALT, NOP
    """
    lines = []
    labels = {}
    
    # First pass: collect labels, strip comments
    pc = 0
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith(';'):
            continue
        # Check for label
        if ':' in line:
            label_part, _, rest = line.partition(':')
            labels[label_part.strip()] = pc
            line = rest.strip()
            if not line:
                continue
        lines.append(line)
        # Calculate instruction size
        mn = line.replace(',', ' ').split()[0].upper()
        if mn in ('HALT', 'NOP'):
            pc += 1
        elif mn in ('INC', 'DEC', 'PUSH', 'POP'):
            pc += 2
        elif mn in ('MOV',):
            pc += 3
        elif mn in ('IADD', 'ISUB', 'IMUL', 'IDIV', 'CMP'):
            pc += 4
        elif mn in ('MOVI', 'JNZ', 'JZ'):
            pc += 4
        elif mn == 'JMP':
            pc += 3
    
    # Second pass: emit bytecode
    bc = bytearray()
    for line in lines:
        parts = line.replace(',', ' ').split()
        mn = parts[0].upper()
        if mn not in OPCODES:
            raise ValueError(f"Unknown mnemonic: {mn}")
        op = OPCODES[mn]
        
        if mn in ('HALT', 'NOP'):
            bc.append(op)
        elif mn in ('INC', 'DEC', 'PUSH', 'POP'):
            bc.append(op)
            bc.append(int(parts[1][1:]))  # Rn
        elif mn == 'MOV':
            bc.append(op)
            bc.append(int(parts[1][1:]))  # Rd
            bc.append(int(parts[2][1:]))  # Rs
        elif mn in ('IADD', 'ISUB', 'IMUL', 'IDIV', 'CMP'):
            bc.append(op)
            bc.append(int(parts[1][1:]))  # Rd
            bc.append(int(parts[2][1:]))  # Ra
            bc.append(int(parts[3][1:]) if len(parts) > 3 else int(parts[2][1:]))  # Rb
        elif mn in ('MOVI',):
            bc.append(op)
            bc.append(int(parts[1][1:]))  # Rd
            val = _resolve_value(parts[2], labels, len(bc) + 2)
            bc.extend(struct.pack('<h', val))
        elif mn in ('JNZ', 'JZ'):
            bc.append(op)
            bc.append(int(parts[1][1:]))  # Rd
            val = _resolve_value(parts[2], labels, len(bc) + 2)
            bc.extend(struct.pack('<h', val))
        elif mn == 'JMP':
            bc.append(op)
            val = _resolve_value(parts[1], labels, len(bc) + 1)
            bc.extend(struct.pack('<h', val))
    
    return bytes(bc)


def _resolve_value(token: str, labels: dict, current_pc: int) -> int:
    """Resolve a value — could be a number, label, or relative offset."""
    token = token.strip()
    # Try as integer
    try:
        return int(token)
    except ValueError:
        pass
    # Try as label
    if token in labels:
        # Convert to relative offset from current PC
        return labels[token] - current_pc
    raise ValueError(f"Cannot resolve: {token}")


# ─── A2A Agent System ──────────────────────────────────────────────────────

class A2AAgent:
    """An agent that runs FLUX bytecode and communicates with others."""

    def __init__(self, agent_id: str, bytecode: bytes, role: str = "worker"):
        self.id = agent_id
        self.vm = FluxVM(bytecode)
        self.role = role
        self.trust = 1.0
        self.inbox = []
        self.outbox = []
        self.generation = 0

    def step(self) -> int:
        """Execute one VM run. Returns cycles consumed."""
        self.vm.execute()
        self.generation += 1
        return self.vm.cycles

    def tell(self, other: 'A2AAgent', payload: int):
        """Send a message to another agent."""
        other.inbox.append({
            'from': self.id,
            'type': 'TELL',
            'payload': payload,
            'gen': self.generation,
        })

    def ask(self, other: 'A2AAgent') -> Optional[int]:
        """Ask another agent for its result."""
        other.inbox.append({
            'from': self.id,
            'type': 'ASK',
            'gen': self.generation,
        })
        if other.vm.halted:
            return other.vm.reg(0)
        return None

    def read_inbox(self) -> list:
        """Read and clear inbox."""
        msgs = self.inbox[:]
        self.inbox.clear()
        return msgs


class Swarm:
    """Coordinate multiple A2A agents."""

    def __init__(self):
        self.agents: Dict[str, A2AAgent] = {}

    def add(self, agent: A2AAgent):
        self.agents[agent.id] = agent

    def tick(self) -> int:
        """Execute all agents one step. Returns total cycles."""
        total = 0
        for a in self.agents.values():
            total += a.step()
        return total

    def broadcast(self, from_id: str, value: int):
        """One agent tells all others a value."""
        sender = self.agents[from_id]
        for aid, agent in self.agents.items():
            if aid != from_id:
                sender.tell(agent, value)

    def vote(self, reg: int = 0) -> Dict[int, int]:
        """Majority vote across all agents on a register value."""
        counts = {}
        for a in self.agents.values():
            if a.vm.halted:
                v = a.vm.reg(reg)
                counts[v] = counts.get(v, 0) + 1
        return counts

    def consensus(self, reg: int = 0) -> Optional[int]:
        """Return the majority value, or None if no consensus."""
        counts = self.vote(reg)
        if not counts:
            return None
        return max(counts, key=counts.get)


# ─── Vocabulary ─────────────────────────────────────────────────────────────

class VocabEntry:
    """Maps a text pattern to bytecode assembly."""

    def __init__(self, pattern: str, assembly: str, name: str = "",
                 result_reg: int = 0, description: str = "", tags: list = None):
        self.pattern = pattern
        self.assembly = assembly
        self.name = name
        self.result_reg = result_reg
        self.description = description
        self.tags = tags or []
        self._regex = None

    def compile(self):
        parts = re.split(r'(\$\w+)', self.pattern)
        regex_parts = []
        for p in parts:
            if p.startswith('$'):
                regex_parts.append(f'(?P<{p[1:]}>\\d+)')
            else:
                regex_parts.append(re.escape(p))
        self._regex = re.compile(''.join(regex_parts), re.IGNORECASE)

    def match(self, text: str) -> Optional[dict]:
        if self._regex is None:
            self.compile()
        m = self._regex.search(text)
        return m.groupdict() if m else None


# Built-in vocabulary
BUILTIN_VOCAB = [
    VocabEntry("compute $a + $b",
               "MOVI R0, ${a}\nMOVI R1, ${b}\nIADD R0, R0, R1\nHALT",
               name="add", result_reg=0, description="Add two numbers"),
    VocabEntry("compute $a - $b",
               "MOVI R0, ${a}\nMOVI R1, ${b}\nISUB R0, R0, R1\nHALT",
               name="sub", result_reg=0, description="Subtract two numbers"),
    VocabEntry("compute $a * $b",
               "MOVI R0, ${a}\nMOVI R1, ${b}\nIMUL R0, R0, R1\nHALT",
               name="mul", result_reg=0, description="Multiply two numbers"),
    VocabEntry("compute $a / $b",
               "MOVI R0, ${a}\nMOVI R1, ${b}\nIDIV R0, R0, R1\nHALT",
               name="div", result_reg=0, description="Divide two numbers"),
    VocabEntry("double $a",
               "MOVI R0, ${a}\nIADD R0, R0, R0\nHALT",
               name="double", result_reg=0, description="Double a number"),
    VocabEntry("square $a",
               "MOVI R0, ${a}\nIMUL R0, R0, R0\nHALT",
               name="square", result_reg=0, description="Square a number"),
    VocabEntry("factorial of $n",
               "MOVI R0, ${n}\nMOVI R1, 1\nIMUL R1, R1, R0\nDEC R0\nJNZ R0, -10\nHALT",
               name="factorial", result_reg=1, description="Compute n!"),
    VocabEntry("fibonacci of $n",
               "MOVI R0, 0\nMOVI R1, 1\nMOVI R2, ${n}\nMOV R3, R1\nIADD R1, R1, R0\nMOV R0, R3\nDEC R2\nJNZ R2, -16\nHALT",
               name="fibonacci", result_reg=0, description="Compute F(n)"),
    VocabEntry("sum $a to $b",
               "MOVI R0, 0\nMOVI R1, ${b}\nIADD R0, R0, R1\nDEC R1\nJNZ R1, -10\nHALT",
               name="sum", result_reg=0, description="Sum from a to b"),
    VocabEntry("power of $base to $exp",
               "MOVI R0, 1\nMOVI R1, ${base}\nMOVI R2, ${exp}\nIMUL R0, R0, R1\nDEC R2\nJNZ R2, -10\nHALT",
               name="power", result_reg=0, description="Compute base^exp"),
    VocabEntry("hello",
               "MOVI R0, 42\nHALT",
               name="hello", result_reg=0, description="Returns 42"),
]


class Interpreter:
    """Natural language → FLUX bytecode via vocabulary matching."""

    def __init__(self, extra_vocab: List[VocabEntry] = None):
        self.vocab = list(BUILTIN_VOCAB)
        if extra_vocab:
            self.vocab.extend(extra_vocab)

    def run(self, text: str) -> Tuple[Optional[int], str]:
        """
        Interpret natural language text.
        Returns (result, status_message).
        """
        # Try vocabulary match
        for entry in self.vocab:
            groups = entry.match(text)
            if groups is not None:
                asm = entry.assembly
                for k, v in groups.items():
                    asm = asm.replace('${' + k + '}', str(v))
                try:
                    bc = assemble(asm)
                    vm = FluxVM(bc)
                    vm.execute()
                    if vm.halted:
                        return vm.reg(entry.result_reg), f"OK ({vm.cycles} cycles)"
                    return None, f"VM error: {vm.error}"
                except Exception as e:
                    return None, f"Assembly error: {e}"

        # Try direct assembly
        if any(l.strip().split()[0].upper() in OPCODES for l in text.split('\n') if l.strip()):
            try:
                bc = assemble(text)
                vm = FluxVM(bc)
                vm.execute()
                return vm.reg(0), f"OK ({vm.cycles} cycles, direct asm)"
            except Exception as e:
                return None, f"Error: {e}"

        # Try inline math
        m = re.search(r'(\d+)\s*([+\-*/×÷])\s*(\d+)', text)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            ops = {'+': a + b, '-': a - b, '*': a * b, '×': a * b, '/': int(a/b), '÷': int(a/b)}
            return ops.get(op), "OK (inline math)"

        return None, f"No match for: {text[:80]}"


# ─── Disassembler ───────────────────────────────────────────────────────────

def disassemble(bytecode: bytes) -> str:
    """Disassemble bytecode into human-readable text."""
    MNEMONICS = {v: k for k, v in OPCODES.items()}
    lines = []
    pc = 0
    while pc < len(bytecode):
        addr = pc
        op = bytecode[pc]
        pc += 1
        mn = MNEMONICS.get(op, f'??? (0x{op:02X})')

        if op in (0x80, 0x00):  # HALT, NOP
            lines.append(f'{addr:04X}: {mn}')
        elif op in (0x0E, 0x0F, 0x10, 0x11):  # INC, DEC, PUSH, POP
            rd = bytecode[pc]; pc += 1
            lines.append(f'{addr:04X}: {mn} R{rd}')
        elif op == 0x01:  # MOV
            rd, rs = bytecode[pc], bytecode[pc+1]; pc += 2
            lines.append(f'{addr:04X}: {mn} R{rd}, R{rs}')
        elif op in (0x08, 0x09, 0x0A, 0x0B, 0x2D):  # ALU 3-operand
            rd, ra, rb = bytecode[pc], bytecode[pc+1], bytecode[pc+2]; pc += 3
            lines.append(f'{addr:04X}: {mn} R{rd}, R{ra}, R{rb}')
        elif op in (0x2B, 0x06, 0x2E):  # MOVI, JNZ, JZ
            rd = bytecode[pc]; pc += 1
            val = bytecode[pc] | (bytecode[pc+1] << 8); pc += 2
            if val >= 32768: val -= 65536
            lines.append(f'{addr:04X}: {mn} R{rd}, {val}')
        elif op == 0x07:  # JMP
            val = bytecode[pc] | (bytecode[pc+1] << 8); pc += 2
            if val >= 32768: val -= 65536
            lines.append(f'{addr:04X}: {mn} {val}')
        else:
            lines.append(f'{addr:04X}: {mn}')

    return '\n'.join(lines)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    """Interactive FLUX REPL."""
    import sys

    print("╔══════════════════════════════════════════════════╗")
    print("║   FLUX.py — Clean Python Bytecode VM            ║")
    print("║   SuperInstance / Oracle1                        ║")
    print("║   Type 'help' for commands, 'quit' to exit       ║")
    print("╚══════════════════════════════════════════════════╝")

    interp = Interpreter()
    print(f"\n  {len(interp.vocab)} vocabulary patterns loaded\n")

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result, msg = interp.run(text)
        print(result if result is not None else msg)
        return

    while True:
        try:
            text = input('flux> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nbye')
            break
        if not text:
            continue
        if text in ('quit', 'exit', 'q'):
            print('bye')
            break
        if text == 'help':
            print('\n  Commands:')
            print('    <text>      — interpret and execute')
            print('    vocab       — list vocabulary patterns')
            print('    bench       — run benchmark')
            print('    demo        — run demos')
            print('    help        — show this help')
            print('    quit        — exit\n')
            continue
        if text == 'vocab':
            for v in interp.vocab:
                print(f'  {v.pattern:30s} {v.description}')
            continue
        if text == 'bench':
            _benchmark()
            continue
        if text == 'demo':
            _demo(interp)
            continue

        result, msg = interp.run(text)
        if result is not None:
            print(f'  → {result}  ({msg})')
        else:
            print(f'  ✗ {msg}')


def _benchmark():
    """Run FLUX performance benchmark."""
    fact_bc = assemble(
        'MOVI R0, 7\nMOVI R1, 1\nIMUL R1, R1, R0\nDEC R0\nJNZ R0, -10\nHALT'
    )
    N = 100_000
    t0 = time.monotonic()
    for _ in range(N):
        FluxVM(fact_bc).execute()
    elapsed = time.monotonic() - t0
    ns = elapsed * 1e9 / N
    print(f'  factorial(7) x {N:,}: {elapsed*1000:.1f} ms | {ns:.0f} ns/iter')


def _demo(interp: Interpreter):
    """Run comprehensive demos."""
    print('\n=== Arithmetic ===')
    for expr in ['compute 3 + 4', 'compute 100 * 7', 'double 21', 'square 8']:
        r, m = interp.run(expr)
        print(f'  {expr} → {r}')

    print('\n=== Loops ===')
    for expr in ['factorial of 7', 'fibonacci of 12', 'sum 1 to 100', 'power of 2 to 10']:
        r, m = interp.run(expr)
        print(f'  {expr} → {r} ({m})')

    print('\n=== Assembler ===')
    bc = assemble('MOVI R0, 42\nHALT')
    print(f'  MOVI R0, 42; HALT → {bc.hex()} → {FluxVM(bc).execute().reg(0)}')

    print('\n=== Disassembler ===')
    print(disassemble(assemble(
        'MOVI R0, 7\nMOVI R1, 1\nIMUL R1, R1, R0\nDEC R0\nJNZ R0, -10\nHALT'
    )))

    print('\n=== A2A Swarm ===')
    bc = assemble('MOVI R0, 42\nHALT')
    swarm = Swarm()
    for i in range(5):
        swarm.add(A2AAgent(f'agent-{i}', bc, role=['worker', 'scout', 'navigator'][i % 3]))
    cycles = swarm.tick()
    result = swarm.consensus(reg=0)
    print(f'  5 agents, {cycles} cycles, consensus: {result}')

    print()


if __name__ == '__main__':
    main()
