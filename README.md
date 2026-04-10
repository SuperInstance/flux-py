# FLUX.py — Clean Python VM

Self-contained FLUX bytecode runtime in a single file. Zero dependencies.

![FLUX Logo](flux-logo.jpg)

## Features

- **VM** — 16 registers, arithmetic, control flow, stack operations
- **Assembler** — text assembly → bytecode with label support
- **Disassembler** — bytecode → human-readable listing
- **Vocabulary** — natural language → bytecode (11 built-in patterns)
- **A2A Agents** — multi-agent coordination with messaging
- **Swarm** — broadcast, vote, consensus across agents

## Quick Start

```python
from flux_vm import FluxVM, assemble

# Assemble and run
bc = assemble('''
    MOVI R0, 7
    MOVI R1, 1
    IMUL R1, R1, R0
    DEC R0
    JNZ R0, -10
    HALT
''')
vm = FluxVM(bc)
vm.execute()
print(vm.reg(1))  # 5040 (factorial of 7)
```

## Natural Language

```python
from flux_vm import Interpreter

interp = Interpreter()
result, msg = interp.run("factorial of 7")
print(result)  # 5040

result, msg = interp.run("sum 1 to 100")
print(result)  # 5050

result, msg = interp.run("power of 2 to 10")
print(result)  # 1024
```

## Assembly Syntax

```
MOVI R0, 42        # Load immediate
MOV R0, R1         # Copy register
IADD R0, R1, R2    # R0 = R1 + R2
ISUB R0, R1, R2    # R0 = R1 - R2
IMUL R0, R1, R2    # R0 = R1 * R2
IDIV R0, R1, R2    # R0 = R1 / R2
INC R0              # R0++
DEC R0              # R0--
CMP R0, R1          # Compare → R13 (-1, 0, 1)
JNZ R0, offset      # Jump if not zero
JZ R0, offset       # Jump if zero
JMP offset          # Unconditional jump
PUSH R0             # Push to stack
POP R0              # Pop from stack
HALT                # Stop execution
```

## A2A Swarm

```python
from flux_vm import A2AAgent, Swarm, assemble

bc = assemble('MOVI R0, 42\nHALT')
swarm = Swarm()

for i in range(5):
    swarm.add(A2AAgent(f'agent-{i}', bc))

swarm.tick()
print(swarm.consensus(reg=0))  # 42
```

## Built-in Vocabulary

| Pattern | Description |
|---------|-------------|
| `compute $a + $b` | Addition |
| `compute $a - $b` | Subtraction |
| `compute $a * $b` | Multiplication |
| `compute $a / $b` | Division |
| `double $a` | Double a number |
| `square $a` | Square a number |
| `factorial of $n` | n! |
| `fibonacci of $n` | F(n) |
| `sum $a to $b` | Sum range |
| `power of $base to $exp` | Exponentiation |
| `hello` | Returns 42 |

## Performance

~26K ns/iter for factorial(7) on Python. For production speed, use the Zig (210ns) or C (403ns) runtimes.

## CLI

```bash
python3 flux_vm.py                    # Interactive REPL
python3 flux_vm.py factorial of 7     # One-shot
```

## Part of the FLUX Ecosystem

- **flux-runtime** — Full Python runtime with compiler, debugger, REPL
- **flux-core** — Rust implementation
- **flux-runtime-c** — C implementation with assembler
- **flux-zig** — Zig implementation (fastest VM)
- **flux-js** — JavaScript implementation
- **flux-swarm** — Go swarm coordinator
- **flux-py** — This repo. Clean Python, single file, zero deps.

Same bytecode, different shells. 🦀
