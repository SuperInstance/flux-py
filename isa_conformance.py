"""
FLUX ISA Translation & Conformance Layer.

Provides tools for managing multiple ISA dialects, translating bytecode
between them, measuring conformance, and computing compatibility.

Instruction encoding (shared across all dialects):
  - HALT:  1 byte  (opcode only)
  - DEC:   2 bytes (opcode + reg)
  - ADD/SUB/MUL/DIV (IADD/ISUB/IMUL/IDIV): 4 bytes (opcode + 3 regs)
  - MOVI:  4 bytes (opcode + reg + i16le)
  - JNZ:   4 bytes (opcode + reg + i16le)

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ─── Instruction Formats ────────────────────────────────────────────────────

class InsnFormat(Enum):
    """Encoding format for an instruction — determines total byte length."""
    NOP   = 1   # opcode only                          → 1 byte total
    R1    = 2   # opcode + 1 register                  → 2 bytes total
    R3    = 4   # opcode + 3 registers                 → 4 bytes total
    RI16  = 4   # opcode + 1 register + i16 (LE)       → 4 bytes total


# ─── Semantic Operation Identifiers ─────────────────────────────────────────

class Op(Enum):
    """Canonical semantic operations understood by all FLUX dialects."""
    HALT   = "HALT"
    DEC    = "DEC"
    ADD    = "ADD"
    SUB    = "SUB"
    MUL    = "MUL"
    DIV    = "DIV"
    MOVI   = "MOVI"
    JNZ    = "JNZ"


#: Default instruction format for each semantic operation.
OP_FORMAT: Dict[Op, InsnFormat] = {
    Op.HALT: InsnFormat.NOP,
    Op.DEC:  InsnFormat.R1,
    Op.ADD:  InsnFormat.R3,
    Op.SUB:  InsnFormat.R3,
    Op.MUL:  InsnFormat.R3,
    Op.DIV:  InsnFormat.R3,
    Op.MOVI: InsnFormat.RI16,
    Op.JNZ:  InsnFormat.RI16,
}


# ─── ISA Dialect ────────────────────────────────────────────────────────────

@dataclass
class ISADialect:
    """
    Represents a single ISA version / dialect.

    Attributes:
        name:            Human-readable name (e.g. "v1", "v2-extended").
        version:         Numeric version tag for ordering.
        mnemonic_to_op:  Map of assembly mnemonic (str) → opcode byte (int).
        op_to_mnemonic:  Map of opcode byte (int) → assembly mnemonic (str),
                         derived automatically.
        custom_ops:      Any opcodes that only exist in this dialect (not in
                         the canonical Op enum), keyed by mnemonic.
    """

    name: str
    version: int
    mnemonic_to_op: Dict[str, int] = field(default_factory=dict)
    custom_ops: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Build inverse mapping
        self.op_to_mnemonic: Dict[int, str] = {
            v: k for k, v in self.mnemonic_to_op.items()
        }

    # ── Known Opcodes ──

    @property
    def opcodes(self) -> Set[int]:
        """All opcode byte values known to this dialect."""
        return set(self.mnemonic_to_op.values())

    @property
    def mnemonic_set(self) -> Set[str]:
        return set(self.mnemonic_to_op.keys())

    def opcode_for(self, mnemonic: str) -> Optional[int]:
        return self.mnemonic_to_op.get(mnemonic)

    def mnemonic_for(self, opcode: int) -> Optional[str]:
        return self.op_to_mnemonic.get(opcode)

    def instruction_size(self, opcode: int) -> int:
        """Return total byte length of the instruction starting with *opcode*."""
        mn = self.mnemonic_for(opcode)
        if mn is None:
            raise ValueError(
                f"Unknown opcode 0x{opcode:02X} in dialect '{self.name}'"
            )
        fmt = self._format_for_mnemonic(mn)
        return fmt.value

    def _format_for_mnemonic(self, mnemonic: str) -> InsnFormat:
        """Resolve instruction format for a mnemonic (canonical or custom)."""
        mnemonic_upper = mnemonic.upper()
        for op in Op:
            if op.value == mnemonic_upper:
                return OP_FORMAT[op]
        # Custom ops default to R3 — callers should override if needed.
        return InsnFormat.R3

    def validate_bytecode(self, bytecode: bytes) -> List[TranslationIssue]:
        """Check whether *bytecode* conforms to this dialect.

        Returns a list of issues (empty = fully conformant).
        """
        issues: List[TranslationIssue] = []
        pc = 0
        while pc < len(bytecode):
            op = bytecode[pc]
            if op not in self.opcodes:
                issues.append(TranslationIssue(
                    offset=pc,
                    opcode=op,
                    kind="unknown_opcode",
                    message=(
                        f"Unknown opcode 0x{op:02X} at offset 0x{pc:04X} "
                        f"in dialect '{self.name}'"
                    ),
                ))
                # Cannot determine instruction length — skip 1 byte
                pc += 1
                continue
            try:
                size = self.instruction_size(op)
                if pc + size > len(bytecode):
                    issues.append(TranslationIssue(
                        offset=pc,
                        opcode=op,
                        kind="truncated",
                        message=(
                            f"Truncated instruction 0x{op:02X} at offset "
                            f"0x{pc:04X}: expected {size} bytes, "
                            f"only {len(bytecode) - pc} available"
                        ),
                    ))
                pc += size
            except ValueError:
                pc += 1
        return issues

    # ── Built-in Dialect Factories ──

    @staticmethod
    def v1() -> ISADialect:
        """ISA v1 as used by the original flux_vm.py.

        Opcodes: MOVI=0x2B, IADD=0x08, IMUL=0x0A, IDIV=0x0B,
                 DEC=0x0F, JNZ=0x06, HALT=0x80
        """
        return ISADialect(
            name="v1",
            version=1,
            mnemonic_to_op={
                "MOVI": 0x2B,
                "IADD": 0x08,
                "IMUL": 0x0A,
                "IDIV": 0x0B,
                "DEC":  0x0F,
                "JNZ":  0x06,
                "HALT": 0x80,
            },
        )

    @staticmethod
    def v2() -> ISADialect:
        """ISA v2 (converged) as used by flux-runtime.

        Opcodes: ADD=0x20, SUB=0x21, MUL=0x22, DIV=0x23,
                 MOVI=0x30, DEC=0x31, JNZ=0x40, HALT=0x00
        """
        return ISADialect(
            name="v2",
            version=2,
            mnemonic_to_op={
                "ADD":  0x20,
                "SUB":  0x21,
                "MUL":  0x22,
                "DIV":  0x23,
                "MOVI": 0x30,
                "DEC":  0x31,
                "JNZ":  0x40,
                "HALT": 0x00,
            },
        )

    @classmethod
    def custom(
        cls,
        name: str,
        version: int,
        mapping: Dict[str, int],
    ) -> ISADialect:
        """Create a dialect from an arbitrary mnemonic→opcode mapping."""
        return cls(name=name, version=version, mnemonic_to_op=mapping)

    # ── Canonical Name Resolution ──

    def _canonical_op(self, mnemonic: str) -> Optional[Op]:
        """Map a dialect-specific mnemonic to a canonical Op enum member.

        v1 uses IADD/IMUL/IDIV while v2 uses ADD/MUL/DIV — both map to
        the same canonical operation.
        """
        upper = mnemonic.upper()
        # Direct match
        for op in Op:
            if op.value == upper:
                return op
        # Aliases
        aliases = {
            "IADD": Op.ADD,
            "ISUB": Op.SUB,
            "IMUL": Op.MUL,
            "IDIV": Op.DIV,
        }
        return aliases.get(upper)

    def canonical_ops(self) -> Dict[int, Op]:
        """Map every opcode in this dialect to its canonical Op."""
        result: Dict[int, Op] = {}
        for opcode, mn in self.op_to_mnemonic.items():
            cop = self._canonical_op(mn)
            if cop is not None:
                result[opcode] = cop
        return result

    # ── Serialisation ──

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "mnemonic_to_op": {
                k: f"0x{v:02X}" for k, v in sorted(self.mnemonic_to_op.items())
            },
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


# ─── Translation Issue ──────────────────────────────────────────────────────

@dataclass
class TranslationIssue:
    """A single problem encountered during translation or validation."""
    offset: int
    opcode: int
    kind: str          # "unknown_opcode" | "truncated" | "unmapped"
    message: str


# ─── Translation Result ─────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """Outcome of a bytecode translation."""
    bytecode: bytes
    source_dialect: str
    target_dialect: str
    total_instructions: int = 0
    mapped: int = 0
    unmapped: int = 0
    issues: List[TranslationIssue] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.unmapped == 0 and not self.issues

    @property
    def coverage(self) -> float:
        """Fraction of instructions successfully mapped (0.0 – 1.0)."""
        if self.total_instructions == 0:
            return 1.0
        return self.mapped / self.total_instructions

    def summary(self) -> str:
        lines = [
            f"Translation: {self.source_dialect} → {self.target_dialect}",
            f"  Instructions: {self.total_instructions} total, "
            f"{self.mapped} mapped, {self.unmapped} unmapped",
            f"  Coverage: {self.coverage:.1%}",
        ]
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for iss in self.issues[:10]:
                lines.append(f"    [0x{iss.offset:04X}] {iss.kind}: {iss.message}")
            if len(self.issues) > 10:
                lines.append(f"    ... and {len(self.issues) - 10} more")
        return "\n".join(lines)


# ─── ISA Translator ─────────────────────────────────────────────────────────

class ISATranslator:
    """
    Translates bytecode from one ISA dialect to another.

    The translator works by building an opcode-to-opcode mapping derived
    from the canonical Op enum shared by both dialects.  Operands are
    copied verbatim since the instruction encoding format is identical
    across all supported dialects.
    """

    def __init__(
        self,
        source: ISADialect,
        target: ISADialect,
        *,
        strict: bool = True,
    ) -> None:
        """
        Args:
            source: The dialect the input bytecode is encoded in.
            target: The dialect to translate into.
            strict: If True, raise on the first unmapped opcode.
                    If False, flag unmapped opcodes as issues and emit
                    a placeholder (0xFF).
        """
        self.source = source
        self.target = target
        self.strict = strict
        self._remap = self._build_remap()

    # ── Internal ──

    def _build_remap(self) -> Dict[int, int]:
        """Build the source-opcode → target-opcode mapping table."""
        src_canon: Dict[int, Op] = self.source.canonical_ops()
        tgt_opcodes: Dict[Op, int] = {}
        for opcode, mn in self.target.op_to_mnemonic.items():
            cop = self.target._canonical_op(mn)
            if cop is not None:
                tgt_opcodes[cop] = opcode

        remap: Dict[int, int] = {}
        for src_opcode, cop in src_canon.items():
            tgt_opcode = tgt_opcodes.get(cop)
            if tgt_opcode is not None:
                remap[src_opcode] = tgt_opcode
        return remap

    def _unmapped_opcodes(self) -> Set[int]:
        """Source opcodes that have no mapping in the target dialect."""
        return self.source.opcodes - set(self._remap.keys())

    # ── Public API ──

    def mapping_table(self) -> Dict[int, int]:
        """Return the computed source→target opcode mapping (read-only view)."""
        return dict(self._remap)

    def unmapped_opcodes(self) -> Set[int]:
        """Return the set of source opcodes that cannot be translated."""
        return set(self._unmapped_opcodes())

    def validate_source(self, bytecode: bytes) -> List[TranslationIssue]:
        """Validate that *bytecode* conforms to the source dialect."""
        return self.source.validate_bytecode(bytecode)

    def translate(self, bytecode: bytes) -> TranslationResult:
        """Translate *bytecode* from source to target dialect.

        Operands (registers, immediates, offsets) are copied unchanged.
        Opcodes not present in the target dialect are flagged as issues.
        """
        issues: List[TranslationIssue] = []
        out = bytearray()
        pc = 0
        total = 0
        mapped = 0
        unmapped = 0

        while pc < len(bytecode):
            op = bytecode[pc]

            # Unknown opcode in source dialect
            if op not in self.source.opcodes:
                issues.append(TranslationIssue(
                    offset=pc,
                    opcode=op,
                    kind="unknown_opcode",
                    message=(
                        f"Unknown opcode 0x{op:02X} at offset 0x{pc:04X} "
                        f"in source dialect '{self.source.name}'"
                    ),
                ))
                if self.strict:
                    raise ValueError(
                        f"Unknown opcode 0x{op:02X} at offset 0x{pc:04X}"
                    )
                # Emit placeholder and skip
                out.append(0xFF)
                pc += 1
                total += 1
                unmapped += 1
                continue

            size = self.source.instruction_size(op)
            # Guard truncated instructions
            if pc + size > len(bytecode):
                issues.append(TranslationIssue(
                    offset=pc,
                    opcode=op,
                    kind="truncated",
                    message=(
                        f"Truncated instruction 0x{op:02X} at 0x{pc:04X}"
                    ),
                ))
                if self.strict:
                    raise ValueError(f"Truncated instruction at 0x{pc:04X}")
                out.extend(bytecode[pc:pc + size])
                pc += size
                total += 1
                unmapped += 1
                continue

            # Translate opcode
            if op in self._remap:
                new_op = self._remap[op]
                out.append(new_op)
                out.extend(bytecode[pc + 1 : pc + size])
                mapped += 1
            else:
                # Opcode exists in source but has no mapping in target
                src_mn = self.source.mnemonic_for(op)
                issues.append(TranslationIssue(
                    offset=pc,
                    opcode=op,
                    kind="unmapped",
                    message=(
                        f"Opcode 0x{op:02X} ({src_mn}) at 0x{pc:04X} has "
                        f"no equivalent in target dialect '{self.target.name}'"
                    ),
                ))
                if self.strict:
                    raise ValueError(
                        f"Unmapped opcode 0x{op:02X} ({src_mn}) at 0x{pc:04X}"
                    )
                out.append(0xFF)
                out.extend(bytecode[pc + 1 : pc + size])
                unmapped += 1

            pc += size
            total += 1

        return TranslationResult(
            bytecode=bytes(out),
            source_dialect=self.source.name,
            target_dialect=self.target.name,
            total_instructions=total,
            mapped=mapped,
            unmapped=unmapped,
            issues=issues,
        )

    def translate_inplace(self, data: bytearray) -> TranslationResult:
        """Translate bytecode in-place within a mutable bytearray.

        This works because the translation is 1:1 in byte length
        (only the opcode byte changes; operands stay the same size).
        """
        issues: List[TranslationIssue] = []
        pc = 0
        total = 0
        mapped = 0
        unmapped = 0

        while pc < len(data):
            op = data[pc]

            if op not in self.source.opcodes:
                issues.append(TranslationIssue(
                    offset=pc, opcode=op, kind="unknown_opcode",
                    message=f"Unknown opcode 0x{op:02X} at 0x{pc:04X}",
                ))
                if self.strict:
                    raise ValueError(f"Unknown opcode 0x{op:02X} at 0x{pc:04X}")
                pc += 1
                total += 1
                unmapped += 1
                continue

            size = self.source.instruction_size(op)
            if pc + size > len(data):
                issues.append(TranslationIssue(
                    offset=pc, opcode=op, kind="truncated",
                    message=f"Truncated at 0x{pc:04X}",
                ))
                if self.strict:
                    raise ValueError(f"Truncated at 0x{pc:04X}")
                pc += size
                total += 1
                unmapped += 1
                continue

            if op in self._remap:
                data[pc] = self._remap[op]
                mapped += 1
            else:
                src_mn = self.source.mnemonic_for(op)
                issues.append(TranslationIssue(
                    offset=pc, opcode=op, kind="unmapped",
                    message=(
                        f"Unmapped 0x{op:02X} ({src_mn}) at 0x{pc:04X}"
                    ),
                ))
                if self.strict:
                    raise ValueError(
                        f"Unmapped 0x{op:02X} ({src_mn}) at 0x{pc:04X}"
                    )
                data[pc] = 0xFF
                unmapped += 1

            pc += size
            total += 1

        return TranslationResult(
            bytecode=bytes(data),
            source_dialect=self.source.name,
            target_dialect=self.target.name,
            total_instructions=total,
            mapped=mapped,
            unmapped=unmapped,
            issues=issues,
        )


# ─── Conformance Vector ─────────────────────────────────────────────────────

@dataclass
class OpcodeConformance:
    """Conformance detail for a single semantic operation."""
    canonical_op: Op
    source_mnemonic: Optional[str]
    target_mnemonic: Optional[str]
    source_opcode: Optional[int]
    target_opcode: Optional[int]
    portable: bool
    note: str = ""


@dataclass
class ConformanceReport:
    """Aggregate conformance report between two dialects."""
    source_name: str
    target_name: str
    score: float              # 0.0 … 1.0
    total_operations: int
    portable_operations: int
    source_only: List[str]    # mnemonics only in source
    target_only: List[str]    # mnemonics only in target
    opcode_details: List[OpcodeConformance] = field(default_factory=list)

    @property
    def is_fully_compatible(self) -> bool:
        return self.score == 1.0

    @property
    def is_completely_incompatible(self) -> bool:
        return self.score == 0.0

    def summary(self) -> str:
        lines = [
            f"Conformance: {self.source_name} → {self.target_name}",
            f"  Score: {self.score:.2%}",
            f"  Operations: {self.portable_operations}/{self.total_operations} portable",
        ]
        if self.source_only:
            lines.append(f"  Source-only: {', '.join(self.source_only)}")
        if self.target_only:
            lines.append(f"  Target-only: {', '.join(self.target_only)}")
        return "\n".join(lines)


class ConformanceVector:
    """
    Computes how well instructions map across ISA dialects.

    For a single instruction the conformance is binary: the source
    opcode either has an equivalent in the target or it does not.

    For a whole program the *aggregate conformance score* is the
    fraction of instructions that are portable, weighted by frequency.
    """

    def __init__(self, source: ISADialect, target: ISADialect) -> None:
        self.source = source
        self.target = target
        self._translator = ISATranslator(source, target, strict=False)
        self._op_conformance = self._compute_op_conformance()

    # ── Single-operation conformance ──

    def _compute_op_conformance(self) -> Dict[Op, OpcodeConformance]:
        """Compute per-operation conformance details."""
        src_canon = self.source.canonical_ops()
        tgt_canon = self.target.canonical_ops()
        tgt_by_canon: Dict[Op, Tuple[str, int]] = {}
        for opc, mn in self.target.op_to_mnemonic.items():
            cop = self.target._canonical_op(mn)
            if cop is not None:
                tgt_by_canon[cop] = (mn, opc)

        details: Dict[Op, OpcodeConformance] = {}
        all_ops = set(src_canon.values()) | set(tgt_canon.values())

        for op in all_ops:
            src_info: Optional[Tuple[str, int]] = None
            tgt_info: Optional[Tuple[str, int]] = None
            for opc, cop in src_canon.items():
                if cop == op:
                    src_info = (self.source.mnemonic_for(opc) or "", opc)
                    break
            if op in tgt_by_canon:
                tgt_info = tgt_by_canon[op]

            portable = src_info is not None and tgt_info is not None
            note = ""
            if src_info and not tgt_info:
                note = f"No equivalent in target dialect"
            elif tgt_info and not src_info:
                note = f"Only exists in target dialect"
            elif portable:
                same_opcode = src_info[1] == tgt_info[1]  # type: ignore[index]
                if same_opcode:
                    note = "Identical encoding"
                else:
                    note = (
                        f"Remapped: 0x{src_info[1]:02X} → "  # type: ignore[index]
                        f"0x{tgt_info[1]:02X}"  # type: ignore[index]
                    )

            details[op] = OpcodeConformance(
                canonical_op=op,
                source_mnemonic=src_info[0] if src_info else None,
                target_mnemonic=tgt_info[0] if tgt_info else None,
                source_opcode=src_info[1] if src_info else None,
                target_opcode=tgt_info[1] if tgt_info else None,
                portable=portable,
                note=note,
            )
        return details

    def op_conformance(self, op: Op) -> OpcodeConformance:
        """Get conformance details for a canonical operation."""
        return self._op_conformance[op]

    # ── Aggregate report ──

    def report(self) -> ConformanceReport:
        """Generate a full conformance report between source and target."""
        src_mns = set(self.source.mnemonic_set)
        tgt_mns = set(self.target.mnemonic_set)

        # Map to canonical to compare across dialects
        src_canonical_mns: Set[str] = set()
        for opc, mn in self.source.op_to_mnemonic.items():
            cop = self.source._canonical_op(mn)
            if cop:
                src_canonical_mns.add(cop.value)

        tgt_canonical_mns: Set[str] = set()
        for opc, mn in self.target.op_to_mnemonic.items():
            cop = self.target._canonical_op(mn)
            if cop:
                tgt_canonical_mns.add(cop.value)

        # Source-only canonical ops
        src_only = sorted(src_canonical_mns - tgt_canonical_mns)
        # Target-only canonical ops
        tgt_only = sorted(tgt_canonical_mns - src_canonical_mns)

        portable = [
            oc for oc in self._op_conformance.values()
            if oc.portable
        ]
        total_ops = len(self._op_conformance)

        if total_ops == 0:
            score = 1.0
        else:
            # Score: fraction of source ops that have a target equivalent
            source_ops = [
                oc for oc in self._op_conformance.values()
                if oc.source_mnemonic is not None
            ]
            if not source_ops:
                score = 1.0
            else:
                score = sum(1 for oc in source_ops if oc.portable) / len(
                    source_ops
                )

        return ConformanceReport(
            source_name=self.source.name,
            target_name=self.target.name,
            score=score,
            total_operations=total_ops,
            portable_operations=len(portable),
            source_only=src_only,
            target_only=tgt_only,
            opcode_details=list(self._op_conformance.values()),
        )

    # ── Program-level score ──

    def program_score(self, bytecode: bytes) -> float:
        """Compute the conformance score for a specific bytecode program.

        This counts the fraction of instructions in the bytecode that
        have a valid mapping in the target dialect.

        Returns 1.0 for empty bytecode.
        """
        result = self._translator.translate(bytecode)
        return result.coverage


# ─── ISA Registry ───────────────────────────────────────────────────────────

@dataclass
class DialectCompatibility:
    """Pairwise compatibility between two dialects."""
    dialect_a: str
    dialect_b: str
    score: float
    report: ConformanceReport


class ISARegistry:
    """
    Central registry of known ISA dialects.

    Supports dynamic registration, pairwise compatibility computation,
    and fleet-wide conformance analysis.
    """

    def __init__(self) -> None:
        self._dialects: Dict[str, ISADialect] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Pre-register the built-in v1 and v2 dialects."""
        self.register(ISADialect.v1())
        self.register(ISADialect.v2())

    # ── Registration ──

    def register(self, dialect: ISADialect) -> None:
        """Register a dialect. Overwrites if the name already exists."""
        self._dialects[dialect.name] = dialect

    def unregister(self, name: str) -> bool:
        """Remove a dialect by name. Returns True if it existed."""
        if name in self._dialects:
            del self._dialects[name]
            return True
        return False

    def get(self, name: str) -> Optional[ISADialect]:
        """Retrieve a dialect by name, or None."""
        return self._dialects.get(name)

    @property
    def dialect_names(self) -> List[str]:
        """List all registered dialect names."""
        return sorted(self._dialects.keys())

    @property
    def dialects(self) -> Dict[str, ISADialect]:
        """Read-only view of all registered dialects."""
        return dict(self._dialects)

    def __len__(self) -> int:
        return len(self._dialects)

    def __contains__(self, name: str) -> bool:
        return name in self._dialects

    # ── Pairwise Compatibility ──

    def compatibility(
        self,
        name_a: str,
        name_b: str,
    ) -> DialectCompatibility:
        """Compute pairwise compatibility between two registered dialects."""
        da = self._dialects.get(name_a)
        db = self._dialects.get(name_b)
        if da is None:
            raise KeyError(f"Dialect '{name_a}' not registered")
        if db is None:
            raise KeyError(f"Dialect '{name_b}' not registered")
        cv = ConformanceVector(da, db)
        rpt = cv.report()
        return DialectCompatibility(
            dialect_a=name_a,
            dialect_b=name_b,
            score=rpt.score,
            report=rpt,
        )

    def compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Compute a full NxN compatibility matrix.

        Returns a nested dict: matrix[a][b] = score of translating
        dialect a → dialect b.
        """
        names = self.dialect_names
        matrix: Dict[str, Dict[str, float]] = {}
        for a in names:
            matrix[a] = {}
            for b in names:
                if a == b:
                    matrix[a][b] = 1.0
                else:
                    matrix[a][b] = self.compatibility(a, b).score
        return matrix

    # ── Fleet-wide Conformance ──

    def max_compatible_subset(
        self,
        threshold: float = 1.0,
    ) -> List[Set[str]]:
        """
        Find the maximum subset(s) of dialects that are mutually
        compatible at or above *threshold*.

        Uses a brute-force approach (suitable for small registries).
        Returns a list of compatible sets, sorted largest first.
        """
        names = self.dialect_names
        if not names:
            return []

        matrix = self.compatibility_matrix()
        best_sets: List[Set[str]] = []
        best_size = 0

        # Check all subsets (2^n). For n ≤ 20 this is feasible.
        n = len(names)
        for mask in range(1, 1 << n):
            subset = {names[i] for i in range(n) if mask & (1 << i)}
            if len(subset) < best_size:
                continue
            # Check pairwise compatibility within the subset
            # Must check BOTH directions for true mutual compatibility
            compatible = True
            members = sorted(subset)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    score_ab = matrix[members[i]][members[j]]
                    score_ba = matrix[members[j]][members[i]]
                    if score_ab < threshold or score_ba < threshold:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                if len(subset) > best_size:
                    best_size = len(subset)
                    best_sets = [subset]
                elif len(subset) == best_size:
                    best_sets.append(subset)

        # Deduplicate and sort
        unique: List[Set[str]] = []
        seen: List[str] = []
        for s in sorted(best_sets, key=lambda x: (-len(x), sorted(x))):
            key = ",".join(sorted(s))
            if key not in seen:
                seen.append(key)
                unique.append(s)
        return unique

    def fleet_report(self, bytecode: bytes) -> Dict[str, float]:
        """
        Given a bytecode program, compute the conformance score against
        every registered dialect.

        Returns a dict: dialect_name → program_score.
        """
        scores: Dict[str, float] = {}
        for name, dialect in self._dialects.items():
            # Validate against each dialect and compute coverage
            issues = dialect.validate_bytecode(bytecode)
            if not issues:
                # Fully conformant
                scores[name] = 1.0
            else:
                # Count how many instruction positions have valid opcodes
                total, valid = self._count_valid_instructions(
                    bytecode, dialect
                )
                scores[name] = valid / total if total > 0 else 1.0
        return scores

    @staticmethod
    def _count_valid_instructions(
        bytecode: bytes,
        dialect: ISADialect,
    ) -> Tuple[int, int]:
        """Count total and valid instruction positions."""
        total = 0
        valid = 0
        pc = 0
        while pc < len(bytecode):
            op = bytecode[pc]
            total += 1
            if op in dialect.opcodes:
                valid += 1
                try:
                    pc += dialect.instruction_size(op)
                except ValueError:
                    pc += 1
            else:
                pc += 1
        return total, valid


# ─── Helper: Build a Translator from names ─────────────────────────────────

def build_translator(
    registry: ISARegistry,
    source_name: str,
    target_name: str,
    **kwargs,
) -> ISATranslator:
    """Convenience: build an ISATranslator from registry lookups."""
    src = registry.get(source_name)
    tgt = registry.get(target_name)
    if src is None:
        raise KeyError(f"Dialect '{source_name}' not in registry")
    if tgt is None:
        raise KeyError(f"Dialect '{target_name}' not in registry")
    return ISATranslator(src, tgt, **kwargs)


# ─── Quick Demo ─────────────────────────────────────────────────────────────

def demo():
    """Quick demonstration of the ISA conformance layer."""
    print("=" * 60)
    print("FLUX ISA Conformance Layer — Demo")
    print("=" * 60)

    # Create dialects
    v1 = ISADialect.v1()
    v2 = ISADialect.v2()

    print(f"\nv1 opcodes: {v1.to_json()}")
    print(f"\nv2 opcodes: {v2.to_json()}")

    # Conformance report
    cv = ConformanceVector(v1, v2)
    report = cv.report()
    print(f"\n{report.summary()}")
    for oc in report.opcode_details:
        marker = "✓" if oc.portable else "✗"
        src_s = f"0x{oc.source_opcode:02X}" if oc.source_opcode else "—"
        tgt_s = f"0x{oc.target_opcode:02X}" if oc.target_opcode else "—"
        print(
            f"  {marker} {oc.canonical_op.value:6s} "
            f"{oc.source_mnemonic or '—':6s} ({src_s}) → "
            f"{oc.target_mnemonic or '—':6s} ({tgt_s})  "
            f"[{oc.note}]"
        )

    # Build a sample program and translate
    import struct as _s
    program = bytearray()
    # MOVI R0, 5
    program.extend([0x2B, 0x00])  # MOVI R0
    program.extend(_s.pack("<h", 5))
    # MOVI R1, 3
    program.extend([0x2B, 0x01])  # MOVI R1
    program.extend(_s.pack("<h", 3))
    # IADD R0, R0, R1
    program.extend([0x08, 0x00, 0x00, 0x01])
    # HALT
    program.extend([0x80])

    bc = bytes(program)
    print(f"\nSample v1 bytecode ({len(bc)} bytes):")
    print(f"  {' '.join(f'{b:02X}' for b in bc)}")

    # Translate v1 → v2
    translator = ISATranslator(v1, v2)
    result = translator.translate(bc)
    print(f"\n{result.summary()}")
    translated = result.bytecode
    print(f"  Translated: {' '.join(f'{b:02X}' for b in translated)}")

    # Translate back v2 → v1
    back_translator = ISATranslator(v2, v1)
    back_result = back_translator.translate(translated)
    print(f"\n{back_result.summary()}")
    print(f"  Round-trip: {' '.join(f'{b:02X}' for b in back_result.bytecode)}")
    print(
        f"  Fidelity: {'PASS ✓' if back_result.bytecode == bc else 'FAIL ✗'}"
    )

    # Registry demo
    print("\n" + "-" * 60)
    reg = ISARegistry()
    print(f"\nRegistry dialects: {reg.dialect_names}")

    matrix = reg.compatibility_matrix()
    print("\nCompatibility matrix:")
    for a in reg.dialect_names:
        for b in reg.dialect_names:
            print(f"  {a} → {b}: {matrix[a][b]:.2%}")

    # Register a custom dialect
    v3 = ISADialect.custom("v3-extended", 3, {
        "ADD": 0x20, "SUB": 0x21, "MUL": 0x22, "DIV": 0x23,
        "MOVI": 0x30, "DEC": 0x31, "JNZ": 0x40, "HALT": 0x00,
        "NOP": 0x01, "SHL": 0x24,
    })
    reg.register(v3)
    print(f"\nAfter registering v3: {reg.dialect_names}")

    compat_v2_v3 = reg.compatibility("v2", "v3-extended")
    print(
        f"\nv2 → v3-extended: {compat_v2_v3.score:.2%}"
    )
    print(f"  {compat_v2_v3.report.summary()}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
