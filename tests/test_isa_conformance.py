"""
Comprehensive tests for the FLUX ISA Conformance Layer.

Covers:
  - Dialect creation and registration
  - v1↔v2 translation and round-trip fidelity
  - Conformance vector computation
  - Unknown opcode handling
  - Edge cases (empty bytecode, single instruction, large programs)
  - Custom dialect support
  - Registry operations and compatibility matrix
  - In-place binary patching
  - Fleet-wide conformance reports
  - Max-compatible subset search
"""

import struct
import sys
import os
import unittest

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isa_conformance import (
    ISADialect,
    ISATranslator,
    ConformanceVector,
    ConformanceReport,
    ISARegistry,
    DialectCompatibility,
    InsnFormat,
    Op,
    TranslationIssue,
    TranslationResult,
    build_translator,
    OP_FORMAT,
)


# ─── Test Helpers ───────────────────────────────────────────────────────────

def make_program_v1(*instructions) -> bytes:
    """Build v1 bytecode from (mnemonic, *operands) tuples."""
    v1 = ISADialect.v1()
    buf = bytearray()
    for item in instructions:
        if isinstance(item, (list, tuple)):
            mn = item[0]
            operands = item[1:]
        else:
            mn = item
            operands = []

        op = v1.opcode_for(mn)
        assert op is not None, f"Unknown mnemonic: {mn}"
        buf.append(op)

        if mn == "HALT":
            pass
        elif mn == "DEC":
            buf.append(operands[0])
        elif mn in ("IADD", "IMUL", "IDIV"):
            buf.append(operands[0])  # rd
            buf.append(operands[1])  # ra
            buf.append(operands[2])  # rb
        elif mn == "MOVI":
            buf.append(operands[0])  # rd
            buf.extend(struct.pack("<h", operands[1]))  # i16
        elif mn == "JNZ":
            buf.append(operands[0])  # rd
            buf.extend(struct.pack("<h", operands[1]))  # offset
    return bytes(buf)


def make_program_v2(*instructions) -> bytes:
    """Build v2 bytecode from (mnemonic, *operands) tuples."""
    v2 = ISADialect.v2()
    buf = bytearray()
    for item in instructions:
        if isinstance(item, (list, tuple)):
            mn = item[0]
            operands = item[1:]
        else:
            mn = item
            operands = []

        op = v2.opcode_for(mn)
        assert op is not None, f"Unknown mnemonic: {mn}"
        buf.append(op)

        if mn == "HALT":
            pass
        elif mn == "DEC":
            buf.append(operands[0])
        elif mn in ("ADD", "SUB", "MUL", "DIV"):
            buf.append(operands[0])
            buf.append(operands[1])
            buf.append(operands[2])
        elif mn == "MOVI":
            buf.append(operands[0])
            buf.extend(struct.pack("<h", operands[1]))
        elif mn == "JNZ":
            buf.append(operands[0])
            buf.extend(struct.pack("<h", operands[1]))
    return bytes(buf)


# ─── Test Cases ─────────────────────────────────────────────────────────────


class TestISADialect(unittest.TestCase):
    """Tests for ISADialect creation and properties."""

    def test_v1_factory(self):
        d = ISADialect.v1()
        self.assertEqual(d.name, "v1")
        self.assertEqual(d.version, 1)
        self.assertEqual(d.opcode_for("MOVI"), 0x2B)
        self.assertEqual(d.opcode_for("IADD"), 0x08)
        self.assertEqual(d.opcode_for("HALT"), 0x80)
        self.assertEqual(len(d.mnemonic_to_op), 7)

    def test_v2_factory(self):
        d = ISADialect.v2()
        self.assertEqual(d.name, "v2")
        self.assertEqual(d.version, 2)
        self.assertEqual(d.opcode_for("ADD"), 0x20)
        self.assertEqual(d.opcode_for("MOVI"), 0x30)
        self.assertEqual(d.opcode_for("HALT"), 0x00)
        self.assertEqual(len(d.mnemonic_to_op), 8)  # includes SUB

    def test_opcodes_set(self):
        d = ISADialect.v1()
        self.assertIn(0x2B, d.opcodes)
        self.assertIn(0x80, d.opcodes)
        self.assertNotIn(0x20, d.opcodes)  # v2 ADD

    def test_mnemonic_for(self):
        d = ISADialect.v1()
        self.assertEqual(d.mnemonic_for(0x2B), "MOVI")
        self.assertEqual(d.mnemonic_for(0x80), "HALT")
        self.assertIsNone(d.mnemonic_for(0xFF))

    def test_instruction_size(self):
        d = ISADialect.v1()
        self.assertEqual(d.instruction_size(0x80), 1)  # HALT
        self.assertEqual(d.instruction_size(0x0F), 2)  # DEC
        self.assertEqual(d.instruction_size(0x08), 4)  # IADD
        self.assertEqual(d.instruction_size(0x2B), 4)  # MOVI

    def test_instruction_size_unknown(self):
        d = ISADialect.v1()
        with self.assertRaises(ValueError):
            d.instruction_size(0xFF)

    def test_custom_dialect(self):
        d = ISADialect.custom("v3", 3, {
            "HALT": 0x00, "ADD": 0x10, "MOVI": 0x20,
        })
        self.assertEqual(d.name, "v3")
        self.assertEqual(d.version, 3)
        self.assertEqual(d.opcode_for("ADD"), 0x10)

    def test_to_dict_and_json(self):
        d = ISADialect.v1()
        dct = d.to_dict()
        self.assertEqual(dct["name"], "v1")
        self.assertEqual(dct["version"], 1)
        self.assertIn("MOVI", dct["mnemonic_to_op"])

        json_str = d.to_json()
        self.assertIn("v1", json_str)
        self.assertIn("MOVI", json_str)

    def test_v1_v2_different_opcodes(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        # HALT is different
        self.assertNotEqual(v1.opcode_for("HALT"), v2.opcode_for("HALT"))
        # MOVI is different
        self.assertNotEqual(v1.opcode_for("MOVI"), v2.opcode_for("MOVI"))
        # v1 has IADD, v2 has ADD
        self.assertIsNotNone(v1.opcode_for("IADD"))
        self.assertIsNone(v2.opcode_for("IADD"))
        self.assertIsNotNone(v2.opcode_for("ADD"))
        self.assertIsNone(v1.opcode_for("ADD"))

    def test_validate_valid_bytecode(self):
        d = ISADialect.v1()
        bc = make_program_v1(
            ("MOVI", 0, 5), ("IADD", 0, 0, 1), ("HALT",)
        )
        issues = d.validate_bytecode(bc)
        self.assertEqual(len(issues), 0)

    def test_validate_unknown_opcode(self):
        d = ISADialect.v1()
        bc = bytes([0xFF, 0x80])  # unknown, then HALT
        issues = d.validate_bytecode(bc)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "unknown_opcode")
        self.assertEqual(issues[0].opcode, 0xFF)

    def test_validate_truncated(self):
        d = ISADialect.v1()
        bc = bytes([0x2B, 0x00])  # MOVI needs 4 bytes, only 2
        issues = d.validate_bytecode(bc)
        self.assertTrue(any(i.kind == "truncated" for i in issues))


class TestISATranslation(unittest.TestCase):
    """Tests for v1↔v2 bytecode translation."""

    def test_v1_to_v2_simple(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("MOVI", 0, 5), ("HALT",))
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.total_instructions, 2)
        self.assertEqual(result.mapped, 2)
        self.assertEqual(result.unmapped, 0)

    def test_v1_to_v2_opcode_values(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("MOVI", 0, 5), ("HALT",))
        result = t.translate(bc)
        out = result.bytecode
        self.assertEqual(out[0], 0x30)   # v2 MOVI
        self.assertEqual(out[1], 0x00)   # R0
        self.assertEqual(out[4], 0x00)   # v2 HALT

    def test_v2_to_v1_simple(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v2, v1)
        bc = make_program_v2(("MOVI", 1, 10), ("ADD", 1, 1, 0), ("HALT",))
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.total_instructions, 3)

    def test_v2_to_v1_opcode_values(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v2, v1)
        bc = make_program_v2(("MOVI", 1, 10), ("HALT",))
        result = t.translate(bc)
        out = result.bytecode
        self.assertEqual(out[0], 0x2B)   # v1 MOVI
        self.assertEqual(out[4], 0x80)   # v1 HALT

    def test_round_trip_v1_v2_v1(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t_fwd = ISATranslator(v1, v2)
        t_back = ISATranslator(v2, v1)

        original = make_program_v1(
            ("MOVI", 0, 42),
            ("MOVI", 1, 7),
            ("IADD", 0, 0, 1),
            ("IMUL", 2, 0, 0),
            ("DEC", 2),
            ("JNZ", 0, -12),
            ("HALT",),
        )

        translated = t_fwd.translate(original).bytecode
        roundtrip = t_back.translate(translated).bytecode
        self.assertEqual(roundtrip, original, "Round-trip should be lossless")

    def test_round_trip_v2_v1_v2(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t_fwd = ISATranslator(v2, v1)
        t_back = ISATranslator(v1, v2)

        original = make_program_v2(
            ("MOVI", 0, 100),
            ("ADD", 0, 0, 0),
            ("MUL", 2, 1, 1),
            ("DIV", 3, 2, 2),
            ("DEC", 3),
            ("JNZ", 3, -6),
            ("HALT",),
        )

        translated = t_fwd.translate(original).bytecode
        roundtrip = t_back.translate(translated).bytecode
        self.assertEqual(roundtrip, original)

    def test_operands_preserved(self):
        """Operands must survive translation unchanged."""
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        # MOVI R5, -100  →  bytes: 2B 05 9C FF (v1)
        bc = make_program_v1(("MOVI", 5, -100))
        result = t.translate(bc)
        out = result.bytecode
        self.assertEqual(out[0], 0x30)   # v2 MOVI
        self.assertEqual(out[1], 0x05)   # R5 preserved
        val = struct.unpack("<h", out[2:4])[0]
        self.assertEqual(val, -100)       # immediate preserved

    def test_unknown_opcode_strict(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2, strict=True)
        bc = bytes([0xFF, 0x80])
        with self.assertRaises(ValueError):
            t.translate(bc)

    def test_unknown_opcode_nonstrict(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2, strict=False)
        bc = bytes([0xFF, 0x80])
        result = t.translate(bc)
        self.assertFalse(result.success)
        self.assertEqual(result.unmapped, 1)
        self.assertEqual(result.bytecode[0], 0xFF)  # placeholder

    def test_unmapped_opcode_strict(self):
        """Opcode in v2 with no v1 equivalent (SUB=0x21)."""
        v2, v1 = ISADialect.v2(), ISADialect.v1()
        t = ISATranslator(v2, v1, strict=True)
        # SUB R0, R0, R0 — SUB doesn't exist in v1
        bc = make_program_v2(("SUB", 0, 0, 0), ("HALT",))
        with self.assertRaises(ValueError):
            t.translate(bc)

    def test_unmapped_opcode_nonstrict(self):
        v2, v1 = ISADialect.v2(), ISADialect.v1()
        t = ISATranslator(v2, v1, strict=False)
        bc = make_program_v2(("SUB", 0, 0, 0), ("HALT",))
        result = t.translate(bc)
        self.assertEqual(result.unmapped, 1)
        self.assertEqual(result.bytecode[0], 0xFF)  # placeholder for SUB
        self.assertEqual(result.bytecode[4], 0x80)  # HALT translated fine

    def test_empty_bytecode(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        result = t.translate(b"")
        self.assertTrue(result.success)
        self.assertEqual(result.total_instructions, 0)
        self.assertEqual(result.bytecode, b"")

    def test_single_instruction(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = bytes([0x80])  # HALT
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.bytecode, bytes([0x00]))  # v2 HALT

    def test_large_program(self):
        """Translate a program with 1000 MOVI instructions."""
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        instructions = [("MOVI", i % 16, i) for i in range(1000)]
        instructions.append(("HALT",))
        bc = make_program_v1(*instructions)
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.total_instructions, 1001)
        self.assertEqual(len(result.bytecode), len(bc))

    def test_mapping_table(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        table = t.mapping_table()
        self.assertEqual(table[0x2B], 0x30)  # MOVI
        self.assertEqual(table[0x08], 0x20)  # IADD → ADD
        self.assertEqual(table[0x80], 0x00)  # HALT
        self.assertEqual(table[0x0F], 0x31)  # DEC

    def test_unmapped_opcodes_report(self):
        v2, v1 = ISADialect.v2(), ISADialect.v1()
        t = ISATranslator(v2, v1)
        unmapped = t.unmapped_opcodes()
        self.assertIn(0x21, unmapped)  # SUB — no v1 equivalent

    def test_validate_source(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = bytes([0xFF])
        issues = t.validate_source(bc)
        self.assertEqual(len(issues), 1)

    def test_translation_result_summary(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("MOVI", 0, 1), ("HALT",))
        result = t.translate(bc)
        summary = result.summary()
        self.assertIn("v1", summary)
        self.assertIn("v2", summary)
        self.assertIn("100.0%", summary)


class TestInPlaceTranslation(unittest.TestCase):
    """Tests for in-place bytecode patching."""

    def test_inplace_basic(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = bytearray(make_program_v1(("MOVI", 0, 5), ("HALT",)))
        original = bytes(bc)
        result = t.translate_inplace(bc)
        self.assertTrue(result.success)
        self.assertEqual(bc[0], 0x30)  # now v2 MOVI
        self.assertEqual(bc[4], 0x00)  # now v2 HALT

    def test_inplace_preserves_length(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        original = make_program_v1(
            ("MOVI", 0, 1), ("IADD", 0, 0, 1), ("DEC", 0),
            ("JNZ", 0, -6), ("HALT",),
        )
        bc = bytearray(original)
        t.translate_inplace(bc)
        self.assertEqual(len(bc), len(original))

    def test_inplace_roundtrip(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        original = make_program_v1(
            ("MOVI", 0, 10), ("IADD", 0, 0, 0), ("HALT",)
        )
        bc = bytearray(original)
        ISATranslator(v1, v2).translate_inplace(bc)
        ISATranslator(v2, v1).translate_inplace(bc)
        self.assertEqual(bytes(bc), original)


class TestConformanceVector(unittest.TestCase):
    """Tests for conformance vector computation."""

    def test_v1_v2_report(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        report = cv.report()
        self.assertEqual(report.source_name, "v1")
        self.assertEqual(report.target_name, "v2")
        # v1 ops should all map to v2 (except SUB which is v2-only)
        self.assertGreater(report.score, 0.5)

    def test_v1_v2_all_v1_ops_portable(self):
        """Every v1 opcode should have a v2 equivalent."""
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        report = cv.report()
        # Source-only should be empty (v1 ops all exist in v2)
        self.assertEqual(len(report.source_only), 0)

    def test_v2_has_sub_only(self):
        """v2 has SUB which v1 doesn't have."""
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        report = cv.report()
        self.assertIn("SUB", report.target_only)

    def test_identical_dialect_score(self):
        v1 = ISADialect.v1()
        cv = ConformanceVector(v1, v1)
        report = cv.report()
        self.assertEqual(report.score, 1.0)
        self.assertTrue(report.is_fully_compatible)

    def test_op_conformance_details(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        halt_oc = cv.op_conformance(Op.HALT)
        self.assertTrue(halt_oc.portable)
        self.assertEqual(halt_oc.source_opcode, 0x80)
        self.assertEqual(halt_oc.target_opcode, 0x00)

    def test_program_score_full(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        bc = make_program_v1(("MOVI", 0, 1), ("HALT",))
        score = cv.program_score(bc)
        self.assertEqual(score, 1.0)

    def test_program_score_partial(self):
        v2, v1 = ISADialect.v2(), ISADialect.v1()
        cv = ConformanceVector(v2, v1)
        # SUB has no v1 equivalent
        bc = make_program_v2(("SUB", 0, 0, 0), ("HALT",))
        score = cv.program_score(bc)
        self.assertLess(score, 1.0)

    def test_program_score_empty(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        self.assertEqual(cv.program_score(b""), 1.0)

    def test_report_summary(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        cv = ConformanceVector(v1, v2)
        report = cv.report()
        summary = report.summary()
        self.assertIn("v1", summary)
        self.assertIn("v2", summary)
        self.assertIn("Score:", summary)


class TestCustomDialect(unittest.TestCase):
    """Tests for custom dialect creation and translation."""

    def test_custom_dialect_creation(self):
        d = ISADialect.custom("micro", 1, {
            "HALT": 0x00, "NOOP": 0x01, "LOAD": 0x10,
        })
        self.assertEqual(d.name, "micro")
        self.assertEqual(d.opcode_for("NOOP"), 0x01)

    def test_custom_dialect_translation_subset(self):
        """Translate from v1 to a subset dialect (fewer opcodes)."""
        subset = ISADialect.custom("subset", 1, {
            "HALT": 0x00, "MOVI": 0x30,
        })
        v1 = ISADialect.v1()
        t = ISATranslator(v1, subset, strict=False)
        bc = make_program_v1(("MOVI", 0, 1), ("IADD", 0, 0, 1), ("HALT",))
        result = t.translate(bc)
        self.assertEqual(result.mapped, 2)  # MOVI and HALT
        self.assertEqual(result.unmapped, 1)  # IADD

    def test_custom_dialect_extended(self):
        """Translate from an extended dialect back to v1."""
        extended = ISADialect.custom("extended", 3, {
            "HALT": 0x00, "MOVI": 0x30, "ADD": 0x20,
            "MUL": 0x22, "DEC": 0x31, "JNZ": 0x40,
            "SHL": 0x50, "XOR": 0x51,
        })
        v1 = ISADialect.v1()
        # Forward: v1 → extended (should work, no SHL/XOR used)
        t = ISATranslator(v1, extended)
        bc = make_program_v1(("MOVI", 0, 5), ("IMUL", 0, 0, 0), ("HALT",))
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.mapped, 3)

        # Back: extended → v1 (SHL/XOR have no v1 equivalent)
        t_back = ISATranslator(extended, v1, strict=False)
        # Build bytecode for the extended dialect
        ext_bc = bytearray()
        ext_bc.extend([0x30, 0x00])  # MOVI R0
        ext_bc.extend(struct.pack("<h", 5))
        ext_bc.extend([0x50, 0x00, 0x01, 0x02])  # SHL R0, R1, R2
        ext_bc.extend([0x00])  # HALT
        result = t_back.translate(bytes(ext_bc))
        self.assertEqual(result.unmapped, 1)  # SHL

    def test_custom_empty_mapping(self):
        """Custom dialect with no opcodes."""
        d = ISADialect.custom("empty", 0, {})
        self.assertEqual(len(d.opcodes), 0)
        with self.assertRaises(ValueError):
            d.instruction_size(0x00)  # unknown in empty dialect


class TestISARegistry(unittest.TestCase):
    """Tests for the ISA registry."""

    def test_builtin_dialects(self):
        reg = ISARegistry()
        self.assertIn("v1", reg)
        self.assertIn("v2", reg)
        self.assertEqual(len(reg), 2)

    def test_register_custom(self):
        reg = ISARegistry()
        d = ISADialect.custom("v3", 3, {"HALT": 0x00})
        reg.register(d)
        self.assertIn("v3", reg)
        self.assertEqual(len(reg), 3)

    def test_unregister(self):
        reg = ISARegistry()
        self.assertTrue(reg.unregister("v1"))
        self.assertNotIn("v1", reg)
        self.assertFalse(reg.unregister("nonexistent"))

    def test_get_dialect(self):
        reg = ISARegistry()
        v1 = reg.get("v1")
        self.assertIsNotNone(v1)
        self.assertEqual(v1.name, "v1")
        self.assertIsNone(reg.get("nonexistent"))

    def test_dialect_names_sorted(self):
        reg = ISARegistry()
        names = reg.dialect_names
        self.assertEqual(names, ["v1", "v2"])

    def test_compatibility_pairwise(self):
        reg = ISARegistry()
        comp = reg.compatibility("v1", "v2")
        self.assertIsInstance(comp, DialectCompatibility)
        self.assertEqual(comp.dialect_a, "v1")
        self.assertEqual(comp.dialect_b, "v2")
        self.assertGreater(comp.score, 0.5)

    def test_compatibility_self(self):
        reg = ISARegistry()
        comp = reg.compatibility("v1", "v1")
        self.assertEqual(comp.score, 1.0)

    def test_compatibility_unknown(self):
        reg = ISARegistry()
        with self.assertRaises(KeyError):
            reg.compatibility("v1", "v99")

    def test_compatibility_matrix(self):
        reg = ISARegistry()
        matrix = reg.compatibility_matrix()
        self.assertIn("v1", matrix)
        self.assertIn("v2", matrix)
        self.assertEqual(matrix["v1"]["v1"], 1.0)
        self.assertEqual(matrix["v2"]["v2"], 1.0)

    def test_fleet_report(self):
        reg = ISARegistry()
        # Build v1 bytecode
        bc = make_program_v1(("MOVI", 0, 1), ("HALT",))
        report = reg.fleet_report(bc)
        self.assertIn("v1", report)
        self.assertEqual(report["v1"], 1.0)  # v1 bytecode, v1 dialect → perfect
        # v2 shouldn't be perfect because opcodes differ
        self.assertLess(report["v2"], 1.0)

    def test_max_compatible_subset_full(self):
        """v1 and v2 are NOT fully compatible (different opcodes)."""
        reg = ISARegistry()
        subsets = reg.max_compatible_subset(threshold=1.0)
        # Each dialect is compatible with itself
        for s in subsets:
            self.assertLessEqual(len(s), 2)
        # There should be at least one subset (each individual dialect)
        self.assertTrue(len(subsets) >= 2)

    def test_max_compatible_subset_relaxed(self):
        """With a relaxed threshold, all dialects should be in one set."""
        reg = ISARegistry()
        subsets = reg.max_compatible_subset(threshold=0.5)
        # At threshold=0.5, v1↔v2 should be in the same set
        self.assertTrue(len(subsets) >= 1)
        # The largest set should contain both v1 and v2
        largest = subsets[0]
        self.assertIn("v1", largest)
        self.assertIn("v2", largest)

    def test_build_translator_helper(self):
        reg = ISARegistry()
        t = build_translator(reg, "v1", "v2")
        self.assertIsInstance(t, ISATranslator)
        bc = make_program_v1(("HALT",))
        result = t.translate(bc)
        self.assertTrue(result.success)

    def test_build_translator_missing(self):
        reg = ISARegistry()
        with self.assertRaises(KeyError):
            build_translator(reg, "v1", "v99")


class TestTranslationResult(unittest.TestCase):
    """Tests for TranslationResult dataclass."""

    def test_coverage(self):
        r = TranslationResult(
            bytecode=b"", source_dialect="a", target_dialect="b",
            total_instructions=10, mapped=7, unmapped=3,
        )
        self.assertAlmostEqual(r.coverage, 0.7)

    def test_coverage_zero(self):
        r = TranslationResult(
            bytecode=b"", source_dialect="a", target_dialect="b",
            total_instructions=0, mapped=0, unmapped=0,
        )
        self.assertAlmostEqual(r.coverage, 1.0)

    def test_success_flag(self):
        r_ok = TranslationResult(
            bytecode=b"", source_dialect="a", target_dialect="b",
            total_instructions=5, mapped=5, unmapped=0,
        )
        self.assertTrue(r_ok.success)

        r_fail = TranslationResult(
            bytecode=b"", source_dialect="a", target_dialect="b",
            total_instructions=5, mapped=4, unmapped=1,
        )
        self.assertFalse(r_fail.success)

    def test_summary_format(self):
        r = TranslationResult(
            bytecode=b"\x00", source_dialect="v1", target_dialect="v2",
            total_instructions=3, mapped=2, unmapped=1,
            issues=[TranslationIssue(4, 0xFF, "test", "test issue")],
        )
        s = r.summary()
        self.assertIn("v1", s)
        self.assertIn("v2", s)
        self.assertIn("66.7%", s)


class TestEnumConstants(unittest.TestCase):
    """Verify enum and constant definitions."""

    def test_op_format_mapping(self):
        self.assertEqual(OP_FORMAT[Op.HALT], InsnFormat.NOP)
        self.assertEqual(OP_FORMAT[Op.DEC], InsnFormat.R1)
        self.assertEqual(OP_FORMAT[Op.ADD], InsnFormat.R3)
        self.assertEqual(OP_FORMAT[Op.MUL], InsnFormat.R3)
        self.assertEqual(OP_FORMAT[Op.MOVI], InsnFormat.RI16)
        self.assertEqual(OP_FORMAT[Op.JNZ], InsnFormat.RI16)

    def test_insn_format_values(self):
        self.assertEqual(InsnFormat.NOP.value, 1)
        self.assertEqual(InsnFormat.R1.value, 2)
        self.assertEqual(InsnFormat.R3.value, 4)
        self.assertEqual(InsnFormat.RI16.value, 4)

    def test_op_enum_members(self):
        self.assertEqual(Op.HALT.value, "HALT")
        self.assertEqual(Op.ADD.value, "ADD")
        self.assertEqual(Op.SUB.value, "SUB")
        self.assertEqual(len(Op), 8)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_halt_only_program(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = bytes([0x80])
        result = t.translate(bc)
        self.assertEqual(result.bytecode, bytes([0x00]))

    def test_many_dec_instructions(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(*[("DEC", 0) for _ in range(50)])
        bc += bytes([0x80])
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.total_instructions, 51)
        self.assertEqual(result.bytecode[-1], 0x00)  # HALT

    def test_negative_immediate(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("MOVI", 0, -32768))
        result = t.translate(bc)
        val = struct.unpack("<h", result.bytecode[2:4])[0]
        self.assertEqual(val, -32768)

    def test_large_positive_immediate(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("MOVI", 0, 32767))
        result = t.translate(bc)
        val = struct.unpack("<h", result.bytecode[2:4])[0]
        self.assertEqual(val, 32767)

    def test_jnz_offset_preserved(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(("JNZ", 5, -20))
        result = t.translate(bc)
        val = struct.unpack("<h", result.bytecode[2:4])[0]
        self.assertEqual(val, -20)
        self.assertEqual(result.bytecode[1], 5)  # register preserved

    def test_all_v1_opcodes_in_single_program(self):
        """Program that uses every v1 opcode."""
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2)
        bc = make_program_v1(
            ("HALT",),
            ("DEC", 0),
            ("IADD", 0, 0, 1),
            ("IMUL", 0, 0, 1),
            ("IDIV", 0, 0, 1),
            ("MOVI", 0, 42),
            ("JNZ", 0, 0),
        )
        result = t.translate(bc)
        self.assertTrue(result.success)
        self.assertEqual(result.mapped, 7)

    def test_truncated_instruction_strict(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2, strict=True)
        # MOVI needs 4 bytes, provide only 2
        bc = bytes([0x2B, 0x00])
        with self.assertRaises(ValueError):
            t.translate(bc)

    def test_truncated_instruction_nonstrict(self):
        v1, v2 = ISADialect.v1(), ISADialect.v2()
        t = ISATranslator(v1, v2, strict=False)
        bc = bytes([0x2B, 0x00])
        result = t.translate(bc)
        self.assertFalse(result.success)
        self.assertTrue(any(i.kind == "truncated" for i in result.issues))


class TestRegistryWithCustomDialects(unittest.TestCase):
    """Registry operations with custom dialects registered."""

    def setUp(self):
        self.reg = ISARegistry()
        self.reg.register(ISADialect.custom("v3-micro", 3, {
            "HALT": 0x00, "ADD": 0x20, "MOVI": 0x30,
        }))
        self.reg.register(ISADialect.custom("v4-full", 4, {
            "HALT": 0x00, "ADD": 0x20, "SUB": 0x21,
            "MUL": 0x22, "DIV": 0x23, "MOVI": 0x30,
            "DEC": 0x31, "JNZ": 0x40, "SHL": 0x24,
            "XOR": 0x25,
        }))

    def test_four_dialects_registered(self):
        self.assertEqual(len(self.reg), 4)
        self.assertIn("v3-micro", self.reg)

    def test_v3_micro_compatibility(self):
        comp = self.reg.compatibility("v1", "v3-micro")
        # v3-micro only has HALT, ADD, MOVI — v1 has more
        self.assertGreater(comp.score, 0.0)
        self.assertLess(comp.score, 1.0)

    def test_v4_full_compatibility(self):
        comp = self.reg.compatibility("v2", "v4-full")
        # v4-full has everything v2 has plus SHL/XOR
        # All v2 ops should map to v4-full
        self.assertEqual(comp.score, 1.0)

    def test_full_matrix(self):
        matrix = self.reg.compatibility_matrix()
        self.assertEqual(len(matrix), 4)
        for a in matrix:
            self.assertEqual(len(matrix[a]), 4)

    def test_v4_full_to_v1_unmapped(self):
        """v4-full has SHL/XOR that don't exist in v1."""
        v4 = self.reg.get("v4-full")
        v1 = self.reg.get("v1")
        t = ISATranslator(v4, v1, strict=False)
        # Build v4-full bytecode with SHL
        bc = bytearray()
        bc.extend([0x24, 0x00, 0x01, 0x02])  # SHL R0, R1, R2
        bc.extend([0x00])  # HALT
        result = t.translate(bytes(bc))
        self.assertEqual(result.unmapped, 1)


class TestCanonicalOps(unittest.TestCase):
    """Test canonical operation mapping."""

    def test_v1_canonical_ops(self):
        v1 = ISADialect.v1()
        canon = v1.canonical_ops()
        # IADD should map to Op.ADD
        self.assertEqual(canon[0x08], Op.ADD)
        # IMUL should map to Op.MUL
        self.assertEqual(canon[0x0A], Op.MUL)
        # HALT should map to Op.HALT
        self.assertEqual(canon[0x80], Op.HALT)

    def test_v2_canonical_ops(self):
        v2 = ISADialect.v2()
        canon = v2.canonical_ops()
        self.assertEqual(canon[0x20], Op.ADD)
        self.assertEqual(canon[0x22], Op.MUL)
        self.assertEqual(canon[0x00], Op.HALT)


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
