"""Tests for Vocabulary pattern matching and natural language Interpreter."""

import pytest
from flux_vm import Vocabulary, Interpreter, Assembler, FluxVM


# ─── Vocabulary Tests ─────────────────────────────────────────────────────────

class TestVocabulary:

    def setup_method(self):
        self.vocab = Vocabulary()

    def test_add_and_match_simple(self):
        """Add a pattern and match text against it."""
        self.vocab.add("add $a and $b", "MOVI R0, ${a}\nMOVI R1, ${b}\nIADD R0, R0, R1\nHALT")
        result = self.vocab.match("add 5 and 3")
        assert result is not None
        assert result['groups'] == {'a': '5', 'b': '3'}

    def test_match_returns_assembly(self):
        """Match returns the assembly template."""
        self.vocab.add("compute $x", "MOVI R0, ${x}\nHALT")
        result = self.vocab.match("compute 99")
        assert result['assembly'] == "MOVI R0, ${x}\nHALT"

    def test_match_returns_result_reg(self):
        """Match returns the result register index."""
        self.vocab.add("do $x", "MOVI R5, ${x}\nHALT", result_reg=5)
        result = self.vocab.match("do 7")
        assert result['result_reg'] == 5

    def test_match_returns_description(self):
        """Match returns the description."""
        self.vocab.add("do $x", "MOVI R0, ${x}\nHALT", description="Test desc")
        result = self.vocab.match("do 7")
        assert result['description'] == "Test desc"

    def test_no_match(self):
        """Returns None when no pattern matches."""
        result = self.vocab.match("this matches nothing")
        assert result is None

    def test_empty_vocab_no_match(self):
        """Empty vocabulary matches nothing."""
        vocab = Vocabulary()
        assert vocab.match("anything") is None

    def test_match_is_case_insensitive(self):
        """Pattern matching should be case-insensitive."""
        self.vocab.add("compute $a + $b", "MOVI R0, ${a}\nHALT")
        result = self.vocab.match("COMPUTE 5 + 3")
        assert result is not None
        assert result['groups'] == {'a': '5', 'b': '3'}

    def test_first_match_wins(self):
        """First matching pattern is returned."""
        self.vocab.add("add $a and $b", "MOVI R0, ${a}\nHALT", description="first")
        self.vocab.add("add $a and $b", "MOVI R1, ${a}\nHALT", description="second")
        result = self.vocab.match("add 1 and 2")
        assert result['description'] == "first"

    def test_get_builtin_populates_patterns(self):
        """get_builtin() adds built-in patterns."""
        vocab = Vocabulary().get_builtin()
        # Should match built-in patterns
        assert vocab.match("compute 3 + 4") is not None
        assert vocab.match("compute 5 * 6") is not None
        assert vocab.match("factorial of 5") is not None
        assert vocab.match("double 10") is not None
        assert vocab.match("square 7") is not None

    def test_get_builtin_addition_groups(self):
        """Built-in addition pattern captures correct groups."""
        vocab = Vocabulary().get_builtin()
        result = vocab.match("compute 100 + 200")
        assert result['groups'] == {'a': '100', 'b': '200'}
        assert result['result_reg'] == 0
        assert result['description'] == "Add two numbers"

    def test_get_builtin_factorial_groups(self):
        """Built-in factorial pattern captures correct groups."""
        vocab = Vocabulary().get_builtin()
        result = vocab.match("factorial of 7")
        assert result['groups'] == {'n': '7'}
        assert result['result_reg'] == 1
        assert result['description'] == "Compute n!"

    def test_get_builtin_multiplication_groups(self):
        """Built-in multiplication pattern captures correct groups."""
        vocab = Vocabulary().get_builtin()
        result = vocab.match("compute 8 * 9")
        assert result['groups'] == {'a': '8', 'b': '9'}

    def test_get_builtin_double_groups(self):
        """Built-in double pattern captures correct groups."""
        vocab = Vocabulary().get_builtin()
        result = vocab.match("double 21")
        assert result['groups'] == {'n': '21'}

    def test_get_builtin_square_groups(self):
        """Built-in square pattern captures correct groups."""
        vocab = Vocabulary().get_builtin()
        result = vocab.match("square 6")
        assert result['groups'] == {'n': '6'}


# ─── Interpreter Tests ────────────────────────────────────────────────────────

class TestInterpreter:

    def setup_method(self):
        self.interp = Interpreter()

    def test_addition(self):
        """'compute 7 + 5' returns 12."""
        result, msg = self.interp.run("compute 7 + 5")
        assert result == 12
        assert "OK" in msg

    def test_multiplication(self):
        """'compute 8 * 9' returns 72."""
        result, msg = self.interp.run("compute 8 * 9")
        assert result == 72

    def test_factorial(self):
        """'factorial of 5' returns 120."""
        result, msg = self.interp.run("factorial of 5")
        assert result == 120

    def test_factorial_of_one(self):
        """'factorial of 1' returns 1."""
        result, msg = self.interp.run("factorial of 1")
        assert result == 1

    def test_double(self):
        """'double 21' returns 42."""
        result, msg = self.interp.run("double 21")
        assert result == 42

    def test_square(self):
        """'square 7' returns 49."""
        result, msg = self.interp.run("square 7")
        assert result == 49

    def test_no_match(self):
        """Unrecognized input returns None with error message."""
        result, msg = self.interp.run("fly to the moon")
        assert result is None
        assert "No match" in msg

    def test_status_message_includes_cycles(self):
        """Successful execution includes cycle count in message."""
        result, msg = self.interp.run("double 10")
        assert "cycles" in msg

    def test_custom_vocabulary(self):
        """Interpreter can use a custom vocabulary."""
        vocab = Vocabulary()
        # MOVI R0, n; MOVI R1, 3; IMUL R0, R0, R1; HALT  →  n * 3
        vocab.add("triple $n", "MOVI R0, ${n}\nMOVI R1, 3\nIMUL R0, R0, R1\nHALT")
        interp = Interpreter(vocab=vocab)
        result, msg = interp.run("triple 7")
        assert result == 21

    def test_builtin_vocabulary_has_5_patterns(self):
        """Default interpreter has 5 built-in vocabulary patterns."""
        assert len(self.interp.vocab.patterns) == 5
