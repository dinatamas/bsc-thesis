from collections import namedtuple
import re
import sys

from config import settings
from grammar.grammar import *

def parse_grammar(description):
    metatokens = MetaLexer.lex(description)
    grammar = MetaParser.parse(metatokens)
    return grammar

# ------------------------------------

class GrammarError(Exception):
    def __init__(self, msg, pos):
        super().__init__()
        self.msg = msg
        self.pos = pos

    def __str__(self):
        return f'Grammar error at {self.pos}: {self.msg}'

# ============================================================================
# Lexical analysis.
# ============================================================================

MetaToken = namedtuple('MetaToken', ['kind', 'string', 'pos'])

# ------------------------------------

class MetaLexerState:
    def __init__(self, description):
        self.buffer = description
        self.pos = (0, 0)

    def advance(self, count):
        for char in self.buffer[:count]:
            if char == '\n':
                self.pos = (self.pos[0] + 1, 0)
            else:
                self.pos = (self.pos[0], self.pos[1] + 1)
        self.buffer = self.buffer[count:]

    def error(self, msg):
        raise GrammarError(msg, (self.pos[0] + 1, self.pos[1] + 1))

# ------------------------------------

class MetaLexer:
    regexs = (
        ('REGEX_SINGLE',  re.compile(r'\'((?:\\.|[^\'\\])*)\'')),
        ('REGEX_DOUBLE',  re.compile(r'\"((?:\\.|[^\"\\])*)\"')),
        ('RULE_NAME',     re.compile(r'[a-zA-Z_]+')),
        ('ACTION_NAME',   re.compile(r'@([a-zA-Z_]+)')),
        ('VERBATIM',      re.compile(r'(?:%\{\n)([^\}%]*)(?:\n\}%)')),
        ('COMMENT',       re.compile(r'#[^\n]*\n')),
        ('=',             re.compile(r'\=')),
        (',',             re.compile(r'\,')),
        (';',             re.compile(r'\;')),
        ('|',             re.compile(r'\|')),
        ('?',             re.compile(r'\?')),
        ('*',             re.compile(r'\*')),
        ('+',             re.compile(r'\+')),
        ('(',             re.compile(r'\(')),
        (')',             re.compile(r'\)')),
        ('>',             re.compile(r'\>')),
        ('WHITESPACE',    re.compile(r'[ \t\n]+')),
        ('MISMATCH',      re.compile(r'.'))
    )

    @staticmethod
    def lex(description):
        state = MetaLexerState(description)
        while state.buffer:
            # Find longest match.
            kind = None
            string = ''
            match_len = 0
            for curr_kind, curr_pattern in MetaLexer.regexs:
                match = re.match(curr_pattern, state.buffer)
                if match:
                    if match.groups():
                        curr_string = match.groups()[0]
                    else:
                        curr_string = match.group()
                    if len(match.group()) > match_len:
                        kind = curr_kind
                        string = curr_string
                        match_len = len(match.group())

            # Handle matched regex.
            if kind in ('WHITESPACE', 'COMMENT'):
                state.advance(match_len)
                continue
            elif kind in ('REGEX_SINGLE', 'REGEX_DOUBLE'):
                kind = 'REGEX'  # Unify regexs with different quotes.
            elif kind == 'MISMATCH':
                state.error(f'Invalid character: {repr(string[0])}')
            yield MetaToken(kind, string, state.pos)
            state.advance(match_len)

# ============================================================================
# Syntactic analysis.
# ============================================================================

class MetaParserState:
    def __init__(self, metatokens):
        self.metatokens = iter(metatokens)
        self.curr = None
        self.finished = False
        self.advance()

    def advance(self):
        try:
            self.curr = next(self.metatokens)
        except StopIteration:
            self.finished = True
            self.curr = None

    def error(self, msg):
        raise GrammarError(msg, (self.pos[0] + 1, self.pos[1] + 1))

    # -------------------
    # Recognize patterns.
    # -------------------

    def peek(self, *kinds):  # Peek but do not consume.
        if self.finished:
            return None
        for kind in kinds:
            if self.curr.kind == kind:
                return self.curr
        return None

    def accept(self, *kinds):  # Consume if possible.
        matched = self.peek(*kinds)
        if matched:
            self.advance()
        return matched

    def expect(self, *kinds):  # Consume or fail.
        matched = self.accept(*kinds)
        if matched:
            return matched
        if self.finished:
            self.error((f'Unexpected EOF: '
                        f'expected one of: {[k for k in kinds]}'))
        self.error((f'Invalid metatoken: {repr(self.curr)}, '
                    f'expected one of: {[k for k in kinds]}'))

# ------------------------------------

class MetaParser:
    @staticmethod
    def parse(metatokens):
        state = MetaParserState(metatokens)
        rules = list()
        verbatim = ''
        while not state.finished:
            if state.peek('VERBATIM'):
                verbatim += state.expect('VERBATIM').string
            else:
                rules.append(MetaParser.parse_rule(state))
        set_rule_references(rules)
        return Grammar(rules, verbatim)

    @staticmethod
    def parse_rule(state):
        name = state.expect('RULE_NAME').string
        state.expect('=')
        rhs = MetaParser.parse_choice(state)
        state.expect(';')
        return Rule(name, rhs)

    # ------------------------------------
    # Parser functions for Rhs variations.
    # ------------------------------------

    @staticmethod
    def parse_choice(state):
        pos = state.curr.pos
        children = [MetaParser.parse_sequence(state)]
        while state.accept('|'):
            children.append(MetaParser.parse_sequence(state))
        if len(children) == 1:  # Single choices can be simplified.
            return children[0]
        return RhsChoice(children, pos)

    @staticmethod
    def parse_sequence(state):
        pos = state.curr.pos
        children = [MetaParser.parse_standalone(state)]
        while state.accept(','):
            children.append(MetaParser.parse_standalone(state))
        if len(children) == 1:  # Single sequences can be simplified.
            return children[0]
        return RhsSequence(children, pos)

    @staticmethod
    def parse_standalone(state):
        pos = state.curr.pos
        if state.accept('('):
            standalone = MetaParser.parse_choice(state)
            state.expect(')')
        elif state.accept('?'):
            standalone = MetaParser.parse_zero_or_one(state)
        elif state.accept('*'):
            standalone = MetaParser.parse_zero_or_many(state)
        elif state.accept('+'):
            standalone = MetaParser.parse_one_or_many(state)
        elif state.accept('>'):
            standalone = MetaParser.parse_skip(state)
        elif state.peek('RULE_NAME'):
            metatoken = state.expect('RULE_NAME')
            standalone = RhsRule(metatoken.string, pos)
        elif state.peek('REGEX'):
            metatoken = state.expect('REGEX')
            standalone = RhsRegex(metatoken.string, pos)
        elif state.peek('ACTION_NAME'):
            metatoken = state.expect('ACTION_NAME')
            standalone = RhsAction(metatoken.string, pos)
        else:
            if state.finished:
                state.error('Unexpected EOF')
            else:
                state.error(f'Invalid metatoken: {repr(state.curr)}')
        return standalone

    @staticmethod
    def parse_zero_or_one(state):
        pos = state.curr.pos
        child = MetaParser.parse_standalone(state)
        return RhsZeroOrOne(child, pos)

    @staticmethod
    def parse_zero_or_many(state):
        pos = state.curr.pos
        child = MetaParser.parse_standalone(state)
        return RhsZeroOrMany(child, pos)

    @staticmethod
    def parse_one_or_many(state):
        pos = state.curr.pos
        child = MetaParser.parse_standalone(state)
        return RhsOneOrMany(child, pos)

    @staticmethod
    def parse_skip(state):
        pos = state.curr.pos
        child = MetaParser.parse_standalone(state)
        return RhsSkip(child, pos)

# ============================================================================
# Postprocessing after parsing.
# ============================================================================

def set_rule_references(rules):
    name2rule = {rule.name:rule for rule in rules}
    for rule in rules:
        set_rule_references_helper(rule.rhs, name2rule)

def set_rule_references_helper(rhs, name2rule):
    if isinstance(rhs, (RhsRegex, RhsAction)):
        pass
    elif isinstance(rhs, RhsRule):
        if rhs.rule not in name2rule:
            raise GrammarError(f'Rule "{rhs.rule}" is not defined', rhs.pos)
        rhs.rule = name2rule[rhs.rule]  # Replace name with reference to rule.
    elif isinstance(rhs, (RhsSequence, RhsChoice)):
        for child in rhs.children:
            set_rule_references_helper(child, name2rule)
    elif isinstance(rhs, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
        set_rule_references_helper(rhs.child, name2rule)
    elif isinstance(rhs, RhsSkip):
        set_rule_references_helper(rhs.child, name2rule)
