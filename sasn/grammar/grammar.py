from collections import Counter, namedtuple
import sys

from config import settings

class Grammar:
    def __init__(self, rules, verbatim):
        self.rules = rules
        self.verbatim = verbatim

class Rule:
    def __init__(self, name, rhs):
        self.name = name
        self.rhs = rhs

class Rhs:
    def __init__(self, pos):
        self.pos = pos  # Position of Rhs definition in grammar file.

# ---------------------------
# Right hand side variations.
# ---------------------------

class RhsRegex(Rhs):
    def __init__(self, pattern, pos):
        super().__init__(pos)
        self.pattern = pattern

class RhsRule(Rhs):
    def __init__(self, rule, pos):
        super().__init__(pos)
        self.rule = rule

class RhsAction(Rhs):
    def __init__(self, name, pos):
        super().__init__(pos)
        self.name = name

class RhsSequence(Rhs):
    def __init__(self, children, pos):
        super().__init__(pos)
        self.children = children

class RhsChoice(Rhs):
    def __init__(self, children, pos):
        super().__init__(pos)
        self.children = children

class RhsZeroOrOne(Rhs):
    def __init__(self, child, pos):
        super().__init__(pos)
        self.child = child

class RhsZeroOrMany(Rhs):
    def __init__(self, child, pos):
        super().__init__(pos)
        self.child = child

class RhsOneOrMany(Rhs):
    def __init__(self, child, pos):
        super().__init__(pos)
        self.child = child

class RhsSkip(Rhs):
    def __init__(self, child, pos):
        super().__init__(pos)
        self.child = child

# ============================================================================
# Advanced grammar utilities.
# ============================================================================

def can_vanish(rhs, cv_cache=None):
    if cv_cache is None:
        cv_cache = dict()
    if rhs not in cv_cache:
        if isinstance(rhs, RhsRegex):
            cv_cache[rhs] = False
        elif isinstance(rhs, RhsRule):
            cv_cache[rhs] = can_vanish(rhs.rule.rhs, cv_cache)
        elif isinstance(rhs, RhsAction):
            # Actions shouldn't rely on what follows them to
            # decide if the parsing path is matched or not.
            cv_cache[rhs] = False
        elif isinstance(rhs, RhsSequence):
            cv_cache[rhs] = True
            for child in rhs.children:
                if not can_vanish(child, cv_cache):
                    cv_cache[rhs] = False
                    break
        elif isinstance(rhs, RhsChoice):
            cv_cache[rhs] = False
            for child in rhs.children:
                if can_vanish(child, cv_cache):
                    cv_cache[rhs] = True
                    break
        elif isinstance(rhs, (RhsZeroOrOne, RhsZeroOrMany)):
            cv_cache[rhs] = True
        elif isinstance(rhs, RhsOneOrMany):
            cv_cache[rhs] = can_vanish(rhs.child, cv_cache)
        elif isinstance(rhs, RhsSkip):
            cv_cache[rhs] = can_vanish(rhs.child, cv_cache)
    return cv_cache[rhs]

# ------------------------------------

def leftmost_rules(rhs, cv_cache=None, parents=None):
    result = [rhs]
    if parents is None:
        parents = list()
    if rhs in parents:
        return result  # Left-recursion encountered.
    parents.append(rhs)
    if isinstance(rhs, RhsRegex):
        pass
    elif isinstance(rhs, RhsRule):
        result += leftmost_rules(rhs.rule.rhs, cv_cache, parents)
    elif isinstance(rhs, RhsAction):
        pass
    elif isinstance(rhs, RhsSequence):
        for child in rhs.children:
            result += leftmost_rules(child, cv_cache, parents)
            if not can_vanish(child, cv_cache):
                break
    elif isinstance(rhs, RhsChoice):
        for child in rhs.children:
            result += leftmost_rules(child, cv_cache, parents)
    elif isinstance(rhs, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
        result += leftmost_rules(rhs.child, cv_cache, parents)
    elif isinstance(rhs, RhsSkip):
        result += leftmost_rules(rhs.child, cv_cache, parents)
    return result

# ------------------------------------

RhsActionMark = namedtuple('RhsActionMark', ('name'))

def first_set(rhs, fs_cache=None, cv_cache=None):
    if fs_cache is None:
        fs_cache = dict()
    if rhs not in fs_cache:
        fs_cache[rhs] = list()
        if isinstance(rhs, RhsRegex):
            fs_cache[rhs].append(rhs.pattern)
        elif isinstance(rhs, RhsRule):
            fs_cache[rhs] = first_set(rhs.rule.rhs, fs_cache, cv_cache)
        elif isinstance(rhs, RhsAction):
            # Actions don't have precalculated first sets.
            # They are called unconditionally and the results is
            # interpreted to decide whether to proceed or not.
            # This is a placeholder for the action call.
            fs_cache[rhs].append(RhsActionMark(rhs.name))
        elif isinstance(rhs, RhsSequence):
            for child in rhs.children:
                fs_cache[rhs].extend(first_set(child, fs_cache, cv_cache))
                if not can_vanish(child, cv_cache):
                    break
        elif isinstance(rhs, RhsChoice):
            for child in rhs.children:
                fs_cache[rhs].extend(first_set(child, fs_cache, cv_cache))
        elif isinstance(rhs, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
            fs_cache[rhs] = first_set(rhs.child, fs_cache, cv_cache)
        elif isinstance(rhs, RhsSkip):
            fs_cache[rhs] = first_set(rhs.child, fs_cache, cv_cache)
        # Produce unique list.
        fs_cache[rhs] = list(dict.fromkeys(fs_cache[rhs]))
    return fs_cache[rhs]

# ------------------------------------

def lookahead_length(rhs, fs_cache=None, cv_cache=None):
    fs = first_set(rhs, fs_cache, cv_cache)
    lengths = [len(e) for e in fs if isinstance(e, str)]
    if not lengths:
        return 0
    return max(lengths)

# ------------------------------------

def will_vanish(rhs, wv_cache=None):
    if wv_cache is None:
        wv_cache = dict()
    if rhs not in wv_cache:
        if isinstance(rhs, RhsRegex):
            wv_cache[rhs] = False
        elif isinstance(rhs, RhsRule):
            wv_cache[rhs] = will_vanish(rhs.rule.rhs, wv_cache)
        elif isinstance(rhs, RhsAction):
            wv_cache[rhs] = False  # Actions can return nodes.
        elif isinstance(rhs, (RhsSequence, RhsChoice)):
            wv_cache[rhs] = True
            for child in rhs.children:
                if not will_vanish(child, wv_cache):
                    wv_cache[rhs] = False
                    break
        elif isinstance(rhs, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
            wv_cache[rhs] = can_vanish(rhs.child, wv_cache)
        elif isinstance(rhs, RhsSkip):
            wv_cache[rhs] = True
    return wv_cache[rhs]

# ============================================================================
# Visitor utilities.
# ============================================================================

class GrammarVisitorBase:
    def __init__(self):
        self.cache = dict()

    def visit(self, obj, *args):
        klass = obj.__class__
        meth = self.cache.get(klass)
        if meth is None:
            methname = 'visit' + klass.__name__
            meth = getattr(self, methname, None)
            self.cache[klass] = meth
        self.visit_wrapper(meth, obj, *args)

    # Override this method to perform specific
    # actions before/after/instead of visiting `obj`.
    def visit_wrapper(self, meth, obj, *args):
        if meth:
            meth(obj, *args)

# ============================================================================
# Grammar check visitor. (Without type checking.)
# ============================================================================

class GrammarCheckVisitorState:
    def __init__(self, grammar):
        self.name2rule = {rule.name:rule for rule in grammar.rules}
        self.verbatim = grammar.verbatim
        self.cv_cache = dict()
        self.fs_cache = dict()

    def warning(self, msg, rhs):
        settings.LOGGER.warning(f'Grammar warning at {rhs.pos}: {msg}')

    def error(self, msg, rhs):
        settings.LOGGER.critical(f'Grammar error at {rhs.pos}: {msg}')
        sys.exit(1)

class GrammarCheckVisitor(GrammarVisitorBase):
    def visitGrammar(self, grammar):
        state = GrammarCheckVisitorState(grammar)
        for rule in grammar.rules:
            self.visit(rule, state)

    def visitRule(self, rule, state):
        if rule.rhs in leftmost_rules(rule.rhs, state.cv_cache)[1:]:
            state.error(f'Rule "{rule.name}" is left-recursive', rule.rhs)
        self.visit(rule.rhs, state)

    def visitRhsRegex(self, rhs, state):
        pass

    def visitRhsRule(self, rhs, state):
        if rhs.rule.name not in state.name2rule:
            state.error(f'Rule "{rhs.rule.name}" is not defined', rule.rhs)

    def visitRhsAction(self, rhs, state):
        if f'parse_{rhs.name}' not in state.verbatim:
            state.error(f'Action "{rhs.name}" is not defined', rule.rhs)

    def visitRhsSequence(self, rhs, state):
        if len(rhs.children) == 0:
            state.warning('Empty sequence', rhs)
        elif len(rhs.children) == 0:
            state.warning('Sequence with single element', rhs)
        for child in rhs.children:
            self.visit(child, state)

    def visitRhsChoice(self, rhs, state):
        if len(rhs.children) == 0:
            state.warning('Empty choice', rhs)
        elif len(rhs.children) == 1:
            state.warning('Choice with single option', rhs)
        counter = Counter()
        for child in rhs.children:
            counter.update(first_set(child, state.fs_cache, state.cv_cache))
        if len(counter) > 0 and counter.most_common(1)[0][1] > 1:
            state.warning('Choice is not LL(1)', rhs)
        for child in rhs.children:
            self.visit(child, state)

    def visitRhsZeroOrOne(self, rhs, state):
        if isinstance(rhs.child, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
            state.warning('Unnecessary zero-or-one', rhs)
        self.visit(rhs.child, state)

    def visitRhsZeroOrMany(self, rhs, state):
        if isinstance(rhs.child, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
            state.warning('Unnecessary zero-or-many', rhs)
        self.visit(rhs.child, state)

    def visitRhsOneOrMany(self, rhs, state):
        if isinstance(rhs.child, (RhsZeroOrOne, RhsZeroOrMany, RhsOneOrMany)):
            state.warning('Unnecessary one-or-many', rhs)
        self.visit(rhs.child, state)

    def visitRhsSkip(self, rhs, state):
        if isinstance(rhs.child, RhsSkip):
            state.warning('Unnecessary skip', rhs)
        self.visit(rhs.child, state)

# --------------
# Main function.
# --------------

def check_grammar(grammar):
    checker = GrammarCheckVisitor()
    checker.visit(grammar)

# ============================================================================
# Grammar print visitor.
# ============================================================================

class GrammarPrintVisitorState:
    def __init__(self, outbuffer):
        self.outbuffer = outbuffer
        self.indent_level = 0

    def emit(self, msg=''):
        self.outbuffer.write(self.indent_level * 4 * ' ' + msg + '\n')

class GrammarPrintVisitor(GrammarVisitorBase):
    def __init__(self):
        super().__init__()

    def visit_wrapper(self, meth, obj, state, *args):
        if meth:
            if isinstance(obj, Rhs):  # Print Rhs type before every visit.
                state.emit(f'Rhs: type = {obj.__class__.__name__}')
                state.indent_level += 1
                meth(obj, state, *args)
                state.indent_level -= 1
            else:
                meth(obj, state, *args)

    def visitGrammar(self, grammar, outbuffer):
        state = GrammarPrintVisitorState(outbuffer)
        state.emit(f'Grammar: root type = {grammar.rules[0].name}')
        state.indent_level += 1
        for rule in grammar.rules:
            self.visit(rule, state)
        state.emit(f'Verbatim: """{grammar.verbatim}"""')
        state.indent_level -= 1

    def visitRule(self, rule, state):
        state.emit(f'Rule: name = {rule.name}')
        state.indent_level += 1
        self.visit(rule.rhs, state)
        state.indent_level -= 1

    def visitRhsRegex(self, rhs, state):
        state.emit(f'Regex: pattern = \'{rhs.pattern}\'')

    def visitRhsRule(self, rhs, state):
        state.emit(f'Rule: rule = {rhs.rule.name}')

    def visitRhsAction(self, rhs, state):
        state.emit(f'Action: name = {rhs.name}')

    def visitRhsSequence(self, rhs, state):
        for child in rhs.children:
            self.visit(child, state)

    def visitRhsChoice(self, rhs, state):
        for child in rhs.children:
            self.visit(child, state)

    def visitRhsZeroOrOne(self, rhs, state):
        self.visit(rhs.child, state)

    def visitRhsZeroOrMany(self, rhs, state):
        self.visit(rhs.child, state)

    def visitRhsOneOrMany(self, rhs, state):
        self.visit(rhs.child, state)

    def visitRhsSkip(self, rhs, state):
        self.visit(rhs.child, state)

# --------------
# Main function.
# --------------

def print_grammar(grammar, outbuffer):
    printer = GrammarPrintVisitor()
    printer.visit(grammar, outbuffer)
