from collections import namedtuple

from grammar.syntax_tree import TokenNode, RuleNode

ApplyRuleAction = namedtuple('ApplyRuleAction', ('name'))

ReduceAction = namedtuple('ReduceAction', ('count'))

GenTokenAction = namedtuple('GenTokenAction', ('token'))

# ============================================================================
# Action utilities.
# ============================================================================

def print_actions(actions, outbuffer):
    for action in actions:
        if isinstance(action, ApplyRuleAction):
            outbuffer.write(f'ApplyRule action: {action.name}\n')
        elif isinstance(action, ReduceAction):
            outbuffer.write(f'Reduce action: {action.count}\n')
        elif isinstance(action, GenTokenAction):
            outbuffer.write(f'GenToken action: {action.token}\n')

# ------------------------------------

def compare_actions(lhs, rhs):
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if (isinstance(lhs[i], GenTokenAction) and
            isinstance(rhs[i], GenTokenAction)):
            return lhs[i].token == rhs[i].token
        elif (isinstance(lhs[i], ReduceAction) and
              isinstance(rhs[i], ReduceAction)):
            return lhs[i].count == rhs[i].count
        elif (isinstance(lhs[i], ApplyRuleAction) and
              isinstance(rhs[i], ApplyRuleAction)):
            return lhs[i].name == rhs[i].name
        else:
            return False

# ============================================================================
# Conversion utilities.
# ============================================================================

def syntax_tree_to_actions(node):
    actions = list()
    if isinstance(node, TokenNode):
        actions.append(GenTokenAction(node.token))
    elif isinstance(node, RuleNode):
        actions.append(ApplyRuleAction(node.name))
        for child in node.children:
            actions.extend(syntax_tree_to_actions(child))
        if isinstance(actions[-1], ReduceAction):
            actions[-1] = ReduceAction(actions[-1].count + 1)
        else:
            actions.append(ReduceAction(1))
    return actions

# ------------------------------------

class ActionsToSyntaxTreeState:
    def __init__(self, actions):
        self.actions = actions
        self.idx = 0
        self.stack = []  # Keep track of rules.

    @property
    def curr(self):
        return self.actions[self.idx]

def actions_to_syntax_tree(actions):
    idx = 0
    stack = []
    # Emit root node.
    syntax_tree = RuleNode(actions[0].name, [])
    stack.append(syntax_tree)
    idx += 1
    while idx < len(actions):
        curr = actions[idx]
        if isinstance(curr, ApplyRuleAction):
            node = RuleNode(curr.name, [])
            stack[-1].children.append(node)
            stack.append(node)  # Add children to this rule.
        elif isinstance(curr, ReduceAction):
            stack = stack[:-curr.count]  # Stop adding children to latest rule.
        elif isinstance(curr, GenTokenAction):
            node = TokenNode(curr.token)
            stack[-1].children.append(node)
        idx += 1
    return syntax_tree
