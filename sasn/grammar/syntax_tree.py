from config import settings

class SyntaxTree:
    pass

class TokenNode(SyntaxTree):
    def __init__(self, token):
        self.token = token

class RuleNode(SyntaxTree):
    def __init__(self, name, children):
        self.name = name
        self.children = children

# ============================================================================
# Syntax tree utilities.
# ============================================================================

def print_syntax_tree(node, outbuffer, indent_level=0):
    if isinstance(node, TokenNode):
        outbuffer.write(indent_level * 4 * ' ' + f'Token: {node.token}\n')
    else:
        outbuffer.write(indent_level * 4 * ' ' + f'Rule: {node.name}\n')
        for child in node.children:
            print_syntax_tree(child, outbuffer, indent_level+1)

# ------------------------------------

def compare_syntax_trees(lhs, rhs):
    if isinstance(lhs, TokenNode) and isinstance(rhs, TokenNode):
        return lhs.token == rhs.token
    elif isinstance(lhs, RuleNode) and isinstance(rhs, RuleNode):
        if lhs.name != rhs.name:
            return False
        if len(lhs.children) != len(rhs.children):
            return False
        for i in range(len(lhs.children)):
            if not compare_syntax_trees(lhs.children[i], rhs.children[i]):
                return False
        return True
