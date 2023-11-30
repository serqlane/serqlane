from lark import Token, Tree
from lark.visitors import Transformer

from serqlane.parser import SerqParser
from serqlane.serq_ast import *


class SerqTransformer(Transformer[Token, AstNode]):
    def id(self, tokens: list[Token]):
        assert len(tokens) == 1
        token = tokens[0]
        return Identification(name=token.value)

    def start(self, args: list[AstNode]):
        return Module(args)

    def literal(self, tokens: list[Tree[Token]]):
        assert len(tokens) == 1
        tree = tokens[0]
        literal_type = tree.data
        value = tree.children[0].value # type: ignore
        return Literal(type=LiteralType[literal_type], value=value)

    def define(self, tokens: list[Assignment | Token]):
        mutable = isinstance(tokens[0], Token) and tokens[0].type == "MUT"
        assignment: Assignment = tokens[-1] # type: ignore
        return Define(mutable, assignment.name, assignment.value)

    def assignment(self, tokens: list[Expression | Identification]):
        return Assignment(tokens[0], tokens[1]) # type: ignore

    def operator(self, tokens):
        return Operator[tokens[0].data.value]
    
    def operation(self, tokens):
        # TODO: handle 1 + 1 + 1
        #  and 1 + (1 + 1)
        left, operator, right = tokens
        return Operation(left, operator, right)

if __name__ == "__main__":
    test = """1 + a"""

    parser = SerqParser()
    tree = parser.parse(test, display=False)

    transformer = SerqTransformer()
    transformed = transformer.transform(tree)

    from pprint import pprint

    pprint(transformed)
