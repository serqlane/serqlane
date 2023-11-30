from pathlib import Path

from lark import Lark

ROOT = Path(__file__).parent
GRAMMAR = ROOT / "grammar.lark"


class SerqParser:
    def __init__(self, grammar_file: str | Path = GRAMMAR):
        with open(grammar_file) as fp:
            grammar = fp.read()

        self.lark = Lark(grammar)

    def parse(self, entry: str, display: bool = True):
        tree = self.lark.parse(entry)
        if display:
            print(tree.pretty())

        return tree


if __name__ == "__main__":
    parser = SerqParser()

    test_assignment = """
// set x to 200
let x = 200
let mut y = 300
"""

    parser.parse(test_assignment, display=False)

    test_conditionals = """
// if statement
if true {
    let x = 300
}

let x = if false {
    "joe"
}

if {
    true => 1,
    else => 2
}

let x = "test string 123"
"""

    parser.parse(test_conditionals, display=False)

    test_contained_if = """
if (input % 3 == 0) and (input % 5 == 0) {
    "FizzBuzz"
}
"""

    parser.parse(test_contained_if, display=False)

    test_functions = """
fn add(a: u4, b: u4) u4 {
    let x = a + b - "test";
    a + b - 1

    if {
        x == 2 => 1,
        else => 2
    }
}
"""

    parser.parse(test_functions, display=False)

    test_operator = """
let x = 1 + 1;
"""

    parser.parse(test_operator)

    test_fizzbuzz = """
fn fizz_buzz(input: u4) str {
    return if (input % 3 == 0) and (input % 5 == 0) {
        "FizzBuzz"
    } else if input % 3 == 0 {
        "Fizz"
    } else if input % 5 == 0 {
        "Buzz"
    } else {
        // ignore type
        input
    }
}

print(fiz_buzz(input=7));
"""

    parser.parse(test_fizzbuzz, display=False)
