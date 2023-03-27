from pathlib import Path

from lark import Lark


ROOT = Path(__file__).parent
GRAMMAR = ROOT / "grammar.lark"

class Parser:
    def __init__(self, grammar_file: str = GRAMMAR):
        with open(grammar_file) as fp:
            grammar = fp.read()

        self.lark = Lark(grammar)

    def parse(self, entry: str, display: bool = False):
        tree = self.lark.parse(entry)
        if display:
            print(tree.pretty())

        return 1


if __name__ == "__main__":
    parser = Parser()

    test_assignment = """
// set x to 200
let x = 200
let mut y = 300
"""

    parser.parse(test_assignment)

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

    parser.parse(test_conditionals)

    test_functions = """
fn add(a: u4, b: u4) u4 {
    a + b
}
"""

    test_fizzbuzz = """
fn fizz_buzz(input: u4) {
    return if (input % 3 == 0) and (input % 5 == 0) {
        "FizzBuzz"
    } else if input % 3 == 0 {
        "Fizz"
    } else if input % 5 == 0 {
        "Buzz"
    } else {
        input
    }
}

@say(fiz_buzz(7));
"""

#     test_conditionals = """
# // if block
# let x = if {
#     true => 1,
#     false => 2,
#     else => 3,
# };

# // if statement
# if true { return 1 };
# """
