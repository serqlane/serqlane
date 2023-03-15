from lark import Lark


class Parser:
    def __init__(self, grammar_file: str = "grammar.lark"):
        with open(grammar_file) as fp:
            grammar = fp.read()

        self.lark = Lark(grammar)

    def parse(self):
        pass


if __name__ == "__main__":
    test_assignment = """
let x = 200;
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

    test_conditionals = """
// if block
let x = if {
    true => 1,
    false => 2,
    else => 3,
};

// if statement
if true { return 1 };
"""
