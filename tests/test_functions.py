def test_variable_shadowing(executor):
    executor("""
let x = 1

fn foo(x: int): int {
    return x
}
""")


def test_return(checking_executor):
    checking_executor("""
fn add(a: int, b: int): int {
    return a + b
}

let x = add(1, 1)
""", "x", 2)

    checking_executor("""
fn add(a: int, b: int): int {
    a + b
}

let x = add(1, 1)
""", "x", 2)


def test_recusive_function(checking_executor):
    checking_executor("""
fn add_one(x: int): int {
    if x == 1 {
        // should give us 4
        return add_one(x + 1) + 1
    }

    x + 1
}

let w = add_one(1)
""", "w", 4)
