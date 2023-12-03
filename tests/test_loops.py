def test_while(checking_executor):
    checking_executor("""
let mut x = 10
while x > 0 {
    x = x - 1
}
""", "x", 0)

def test_break(checking_executor):
    checking_executor("""
let mut x = 10
while x > 0 {
    x = x - 1
    if x == 5 {
        break
    }
}
""", "x", 5)

def test_continue(checking_executor):
    checking_executor("""
let mut x = 3
while x > 0 {
    x = x - 1
    if x == 2 {
        continue
    }
    x = x - 1
}
""", "x", 0)
