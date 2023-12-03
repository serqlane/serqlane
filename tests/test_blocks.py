def test_mut_block(checking_executor):
    checking_executor(
        """
let mut i = 1
{
    i = i + i
}
""", "i", 2
    )


def test_block_expression(checking_executor):
    checking_executor("""
let x = {
    1 + 1
}
""", "x", 2)
