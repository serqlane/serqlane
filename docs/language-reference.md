# Language Reference
This page contains the language reference for serqlane. By the end of this page you should have a good understanding of the language and be able to write your own programs.

!!! note
    Expect this page to be updated frequently as the language is still in development. 

## Naming Convention
The naming convention for serqlane is snake_case. This means that all names should be lowercase and words should be separated by underscores. For example, `my_variable` is a valid name, but `myVariable` is not.

## Comments
Comments are used to add notes to your code. They are ignored by the compiler and are only used to help you and future developers understand your code. To add a comment, start a line with `//`. Everything after the `//` will be ignored by the compiler. For example:  

``` rust linenums="1"
// This is a comment
dbg("This code should still run") // This is also a comment
```
```
DBG: This code should still run
```

## Variables
Variables are used to store data. They can be used to store any type of data. You cannot change the type of the variable once it is created. Addtionally, you cannot change the value of the variable unless it is marked as [mutable](#mutability). 

To create a inferrable variable, use the `let` keyword followed by the name of the variable, an equals sign, and the value you want to store in the variable. For example this variable will be inferred as a string:

``` rust linenums="1"
let my_variable = "this is a string"
dbg(my_variable)
```
```
DBG: This is a string
```

Or explicitly declare the type of the variable by adding a colon and the type of the variable after the name of the variable. For example this variable will be explicitly declared as a string:

``` rust linenums="1"
let my_variable: string = "this is a string"
dbg(my_variable)
```
```
DBG: This is a string
```

Attempting to change the value of a variable that lacks the `mut` keyword will result in an error. For example:

``` rust linenums="1"
let my_variable = "This is a string"
my_variable = "This is a different string"
dbg(my_variable)
```
```
ValueError: my_variable is not mutable
```

Similarly, attempting to change the type of a variable will also result in an error. For example:

``` rust linenums="1"
let my_variable = "This is a string"
my_variable = 42
dbg(my_variable)
```
```
AssertionError
```


### Mutability
By default, variables are immutable. This means that once you create a variable, you cannot change its value. To make a variable mutable, use the `mut` keyword before the name of the variable. For example:

``` rust linenums="1"
let mut my_variable = "Hello World!"
my_variable = "Goodbye World!"
dbg(my_variable)
```
```
DBG: Goodbye World!
```

## Functions
Functions are used to group together a set of instructions. They can be called from anywhere in your program. To create a function, use the `fn` keyword followed by the name of the function, a set of parentheses, optional parameters (separated by commas), and a set of curly braces.

A function in its simplest form looks like this:  

``` rust linenums="1"
fn my_function() {
    dbg("This is my function!")
}
my_function()
```
```
DBG: This is my function!
```

### Function Parameters
Functions can take parameters. Parameters are used to pass data into a function. Each paramter is required to have a type. To add parameters to a function, add the name of the parameter followed by a colon and the type of the parameter. Parameters are separated by commas. For example:

``` rust linenums="1"
fn my_function(my_parameter: string) {
    dbg(my_parameter)
}
my_function("This is my parameter!")
```
```
DBG: This is my parameter!
```
Or, if you want to pass in multiple parameters:

``` rust linenums="1"
fn multiple_parameters(parameter_one: string, parameter_two: int) {
    dbg(parameter_one)
    dbg(parameter_two)
}
multiple_parameters("This is my first parameter!", 42)
```
```
DBG: This is my first parameter!
DBG: 42
```

### Function Return Values
Functions can return values. To return a value from a function, use the `return` keyword followed by the value you want to return. Lastly, add the type of the value you are returning after the function parameters using `->`. For example:

``` rust linenums="1"
fn my_function() -> string {
    let x = "This is my return value!"
    return x
}

fn my_other_function() -> string {
    let x = "This is my other return value!"
    x
}
dbg(my_function())
dbg(my_other_function())
```
``` 
DBG: This is my return value!
DBG: This is my other return value!
```

### Function Overloading
Functions can be overloaded. This means that you can have multiple functions with the same name, but different parameters. For example:

``` rust linenums="1"
fn my_function() {
    dbg("This is my function!")
}

fn my_function(parameter: string) {
    dbg(parameter)
}

my_function()
my_function("This is my parameter!")
```
```
DBG: This is my function!
DBG: This is my parameter!
```

### Function Scope
Functions have their own scope. This means that variables declared inside of a function are not accessible outside of the function. For example:

``` rust linenums="1"
fn my_function() {
    let x = "This is my variable!"
}
my_function()
dbg(x)
```
```
ValueError: Bad identifier: x
```

## Shadowing
Shadowing is allowed in serqlane. This means that you can declare a variable with the same name as another variable as long as the new variable is in a different scope. For example:

``` rust linenums="1"
let x = "This is my first variable!"

fn foo(x: string) -> string {
    dbg(x)
    return x
}

foo("This is my second variable!")
dbg(x)
```
```
DBG: This is my second variable!
DBG: This is my first variable!
```

## Types
serqlane is a statically typed language. This means that every variable and function must have a type. The following types are supported by serqlane:

### Primitive Types
Primitive types are the most basic types in serqlane. They are the building blocks for all other types. The following primitive types are supported by serqlane:

- `bool`: A boolean value. Can be either `true` or `false`.
- `int`: An integer value. Can be any whole number.
    - `int8`: An 8-bit integer value. Can be any whole number between -128 and 127.
    - `int16`: A 16-bit integer value. Can be any whole number between -32,768 and 32,767.
    - `int32`: A 32-bit integer value. Can be any whole number between -2,147,483,648 and 2,147,483,647.
    - `int64`: A 64-bit integer value. Can be any whole number between -9,223,372,036,854,775,808 and 9,223,372,036,854,775,807.
    
    
    - `uint8`: An 8-bit unsigned integer value. Can be any whole number between 0 and 255.
    - `uint16`: A 16-bit unsigned integer value. Can be any whole number between 0 and 65,535.
    - `uint32`: A 32-bit unsigned integer value. Can be any whole number between 0 and 4,294,967,295.
    - `uint64`: A 64-bit unsigned integer value. Can be any whole number between 0 and 18,446,744,073,709,551,615.

- `float`: A floating point value. Can be any decimal number.
    - `float32`: A 32-bit floating point value. Can be any decimal number between 1.175494351e-38 and 3.402823466e+38.
    - `float64`: A 64-bit floating point value. Can be any decimal number between 2.2250738585072014e-308 and 1.7976931348623157e+308.

- `char`: A single character. Can be any single character.


### Compound Types
Compound types are types that are made up of other types. The following compound types are supported by serqlane:

- `string`: A string of characters. Can be any sequence of characters.

!!! Not-Implemented
    - `array`: An array of values. Can be any sequence of values.

### Structs
You can create custom types by using the `struct` keyword. A struct is a collection of named values. To create a struct, use the `struct` keyword followed by the name of the struct, a set of curly braces, and a list of fields. Each field is required to have a name and a type. Fields are separated by commas. For example:

``` rust linenums="1"
struct Person {
    age: int64
}
let mut p = Person()
p.age = 42
```
To access a field in a struct, use the dot operator. From the previous example, to access the `age` field in the `Person` struct, you would use `p.age`. For example:

``` rust linenums="1"
struct Person {
    age: int64
}
let mut p = Person()
p.age = 42
dbg(p.age)
```
```
DBG: 42
```

### Aliases
You can create aliases for types by using the `alias` keyword. An alias is a new name for an existing type. To create an alias, use the `alias` keyword followed by the name of the alias, an equals sign, and the type you want to alias. For example:

``` rust linenums="1"
alias int = int64
let x: int = 42
dbg(x)
```
```
DBG: 42
```

## Loops
Loops are used to repeat a set of instructions. The following types of loops are supported by serqlane:

### For Loops
!!! Not-Implemented
    For loops are not yet implemented. 
    What is written below is a rough idea of what they will look like.

For loops are used to repeat a set of instructions a set number of times. To create a for loop, use the `for` keyword followed by a set of parentheses, a variable name, the `in` keyword, a range or array, and a set of curly braces. For example:

``` rust linenums="1"
for i in 0..3 {
    dbg(i)
}
```
```
DBG: 0
DBG: 1
DBG: 2
```


``` rust linenums="1"
let x = ["foo","bar","baz"]
for i in x {
    dbg(i)
}
```
```
DBG: foo
DBG: bar
DBG: baz
```

### While Loops
While loops are used to repeat a set of instructions while a condition is true. To create a while loop, use the `while` keyword followed by a set of parentheses, a condition, and a set of curly braces. For example:

``` rust linenums="1"
let mut x = 0
while x < 4 {
    dbg(x)
    x = x + 1
}
```
```
DBG: 0
DBG: 1
DBG: 2
DBG: 3
```

## Conditionals
Conditionals are used to execute a set of instructions based on a condition. The following types of conditionals are supported by serqlane:

### If Statements
If statements are used to execute a set of instructions if a condition is true. To create an if statement, use the `if` keyword followed by a set of parentheses, a condition, and a set of curly braces. For example:

``` rust linenums="1"
let x = 42
if x == 42 {
    dbg("x is 42")
}
```
```
DBG: x is 42
```

### If-Else Statements
If-else statements are used to execute a set of instructions if a condition is true and another set of instructions if the condition is false. To create an if-else statement, use the `if` keyword followed by a set of parentheses, a condition, a set of curly braces, the `else` keyword, and another set of curly braces. For example:

``` rust linenums="1"
let x = 54
if x == 42 {
    dbg("x is 42")
} else {
    dbg("x is not 42")
}
```
```
DBG: x is not 42
```

### Elif Statements
!!! Not-Implemented
    Elif statements are not yet implemented. 
    What is written below is a rough idea of what they will look like.

Elif statements are used to execute a set of instructions if a condition is true and another set of instructions if the condition is false. To create an elif statement, use the `if` keyword followed by a set of parentheses, a condition, a set of curly braces, the `elif` keyword, another set of parentheses, another condition, another set of curly braces, and the `else` keyword followed by another set of curly braces. For example:

``` rust linenums="1"
let x = 54
if x == 42 {
    dbg("x is 42")
} elif x == 54 {
    dbg("x is 54")
} else {
    dbg("x is not 42 or 54")
}
```
```
DBG: x is 54
```

## Operators
Operators are used to perform operations on values. The following operators are supported by serqlane:

### Binary Operators
Arithmetic operators are used to perform arithmetic operations on values. The following arithmetic operators are supported by serqlane:

- `+`: Addition
- `-`: Subtraction
- `*`: Multiplication
- `/`: Division
- `%`: Modulo

### Comparison Operators
Comparison operators are used to compare values. The following comparison operators are supported by serqlane:

- `==`: Equal to
- `!=`: Not equal to
- `<`: Less than
- `>`: Greater than
- `<=`: Less than or equal to
- `>=`: Greater than or equal to

### Logical Operators
Logical operators are used to perform logical operations on values. The following logical operators are supported by serqlane:

- `and`: Logical and
- `or`: Logical or
- `not`: Logical not

### Unary Operators
!!! Need-help-defining
    The lark file seems to have a mix of potentially wrong definitions
