def fibonacci(): # generator
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()

print(next(fib))
