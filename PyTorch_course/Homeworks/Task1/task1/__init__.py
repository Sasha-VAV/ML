def my_sum(my_a: float, my_b: float) -> float:
    return my_a + my_b


a, b = map(float, input("Enter a and b\n").split())
print(f"Sum of a and b = {a+b}")
