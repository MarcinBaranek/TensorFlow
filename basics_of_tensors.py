import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # ignore message with dynamic library 'cudart64_110.dll'
from functools import wraps
import tensorflow as tf


def print_tensor(tensor, message):
    print(message + f":\n{tensor}")


def print_title(function):
    @wraps(function)
    def wrap():
        print(60 * "=" + f"\nStart   {function.__name__}\n" + 60 * "=")
        function()
        print(f"End   {function.__name__}\n" + 60 * "_")
    return wrap


@print_title
def initialization_of_tensors():
    tensor = tf.constant(4.0, shape=(1, 1), dtype=tf.float32)
    print_tensor(tensor, "first tensor")
    tensor_matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
    print_tensor(tensor_matrix, "matrix as tensor")
    tensor_of_ones = tf.ones((3, 3))
    print_tensor(tensor_of_ones, "tensor of ones")
    tensor_of_zeros = tf.zeros((2, 3))
    print_tensor(tensor_of_zeros, "tensor of zeros")
    tensor_eye = tf.eye(3)   # identity matrix
    print_tensor(tensor_eye, "identity tensor")
    random_tensor = tf.random.normal(shape=(3, 3), mean=0.1, stddev=1.0)
    print_tensor(random_tensor, "random tensor")
    random_uniform_tensor = tf.random.uniform((1, 3), minval=0.0, maxval=1.5)
    print_tensor(random_uniform_tensor, "random uniform tensor")
    range_tensor = tf.range(start=1, limit=6, delta=2)
    print_tensor(range_tensor, "range tensor from 1 to 6 with delta 2")


@print_title
def mathematical_operation():
    _x, _y = tf.constant([1, 2, 3]), tf.constant([9, 8, 7])
    add = tf.add(_x, _y)    # can be write as _x + _y
    print_tensor(add, "results of add _x and _y")
    subtract = tf.subtract(_x, _y)  # can be write as _x - _y
    print_tensor(subtract, "results of subtract _x and _y")
    dot = tf.tensordot(_x, _y, axes=1)
    print_tensor(dot, "scalar dot _x and _y along axes 1")
    reduce_sum = tf.reduce_sum(_x * _y, axis=0)
    print_tensor(reduce_sum, "dot product obtained by a different route")
    power = _x ** 5     # can be write as tf.pow(_x, 5)
    print_tensor(power, "_x to the power 5")
    _x = tf.random.normal((2, 3))
    _y = tf.random.normal((3, 4))
    tensor_multiply = tf.matmul(_x, _y)     # can be write as _x @ _y
    print_tensor(tensor_multiply, "multiply of _x and _y")


@print_title
def indexing():
    tensor = tf.range(1, 6)
    print_tensor(tensor, "Main tensor")
    print_tensor(tensor[::2], "tensor with every second element")
    indices = tf.constant([0, 3])
    print_tensor(tf.gather(tensor, indices=indices), "tensor of elements with index 0 and 3")
    matrix = tf.constant([[1, 2],
                          [3, 4],
                          [5, 6]])
    print_tensor(matrix[0, :], "first row of matrix")
    print_tensor(matrix[0:2, :], "first 2 row of matrix")


if __name__ == '__main__':
    initialization_of_tensors()
    mathematical_operation()
    indexing()
