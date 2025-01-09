import tensorflow as tf

# Test TensorFlow installation
print("TensorFlow version:", tf.__version__)

# Create a simple computation graph
a = tf.constant(2.0, name="a")
b = tf.constant(3.0, name="b")
c = tf.add(a, b, name="c")

# Run the computation
print("a + b =", c.numpy())

# Test GPU availability
print(tf.config.list_logical_devices())
