# taken from https://github.com/tensorflow/tensorflow/pull/2581/files

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import dynamic_rnn
import tensorflow as tf

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  """Creates a bidirectional recurrent neural network, dynamic version.
  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.
  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, input_size]`.
      If time_major == True, this must be a tensor of shape:
        `[max_time, batch_size, input_size]`.
      [batch_size, input_size].
    sequence_length: An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      `[batch_size x cell_fw.state_size]`.
      If `cell_fw.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using
      the corresponding properties of `cell_bw`.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
      either of the initial states are not provided.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"
  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: The RNN output `Tensor`.
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
      output_state_fw is the final state of the forward rnn.
      output_state_bw is the final state of the backward rnn.
  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")

  name = scope or "BiRNN"
  # Forward direction
  with vs.variable_scope(name + "_FW") as fw_scope:
    output_fw, output_state_fw = dynamic_rnn(
        cell_fw, inputs, sequence_length, initial_state_fw, dtype,
        parallel_iterations, swap_memory, time_major, scope=fw_scope)
  # Backward direction
  if not time_major:
    time_dim = 1
    batch_dim = 0
  else:
    time_dim = 0
    batch_dim = 1
  with vs.variable_scope(name + "_BW") as bw_scope:
    inputs_reverse = array_ops.reverse_sequence(
        inputs, tf.cast(sequence_length, tf.int64), time_dim, batch_dim)
    tmp, output_state_bw = dynamic_rnn(
        cell_bw, inputs_reverse, sequence_length, initial_state_bw, dtype,
        parallel_iterations, swap_memory, time_major, scope=bw_scope)
  output_bw = array_ops.reverse_sequence(
      tmp, tf.cast(sequence_length, tf.int64), time_dim, batch_dim)
  # Concat each of the forward/backward outputs
  outputs = array_ops.concat(2, [output_fw, output_bw])

  return (outputs, output_state_fw, output_state_bw)