import scipy.io as sio
import numpy as np
import math

def fullyconnect_feedforward(in_, weight, bias):
    '''
    The feedward process of fullyconnect
      input parameters:
          in_     : the intputs, shape: [number of images, number of inputs]
          weight  : the weight matrix, shape: [number of inputs, number of outputs]
          bias    : the bias, shape: [number of outputs, 1]

      output parameters:
          out     : the output of this layer, shape: [number of images, number of outputs]
    '''
    # TODO
    # begin answer
    out = np.dot(in_, weight) + bias.T
    # end answer
    return out

def relu_feedforward(in_):
    '''
    The feedward process of relu
      in_:
              in_	: the input, shape: any shape of matrix
      
      outputs:
              out : the output, shape: same as in
    '''
    # TODO
    # begin answer
    out = np.maximum(0, in_)
    # end answer
    return out

def relu_backprop(in_sensitivity, in_):
    '''
    The backpropagation process of relu
      input paramter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: same as in_sensitivity
      
      output paramter:
          out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity
    '''
    # TODO

    # begin answer
    out_sensitivity = in_sensitivity * (in_ > 0)
    # end answer
    return out_sensitivity

def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          out_sensitivity : the sensitivity to the lower layer, shape: 
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO
    # begin answer
    weight_grad = np.dot(in_.T, in_sensitivity) / in_.shape[0]
    bias_grad = np.mean(in_sensitivity, axis=0, keepdims=True).T  # Transpose added here
    out_sensitivity = np.dot(in_sensitivity, weight.T)
    # end answer

    return weight_grad, bias_grad, out_sensitivity

def softmax_loss(in_, label):
    '''
    The softmax loss computing process
      inputs:
          in_     : the output of previous layer, shape: [number of images, number of kinds of labels]
          label   : the ground true of these images, shape: [1, number of images]

      outputs
          loss    : the average loss, scale variable
          accuracy: the accuracy of the classification
          sentivity     : the sentivity for in, shape: [number of images, number of kinds of labels]
    '''
    n, k = in_.shape
    in_ = in_ - np.tile(np.max(in_, axis=1, keepdims=True), (1, k))
    h = np.exp(in_)
    total = np.sum(h, axis=1, keepdims=True)
    probs = h / np.tile(total, k)
    idx = (np.arange(n), label.flatten() - 1)
    loss = -np.sum(np.log(probs[idx])) / n
    max_idx = np.argmax(probs, axis=1)

    accuracy = np.sum(max_idx == (label - 1).flatten()) / n
    sensitivity = np.zeros((n, k))
    sensitivity[idx] = -1
    sensitivity = sensitivity + probs
    return loss, accuracy, sensitivity

def feedforward_backprop(data, label, weights):

    # feedforward hidden layer and relu
    fully1_out = fullyconnect_feedforward(data, weights['fully1_weight'], weights['fully1_bias']);
    relu1_out = relu_feedforward(fully1_out)

    # softmax loss (probs = e^(w*x+b) / sum(e^(w*x+b))) is implemented in two parts for convenience.
    # first part: y = w * x + b is a fullyconnect.
    fully2_out = fullyconnect_feedforward(relu1_out, weights['fully2_weight'], weights['fully2_bias'])
    # second part: probs = e^y / sum(e^y) is the so-called softmax_loss here.
    loss, accuracy, fully2_sensitivity = softmax_loss(fully2_out, label)
    gradients = {}
    gradients['fully2_weight_grad'], gradients['fully2_bias_grad'], relu1_sensitivity = fullyconnect_backprop(fully2_sensitivity, relu1_out, weights['fully2_weight'])
    # backprop of relu and then hidden layer 
    fully1_sensitivity = relu_backprop(relu1_sensitivity, fully1_out)
    gradients['fully1_weight_grad'], gradients['fully1_bias_grad'], _ = fullyconnect_backprop(fully1_sensitivity, data, weights['fully1_weight'])

    return loss, accuracy, gradients

def get_new_weight_inc(weight_inc, weight, momW, wc, lr, weight_grad):
    '''
    Get new increment weight, the update weight policy.
      inputs:
              weight_inc:     old increment weights
              weight:         old weights
              momW:           weight momentum
              wc:             weight decay
              lr:             learning rate
              weight_grad:    weight gradient

      outputs:
              weight_inc:   new increment weights
    '''
    
    weight_inc = momW * weight_inc - wc * lr * weight - lr * weight_grad

    return weight_inc

digit_data = sio.loadmat('digit_data.mat')
X = digit_data['X']
y = digit_data['y']
_, num_cases = X.shape
X = X.reshape((400, num_cases))
X = X.transpose() # X has the shape of (number of samples, number of pixels)、
train_num_cases = num_cases * 4 // 5
train_data = X[:train_num_cases,:]
train_label = y[:, :train_num_cases]
test_data = X[train_num_cases:, :]
test_label = y[:, train_num_cases:]

weights = {}
weights['fully1_weight'] = np.random.randn(400, 25) / 400
weights['fully1_bias'] = np.random.rand(25, 1) 
weights['fully2_weight'] = np.random.randn(25, 10) / 25
weights['fully2_bias'] = np.random.rand(10, 1)

# training setting
weight_inc = {}
for name in ('fully1_weight', 'fully1_bias', 'fully2_weight', 'fully2_bias'):
    weight_inc[name] = np.zeros(weights[name].shape)
batch_size = 100
max_epoch = 10
momW = 0.9
wc = 0.0005
learning_rate = 0.1 / batch_size

# Training iterations
for epoch in range(max_epoch):
    for i in range(math.ceil(train_num_cases/batch_size)):
        data = train_data[i * batch_size:min((i + 1) * batch_size, train_num_cases), :]
        label = train_label[:, i * batch_size:min((i + 1) * batch_size, train_num_cases)]
        # The feedforward and backpropgation processes
        loss, accuracy, gradients = feedforward_backprop(data, label, weights)
        print('{:3}.{:2} loss:{:.3}, accuracy:{}'.format(epoch + 1, i + 1, loss, accuracy))
        # Updating weights
        for name in ('fully1_weight', 'fully1_bias', 'fully2_weight', 'fully2_bias'):
            weight_inc[name] = get_new_weight_inc(weight_inc[name], weights[name], momW, wc, learning_rate, gradients[name + '_grad'])
            weights[name] += weight_inc[name]

loss, accuracy, _ = feedforward_backprop(test_data, test_label, weights)
print('loss:{:.3}, accuracy:{}'.format(loss, accuracy))