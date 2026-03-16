# TODO: import dependencies and write unit tests below
import pytest
import numpy as np
from numpy.testing import assert_allclose
from nn.nn import NeuralNetwork
from nn.preprocess import (sample_seqs, one_hot_encode_seqs)

test_seed = 1
test_lr = 0.01

def test_single_forward():
    # simple example nn and weights/biases/A_prev matrices
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":2,"output_dim":2,"activation":"relu"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    W = np.array([[1,2],
                  [3,4]])
    b = np.array([[1],
                  [1]])
    A_prev = np.array([[1,1],
                       [2,2]])
    

    # compare output of _single_forward against externally computed Z and A
    A, Z = test_nn._single_forward(W,b,A_prev,"relu")

    expected_Z = A_prev @ W.T + b.T
    expected_A = np.maximum(0, expected_Z)

    assert_allclose(Z, expected_Z) # also implicitly checking shape/format of Z and A
    assert_allclose(A, expected_A)

def test_forward():
    # test architecture that uses both of the currently implemented activation functions
    test_arch = [
        {"input_dim":2,"output_dim":3,"activation":"relu"},
        {"input_dim":3,"output_dim":1,"activation":"sigmoid"}
    ]
    test_nn = NeuralNetwork(
        nn_arch=test_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    X = np.array([[1,2],
                  [3,4]])

    # assert that output has correct shape and contains all expected cached matrices
    output, cache = test_nn.forward(X)
    assert output.shape == (test_arch[0]["input_dim"],test_arch[-1]["output_dim"])

    for l in range(1, len(test_arch)+1):
        assert f"A{l}" in cache
        assert f"Z{l}" in cache

    # test that model catches invalid activation functions and throws error
    failed_arch = [
        {"input_dim":2,"output_dim":3,"activation":"unknown_activation_function"},
        {"input_dim":3,"output_dim":1,"activation":"sigmoid"}
    ]
    bad_nn = NeuralNetwork(
        nn_arch=failed_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    with pytest.raises(ValueError):
        bad_nn.forward(X)

def test_single_backprop():
    # simple example nn with appropriate input matrices
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":2,"output_dim":2,"activation":"relu"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    W = np.random.randn(3,2)
    b = np.random.randn(3,1)
    A_prev = np.random.randn(4,2)
    Z = np.random.randn(4,3)
    dA = np.random.randn(4,3)

    # run single backprop and assert that partial deriv matrices are same shape as their respective inputs
    dA_prev, dW, db = test_nn._single_backprop(W,b,Z,A_prev,dA,"relu")

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

def test_backprop():
    # test architecture with both activation functions
    test_arch = [
        {"input_dim":2,"output_dim":3,"activation":"relu"},
        {"input_dim":3,"output_dim":1,"activation":"sigmoid"}
    ]
    test_nn = NeuralNetwork(
        nn_arch=test_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    # simple input and target
    X = np.array([[1,2],
                  [3,4]])
    y = np.array([[1],[0]])

    # run forward pass
    y_hat, cache = test_nn.forward(X)

    # run backprop
    grad_dict = test_nn.backprop(y, y_hat, cache)

    # assert that all expected gradient keys are present and grad dicts are correct shape
    for l in range(1, len(test_arch)+1):
        assert f"dW{l}" in grad_dict
        assert f"db{l}" in grad_dict

        # check shapes
        assert grad_dict[f"dW{l}"].shape == test_nn._param_dict[f"W{l}"].shape
        assert grad_dict[f"db{l}"].shape == test_nn._param_dict[f"b{l}"].shape

    # test that invalid activation and loss functions triggers error
    failed_arch = [
        {"input_dim":2,"output_dim":3,"activation":"unknown_activation_function"},
        {"input_dim":3,"output_dim":1,"activation":"sigmoid"}
    ]
    bad_nn_1 = NeuralNetwork(
        nn_arch=failed_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="mse"
    )
    bad_nn_2 = NeuralNetwork(
        nn_arch=test_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="unknown_loss_function"
    )
    with pytest.raises(ValueError):
        bad_nn_1.backprop(y, y_hat, cache)
    with pytest.raises(ValueError):
        bad_nn_2.backprop(y, y_hat, cache)

def test_predict():
    test_arch = [{"input_dim":2,"output_dim":1,"activation":"sigmoid"}]
    test_nn = NeuralNetwork(
        nn_arch=test_arch,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="bce"
    )
    X = np.array([[1,2],[3,4]])
    y_hat = test_nn.predict(X)

    # assert that y_hat's shape is (m_samples, 1) 
    # and test that sigmoid leaves all predictions between 0 and 1
    assert y_hat.shape == (X.shape[0], 1)
    assert np.all((y_hat >= 0) & (y_hat <= 1))


    test_arch2 = [{"input_dim":2,"output_dim":1,"activation":"relu"}]
    test_nn2 = NeuralNetwork(
        nn_arch=test_arch2,
        lr=test_lr,
        seed=test_seed,
        batch_size=2,
        epochs=1,
        loss_function="bce"
    )
    y_hat2 = test_nn2.predict(X)

    # assert that y_hat's shape is (m_samples, 1) 
    # and test that ReLU leaves all outputs >=0
    assert y_hat2.shape == (X.shape[0], 1)
    assert np.all((y_hat2 >= 0))


def test_binary_cross_entropy():
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":1,"output_dim":1,"activation":"sigmoid"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=1,
        epochs=1,
        loss_function="bce"
    )
    y = np.array([[1],[0]])
    y_hat = np.array([[0.9],[0.2]])

    # externally calculate bce and then compare to output of class function 
    expected = -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    loss = test_nn._binary_cross_entropy(y,y_hat)

    assert_allclose(loss, expected)

    # sanity check: assert that loss is greater for a prediction that is way off
    worse_y_hat = np.array([[0.2],[0.9]])
    worse_loss = test_nn._binary_cross_entropy(y,worse_y_hat)

    assert worse_loss > loss
    

def test_binary_cross_entropy_backprop():
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":1,"output_dim":1,"activation":"sigmoid"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=1,
        epochs=1,
        loss_function="bce"
    )
    y = np.array([[1],[0]])
    y_hat = np.array([[0.9],[0.2]])

    # externally calculate bce backprop and then compare to output of class function 
    expected = -(y/y_hat) + ((1-y)/(1-y_hat))
    dA = test_nn._binary_cross_entropy_backprop(y,y_hat)

    assert_allclose(dA, expected)

    # sanity check: assert that |dA| is greater when prediction that is way off
    worse_y_hat = np.array([[0.2],[0.9]])
    worse_dA = test_nn._binary_cross_entropy_backprop(y,worse_y_hat)

    assert np.all(np.abs(worse_dA) > np.abs(dA))

def test_mean_squared_error():
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":1,"output_dim":1,"activation":"sigmoid"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=1,
        epochs=1,
        loss_function="mse"
    )
    y = np.array([[1],[0]])
    y_hat = np.array([[0.9],[0.2]])

    # externally calculate mse and then compare to output of class function 
    expected = np.mean((y-y_hat)**2)
    loss = test_nn._mean_squared_error(y,y_hat)

    assert_allclose(loss, expected)

    # sanity check: assert that loss is greater for a prediction that is way off
    worse_y_hat = np.array([[0.2],[0.9]])
    worse_loss = test_nn._mean_squared_error(y,worse_y_hat)

    assert worse_loss > loss

def test_mean_squared_error_backprop():
    test_nn = NeuralNetwork(
        nn_arch=[{"input_dim":1,"output_dim":1,"activation":"sigmoid"}],
        lr=test_lr,
        seed=test_seed,
        batch_size=1,
        epochs=1,
        loss_function="mse"
    )
    y = np.array([[1],[0]])
    y_hat = np.array([[0.9],[0.2]])

    # externally calculate mse backprop and then compare to output of class function 
    expected = 2*(y_hat-y)
    dA = test_nn._mean_squared_error_backprop(y,y_hat)

    assert_allclose(dA, expected)

    # sanity check: assert that |dA| is greater when prediction that is way off
    worse_y_hat = np.array([[0.2],[0.9]])
    worse_dA = test_nn._mean_squared_error_backprop(y,worse_y_hat)

    assert np.all(np.abs(worse_dA) > np.abs(dA))

def test_sample_seqs():
    # simple example seqs and labels
    seqs = ["A","B","C","D","E"]
    labels = [True,True,True,False,False]

    # run sample_seqs
    sampled_seqs, sampled_labels = sample_seqs(seqs,labels)

    # assert that the number of samples matches between classes
    pos = sum(sampled_labels)
    neg = len(sampled_labels) - pos

    assert pos == neg

    # assert that labels are returned for all seqs
    assert len(sampled_seqs) == len(sampled_labels)

def test_one_hot_encode_seqs():
    seqs = ["AT","TA"]
    encodings = one_hot_encode_seqs(seqs)

    len_encoding = len({char for seq in seqs for char in seq})
    len_seq = len(seqs[0])

    # assert that each seq has an encoding 
    # AND each encoding is the length of one encoded char times number of chars
    assert encodings.shape == (len(seqs),len_encoding*len_seq)

    # ensure one-hot property (every value is either a 0 or 1)
    assert np.all((encodings == 0) | (encodings == 1))

    # each character should have exactly one 1, leading to row sums of the length of each sequence
    assert np.all(encodings.sum(axis=1) == len_seq)