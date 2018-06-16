def check():
  '''
  Compare in terms of results our implementation with PyTorch.
  '''

  #Threshold for numerical errors when comparing float tensors.
  threshold = 1.e-5


  import torch
  from torch import nn
  from torch.autograd import Variable
  from torch import autograd
  from modules.activation import ReLU, Tanh
  from modules.linear import Linear
  from modules.loss import MSELoss, MAELoss
  from modules.container import Sequential
  from modules.optimizer import SGD


  print('Asserting ReLU forward and backward phase. ', end='')
  a = torch.randn(3,2) - 0.5
  b = torch.randn(3,2)
  va = Variable(a, requires_grad=True)

  #forward
  relu_forward = nn.ReLU()(va)
  my_relu = ReLU()
  my_relu_forward = my_relu(a)

  #backward
  relu_backward = autograd.grad(outputs=relu_forward, inputs=va, grad_outputs=b)[0]
  my_relu_backward = my_relu.backward(b)

  assert(torch.sum(relu_forward.data - my_relu_forward) < threshold)
  assert(torch.sum(relu_backward.data - my_relu_backward) <threshold)
  print('DONE')

  print('Asserting Tanh forward and backward phase. ', end='')
  a = torch.randn(3,2) - 0.5
  b = torch.randn(3,2)
  va = Variable(a, requires_grad=True)

  #forward
  tanh_forward = nn.Tanh()(va)
  my_tanh = Tanh()
  my_tanh_forward = my_tanh(a)

  #backward
  tanh_backward = autograd.grad(outputs=tanh_forward, inputs=va, grad_outputs=b)[0]
  my_tanh_backward = my_tanh.backward(b)

  assert(torch.sum(tanh_forward.data - my_tanh_forward) < threshold)
  assert(torch.sum(tanh_backward.data - my_tanh_backward) <threshold)
  print('DONE')


  print('Asserting Linear forward and backward phase. ', end='')
  a = torch.randn(3,2) - 0.5
  va = Variable(a, requires_grad=True)
  c = torch.rand(3,4)
  w = torch.rand(4,2)
  b = torch.rand(4)

  #forward
  linear = nn.Linear(2,4)
  linear.weight.data = w.clone()
  linear.bias.data = b.clone()
  linear_forward = linear(va)
  my_linear = Linear(2,4)
  my_linear.params[0]['value'] = w
  my_linear.params[1]['value'] = b
  my_linear_forward = my_linear(a)

  #backward
  linear_backward = autograd.grad(outputs=linear_forward, inputs=va, grad_outputs=c, only_inputs=False)[0]
  my_linear_backward = my_linear.backward(c)

  assert(torch.sum(linear.weight.data - my_linear.params[0]['value']) < threshold)
  assert(torch.sum(linear.bias.data - my_linear.params[1]['value']) < threshold)
  assert(torch.sum(linear.weight.grad.data - my_linear.params[0]['grad']) < threshold)
  assert(torch.sum(linear.bias.grad.data - my_linear.params[1]['grad']) < threshold)
  assert(torch.sum(linear_forward.data - my_linear_forward) < threshold)
  assert(torch.sum(linear_backward.data - my_linear_backward) < threshold)
  print('DONE')

  print('Asserting MSE forward and backward phase. ', end='')
  y = torch.rand(1000)
  y_ = torch.rand(1000) - 0.5
  vy = Variable(y, requires_grad=True)
  vy_ = Variable(y_, requires_grad=False)
  c = torch.rand(1)

  #forward
  mse = nn.MSELoss()
  loss_forward = mse(vy, vy_)

  my_mse = MSELoss(y, y_)
  my_loss_forward = my_mse()

  #backward
  loss_backward = autograd.grad(outputs=loss_forward, inputs=vy, grad_outputs=c, only_inputs=False)[0]
  my_loss_backward = my_mse.backward()

  assert(torch.sum(loss_forward.data - my_loss_forward) < threshold)
  assert(torch.sum(loss_backward.data - my_loss_backward) < threshold)
  print('DONE')


  print('Asserting MAE forward and backward phase. ', end='')
  y = torch.rand(1000)
  y_ = torch.rand(1000) - 0.5
  vy = Variable(y, requires_grad=True)
  vy_ = Variable(y_, requires_grad=False)
  c = torch.rand(1)

  #forward
  mae = nn.L1Loss()
  loss_forward = mae(vy, vy_)

  my_mae = MAELoss(y, y_)
  my_loss_forward = my_mae()

  #backward
  loss_backward = autograd.grad(outputs=loss_forward, inputs=vy, grad_outputs=c, only_inputs=False)[0]
  my_loss_backward = my_mae.backward()

  assert(torch.sum(loss_forward.data - my_loss_forward) < threshold)
  assert(torch.sum(loss_backward.data - my_loss_backward) < threshold)
  print('DONE')


  print('Asserting Sequential Container. ', end='')
  y = torch.rand(1000)
  vy = Variable(y, requires_grad=True)

  model = nn.Sequential(
            nn.ReLU(),
            nn.Tanh()
          )

  my_model = Sequential(
            ReLU(),
            Tanh()
          )

  y_ = model(vy)
  my_y_ = my_model(y)

  assert(torch.sum(y_.data - my_y_) < threshold)
  print('DONE')


  print('Asserting SGD step parameter update. ', end='')

  learning_rate = 1.e-5
  weight_decay = 0.1

  w = torch.rand(4,2)
  b = torch.rand(4)
  vw = Variable(w.clone(), requires_grad=True)
  vb = Variable(b.clone(), requires_grad=True)


  sgd = torch.optim.SGD([vw,vb], lr=learning_rate, weight_decay=weight_decay) 
  my_sgd = SGD([{'value':w, 'grad':w}, {'value':b, 'grad':b}], lr=learning_rate, weight_decay=weight_decay) 

  sgd.zero_grad()
  my_sgd.zero_grad()

  sgd.step()
  my_sgd.step()

  assert(torch.sum(w - vw.data) < threshold)
  assert(torch.sum(b - vb.data) < threshold)
  print('DONE')
