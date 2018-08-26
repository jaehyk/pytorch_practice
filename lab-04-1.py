import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = Variable(torch.Tensor(x_data).cuda())
Y = Variable(torch.Tensor(y_data).cuda())

model = nn.Linear(3, 1, bias=True).cuda()

loss_fn = nn.MSELoss()

# Minimize
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model
for step in range(2001):
    optimizer.zero_grad()
    # Our hypothesis
    hypothesis = model(X).cuda()
    cost = loss_fn(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, "Cost: ", cost.data.cpu().numpy(), "\nPrediction:\n", hypothesis.data.cpu().numpy())
