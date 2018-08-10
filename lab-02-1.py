import torch

x_train = torch.t(torch.tensor([[1.0,2.0,3.0]]))
y_train = torch.t(torch.tensor([[1.0,2.0,3.0]]))

model = torch.nn.Sequential(
	torch.nn.Linear(1, 1)
)

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for t in range(5001):
	y = model(x_train)

	loss = loss_fn(y, y_train)

	if t % 20 == 0:
		print(t, loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
