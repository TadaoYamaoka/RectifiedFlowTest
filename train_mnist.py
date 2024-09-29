import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from scipy import integrate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from unet import Unet

batch_size = 128
learning_rate = 0.001
num_epochs = 10
eps = 0.001


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(
    dim=32,
    channels=1,
    dim_mults=(1, 2, 4),
)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def euler_sampler(model, shape, sample_N):
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        dt = 1.0 / sample_N
        for i in range(sample_N):
            num_t = i / sample_N * (1 - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            pred = model(x, t * 999)

            x = x.detach().clone() + pred * dt

        nfe = sample_N
        return x.cpu(), nfe


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def rk45_sampler(model, shape):

    rtol = atol = 1e-05
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t * 999)

            return to_flattened_numpy(drift)

        solution = integrate.solve_ivp(
            ode_func,
            (eps, 1),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)

        return x, nfe


def imshow(img, filename):
    img = img * 0.3081 + 0.1307
    img = np.clip(img, 0, 1)
    npimg = img.numpy()
    plt.imshow(npimg[0], cmap="gray")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)


def save_img_grid(img, filename):
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    imshow(img_grid, filename)


for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch, _ in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        z0 = torch.randn_like(batch)
        t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps

        t_expand = t.view(-1, 1, 1, 1).repeat(
            1, batch.shape[1], batch.shape[2], batch.shape[3]
        )
        perturbed_data = t_expand * batch + (1 - t_expand) * z0
        target = batch - z0

        score = model(perturbed_data, t * 999)

        losses = torch.square(score - target)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    images, nfe = euler_sampler(model, shape=(100, 1, 28, 28), sample_N=1)
    save_img_grid(images, f"euler_epoch_{epoch + 1}_nfe_{nfe}.png")

    images, nfe = euler_sampler(model, shape=(100, 1, 28, 28), sample_N=2)
    save_img_grid(images, f"euler_epoch_{epoch + 1}_nfe_{nfe}.png")

    images, nfe = euler_sampler(model, shape=(100, 1, 28, 28), sample_N=10)
    save_img_grid(images, f"euler_epoch_{epoch + 1}_nfe_{nfe}.png")

    images, nfe = rk45_sampler(model, shape=(100, 1, 28, 28))
    save_img_grid(images, f"rk45_epoch_{epoch + 1}_nfe_{nfe}.png")
