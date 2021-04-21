import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class CAEModel:
    def __init__(self, nn_module, **kwargs):
        self.dtype = kwargs.get('dtype', torch.float32)
        self.device = kwargs.get('device', torch.device('cuda'))
        self.nn_module = nn_module.to(self.device)
        self.optim = torch.optim.Adam(self.nn_module.parameters(), lr=1e-3)
        self.losses_epoch = []
    
    def train(self, dataset, n_epochs, verbose=False, **kwargs):
        self.nn_module.train()
        dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
        for epoch in tqdm(range(n_epochs)):
            losses_batch = []
            for x, y in dataloader:
                self.optim.zero_grad()
                x = x.to(self.device)
                x_pred = self.nn_module(x)
                loss_mse = torch.mean((x - x_pred)**2)
                losses_batch.append(float(loss_mse))
                loss_mse.backward()
                self.optim.step()
            loss_epoch = sum(losses_batch)/len(losses_batch)
            self.losses_epoch.append(loss_epoch)
            if verbose:
                print(loss_epoch)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.nn_module.load_state_dict(checkpoint['nn_module'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.losses_epoch = checkpoint['losses_epoch']
    
    def save(self, path):
        torch.save({
            'nn_module': self.nn_module.state_dict(),
            'optim': self.optim.state_dict(),
            'losses_epoch': self.losses_epoch,
        }, path)

    def evaluate(self, datasets, **kwargs):
        self.nn_module.eval()
        outputs = dict()
        for dataset in datasets:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, **kwargs)
            for x, y in dataloader:
                    x = x.to(self.device)
                    x_pred = self.nn_module(x)
                    mse = torch.mean((x - x_pred)**2, dim=[1, 2, 3])
                    y = int(y)
                    if y not in outputs.keys():
                        outputs[y] = []
                    outputs[y].append({'x':x.cpu().detach().numpy()[0,:,:,:],
                                       'x_pred':x_pred.cpu().detach().numpy()[0,:,:,:],
                                       'mse':float(mse.cpu().detach())})
        return outputs
    
    def visualize(self, dataset, n):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
        dataiter = iter(dataloader)
        x, y = dataiter.next()
        x = x.to(self.device)
        x_pred = self.nn_module(x)

        mses = []
        for i in range(n):
            mse = torch.mean((x[i] - x_pred[i])**2)
            mses.append(float(mse))

        x = x.to('cpu').numpy()
        x_pred = x_pred.to('cpu').detach().numpy()

        fig, axes = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, figsize=(25,4))

        for j in range(n):
            axes[0][j].imshow(x[j][0], cmap='gray')
            axes[1][j].imshow(x_pred[j][0], cmap='gray')
            axes[1][j].set_xlabel(str(round(mses[j], 3)))