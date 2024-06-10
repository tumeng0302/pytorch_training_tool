from log_tool import Training_Log
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from tqdm import tqdm
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, split='train'):
        self.data = torch.randn(100, 10)
        self.target = torch.randn(100, 5)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def main():
    log = Training_Log(model_name='MyModel')
    torch.cuda.set_device(log.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log.compile:
        model = torch.compile(MyModel().to(device))
    else:
        model = MyModel().to(device)

    if log.resume is not None:
        model.load_state_dict(torch.load(log.resume))

    train_set = MyDataset()
    train_loader = DataLoader(train_set, batch_size=log.batch, shuffle=True, 
                            num_workers=log.num_workers)
    
    test_set = MyDataset(split='test')
    test_loader = DataLoader(test_set, batch_size=24, shuffle=True, 
                                 num_workers=log.num_workers)
    
    val_set = MyDataset(split='val')
    val_loader = DataLoader(val_set, batch_size=24, shuffle=True, 
                                 num_workers=log.num_workers)
    
    if log.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=log.lr)
    else:
        optimizer = SGD(model.parameters(), lr=log.lr)

    def train_loss_fn(pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        l1_loss = nn.L1Loss()(pred, target)
        total_loss = mse_loss + l1_loss
        return total_loss, mse_loss, l1_loss
    
    def test_loss_fn(pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        return mse_loss, mse_loss
    
    def metrics_fn(pred, target):
        mse = nn.MSELoss()(pred, target)
        return mse
    
    train_losses = ['total_loss', 'mse_loss', 'l1_loss']
    test_losses = ['total_loss', 'mse_loss']
    val_losses = ['total_loss', 'mse_loss']
    metrics = ['mse']

    log.init_loss(train_losses=train_losses, 
                  val_losses=val_losses, 
                  test_losses=test_losses, 
                  metrics=metrics)
    
    for eps in range(log.total_epochs):
        model.train()
        i_bar = tqdm(train_loader, unit='iter', desc=f"epoch {eps+1}")
        for data, target in i_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss, mse_loss, l1_loss = train_loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            log.train_loss.push_loss([
                loss.item(), 
                mse_loss.item(), 
                l1_loss.item()
            ])
            loss_str = log.train_loss.avg_loss()
            i_bar.set_postfix_str(str(f"{loss_str}"))
        
        model.eval()
        t_bar = tqdm(test_loader, unit='iter', desc=f"epoch {eps+1}")
        v_bar = tqdm(val_loader, unit='iter', desc=f"epoch {eps+1}")
        with torch.no_grad():
            for data, target in t_bar:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss, mse_loss = test_loss_fn(pred, target)
                log.test_loss.push_loss([
                    loss.item(),
                    mse_loss.item()
                ])
                loss_str = log.test_loss.avg_loss()
                t_bar.set_postfix_str(str(f"{loss_str}"))
            
            for data, target in v_bar:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss, mse_loss = test_loss_fn(pred, target)
                mse_loss = metrics_fn(pred, target)

                log.val_loss.push_loss([
                    loss.item(),
                    mse_loss.item()
                ])
                log.metrics.push_loss([
                    mse_loss.item()
                ])
                loss_str = log.val_loss.avg_loss()
                metrics_str = log.metrics.avg_loss()
                v_bar.set_postfix_str(str(f"{loss_str}"))
        
        log.step(epochs=eps)


    