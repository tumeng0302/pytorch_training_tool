# pytorch_training_tool

## usage
```
from log_tool import Training_Log

losses_name = ["loss1(must be your total loss)", "loss2", "loss3", "loss4"]
log = Training_Log('<your_project_name>', train_lossesname=losses_name, test_lossesname=losses_name)

for e in epochs:
    # training
    for i in interations:
        ...
        <Your training processes>
        loss2 = loss_func()
        loss3 = loss_func()
        loss4 = loss_func()
        loss1 = loss2 + loss3 + loss4
        ...

        log.train_loss.push_loss([
            loss1.item(),
            loss2.item(),
            loss3.item(),
            loss4.item()
            ])
        #if you want to print loss information
        loss_str = log.train_loss.avg_loss()
        print(loss_str)

    # testing
    for i in interations:
        ...
        <Your testing processes>
        loss2 = loss_func()
        loss3 = loss_func()
        loss4 = loss_func()
        loss1 = loss2 + loss3 + loss4
        ...

        log.test_loss.push_loss([
            loss1.item(),
            loss2.item(),
            loss3.item(),
            loss4.item()
            ])
        #if you want to print loss information
        loss_str = log.test_loss.avg_loss()
        print(loss_str)
        # Your result image with shape <n, c, h, w>
        img = net(data)
    log.step(save_img, net.state_dict())

```
