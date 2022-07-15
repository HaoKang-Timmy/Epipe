import torch
import time

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class BaseTrainer:
    def __init__(self, model, criterion, metric, opt, train_loader, val_loader, args):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        
class TrainerCV(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric,
        opt,
        train_loader,
        val_loader,
        lr_scheduler,
        device,
        train_sampler,
        args,
    ):
        super().__init__(model, criterion, metric, opt, train_loader, val_loader, args)
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.train_sampler = train_sampler
    #in order to speed up training processes, we do not do evaluation here.
    def train(self):
        args = self.args
        train_loader = self.train_loader
        model = self.model
        criterion = self.criterion
        metric = self.metric
        optimizer = self.optimizer
        device = self.device
        lr_scheduler = self.lr_scheduler

        train_sampler = self.train_sampler
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            train_loss = 0.0
            train_acc1 = 0.0
            time_avg = 0.0
            start = time.time()
            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                outputs = model(image)
                loss = criterion(outputs, label)
                acc, _ = metric(outputs, label, topk=(1, 2))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                train_acc1 += acc.item()
                end = time.time() - start
                time_avg += end
                if i % args.showperiod == 0 and device == 0:
                    print(
                        "train_loss", loss.item(), "train_acc", acc.item(), "time", end
                    )
                start = time.time()
            train_loss /= len(train_loader)
            train_acc1 /= len(train_loader)
            time_avg /= len(train_loader)
            lr_scheduler.step()
            
            with torch.no_grad():
                
                if device == 0:
                    print("lr:", get_lr(optimizer))
                    print(
                        "epoch:",
                        epoch,
                        "train_loss",
                        train_loss,
                        "train_acc"
                    )
                    for i, conv in enumerate(model.module.convsets):
                            torch.save(
                            conv.state_dict(),
                            args.savepath + str(args.type) + "conv" + str(i) + ".pth",
                        )
                    file_save = open(args.log, mode="a")
                    file_save.write(
                        "\n"
                        + "step:"
                        + str(epoch)
                        + "  loss_train:"
                        + str(train_loss)
                        + "  acc1_train:"
                        + str(train_acc1)
                        + "  time_per_batch:"
                        + str(time_avg)
                        + "  lr:"
                        + str(get_lr(optimizer))
                    )
                    file_save.close()
    def traineval(self):
        args = self.args
        train_loader = self.train_loader
        model = self.model
        criterion = self.criterion
        metric = self.metric
        optimizer = self.optimizer
        device = self.device
        lr_scheduler = self.lr_scheduler
        val_loader = self.val_loader
        train_sampler = self.train_sampler
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            train_loss = 0.0
            train_acc1 = 0.0
            time_avg = 0.0
            start = time.time()
            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                outputs = model(image)
                loss = criterion(outputs, label)
                acc, _ = metric(outputs, label, topk=(1, 2))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                train_acc1 += acc.item()
                end = time.time() - start
                time_avg += end
                if i % args.showperiod == 0 and device == 0:
                    print(
                        "train_loss", loss.item(), "train_acc", acc.item(), "time", end
                    )
                start = time.time()
            train_loss /= len(train_loader)
            train_acc1 /= len(train_loader)
            time_avg /= len(train_loader)
            lr_scheduler.step()
            model.eval()
            if device == 0:
                print("lr:", get_lr(optimizer))
            val_loss = 0.0
            val_acc1 = 0.0
            with torch.no_grad():
                for i, (image, label) in enumerate(val_loader):
                    image = image.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    outputs = model(image)
                    loss = criterion(outputs, label)
                    acc, _ = metric(outputs, label, topk=(1, 2))

                    val_loss += loss.item()
                    val_acc1 += acc.item()
                    if i % 20 == 0 and device == 0:
                        print("val_loss", loss.item(), "val_acc", acc.item())
                val_loss /= len(val_loader)
                val_acc1 /= len(val_loader)
                if device == 0:
                    print(
                        "epoch:",
                        epoch,
                        "train_loss",
                        train_loss,
                        "train_acc",
                        train_acc1,
                        "val_loss",
                        val_loss,
                        "val_acc",
                        val_acc1,
                    )
                    file_save = open(args.log, mode="a")
                    file_save.write(
                        "\n"
                        + "step:"
                        + str(epoch)
                        + "  loss_train:"
                        + str(train_loss)
                        + "  acc1_train:"
                        + str(train_acc1)
                        + "  loss_val:"
                        + str(val_loss)
                        + "  acc1_val:"
                        + str(val_acc1)
                        + "  time_per_batch:"
                        + str(time_avg)
                        + "  lr:"
                        + str(get_lr(optimizer))
                    )
                    file_save.close()