from part2_cnns.solutions import Linear, ResNet34
from part3_optimization.solutions import Linear, ResNet34, WandbResNetFinetuningArgs
from dataclasses import dataclass
from broadcast import broadcast
from reduce import all_reduce
from torchvision import datasets, transforms

from pathlib import Path
import sys
# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from torch import Tensor
from tqdm import tqdm

import time
from dotenv import load_dotenv
import os

#wandb login
load_dotenv()
wandb.login(key=os.getenv("WANDB_KEY"))

def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Returns CIFAR-10 train and test sets."""
    cifar_trainset = datasets.CIFAR10(exercises_dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(exercises_dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM)
    return cifar_trainset, cifar_testset


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def get_untrained_resnet(n_classes: int) -> ResNet34:
    """Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your implementation)."""
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        if self.args.world_size > 1:
            for param in self.model.parameters():
                broadcast(param.data, self.rank, self.args.world_size, src=0)
        
        self.optimiser = t.optim.AdamW(
            self.model.parameters(), self.args.learning_rate, weight_decay=self.args.weight_decay
        )

        self.train_set, self.test_set = get_cifar()
        self.train_sampler = self.test_sampler = None
        if self.args.world_size > 1:
            self.train_sampler = DistributedSampler(self.train_set, self.args.world_size, self.rank)
            self.test_sampler = DistributedSampler(self.test_set, self.args.world_size, self.rank)

        dataloader_shared_kwargs = dict(batch_size=self.args.batch_size, num_workers=8, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, sampler=self.train_sampler, **dataloader_shared_kwargs)
        self.test_loader = DataLoader(self.test_set, sampler=self.test_sampler, **dataloader_shared_kwargs)

        self.examples_seen = 0

        if self.rank == 0:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)


    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)

        t1 = time.time()


        loss = F.cross_entropy(logits, labels)
        loss.backward()

        t2 = time.time()

        if self.args.world_size > 1:
            for param in self.model.parameters():
                all_reduce(param.grad, self.rank, self.args.world_size, op="mean")
        t3 = time.time()

        self.optimiser.step()
        self.optimiser.zero_grad()

        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log(
                {"loss": loss.item(), "fwd_time": (t1 - t0), "bwd_time": (t2 - t1), "dist_time": (t3 - t2)},
                step=self.examples_seen,
            )
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_seen = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)

            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.shape[0]
        
                
        # Turn total_correct & total_seen into a tensor, so we can use all_reduce to sum them across processes
        tensor = t.tensor([total_correct, total_seen], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, op="sum")
        total_correct, total_seen = tensor.tolist()

        accuracy = total_correct / total_seen
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t0 = time.time()

            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen=:06}")


            self.model.eval()

            accuracy = self.evaluate()

            if self.rank == 0:
                wandb.log({"epoch_duration": time.time() - t0}, step=self.examples_seen)
                pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.3f}", ex_seen=f"{self.examples_seen=:06}")

        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")

            


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()

@dataclass
class DistResNetTrainingDDPArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training-ddp"

from torch.nn.parallel import DistributedDataParallel as DDP

class DistResNetTrainerDDP:
    args: DistResNetTrainingDDPArgs

    def __init__(self, args: DistResNetTrainingDDPArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = DDP(get_untrained_resnet(self.args.n_classes).to(self.device), device_ids=[self.rank])

        
        self.optimiser = t.optim.AdamW(
            self.model.parameters(), self.args.learning_rate, weight_decay=self.args.weight_decay
        )

        self.train_set, self.test_set = get_cifar()
        self.train_sampler = self.test_sampler = None
        if self.args.world_size > 1:
            self.train_sampler = DistributedSampler(self.train_set, self.args.world_size, self.rank)
            self.test_sampler = DistributedSampler(self.test_set, self.args.world_size, self.rank)

        dataloader_shared_kwargs = dict(batch_size=self.args.batch_size, num_workers=8, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, sampler=self.train_sampler, **dataloader_shared_kwargs)
        self.test_loader = DataLoader(self.test_set, sampler=self.test_sampler, **dataloader_shared_kwargs)

        self.examples_seen = 0

        if self.rank == 0:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)


    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)

        t1 = time.time()


        loss = F.cross_entropy(logits, labels)
        loss.backward()

        t2 = time.time()


        self.optimiser.step()
        self.optimiser.zero_grad()

        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log(
                {"loss": loss.item(), "fwd_time": (t1 - t0), "bwd_time": (t2 - t1)},
                step=self.examples_seen,
            )
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_seen = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)

            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.shape[0]
        
                
        # Turn total_correct & total_seen into a tensor, so we can use all_reduce to sum them across processes
        tensor = t.tensor([total_correct, total_seen], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, op="sum")
        total_correct, total_seen = tensor.tolist()

        accuracy = total_correct / total_seen
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t0 = time.time()

            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen=:06}")


            self.model.eval()

            accuracy = self.evaluate()

            if self.rank == 0:
                wandb.log({"epoch_duration": time.time() - t0}, step=self.examples_seen)
                pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.3f}", ex_seen=f"{self.examples_seen=:06}")

        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"/workspace/ARENA-workthrough/chapter0_fundamentals/exercises/part3_optimization/model/resnet_{self.rank}_DDP.pth")




def dist_train_resnet_from_scratch_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingDDPArgs(world_size=world_size)
    trainer = DistResNetTrainerDDP(args, rank)
    trainer.train()
    dist.destroy_process_group()

    