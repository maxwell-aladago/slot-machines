from argparse import ArgumentParser
from os import path

import torch
from dataset_utils import get_dataset
from model import get_model
from os_utils import OSUtils
from save_slot_checkpoint import save_slot
from train import train
from training_schedule import schedule, get_optimizer


def main(args):
    args.path = f"{args.base_path}/{args.dataset}/{args.model_type}/{args.seed}"
    model = get_model(args.model_type, args.method, args.bn)

    if args.method == 'learned':
        model = model.weight_updates()
        print(f"training {args.model_type} via weights updates")
        try:
            if args.use_checkpoint:
                print("loading state", args.finetune_epoch)
                state = torch.load(f"{args.path}/greedy/checks/subnet_{args.k}_{args.finetune_epoch}.pt")
                args.path = f"{args.path}/ft-{args.finetune_epoch}-{args.lr}"
            else:
                print("loading initial weights")
                state = torch.load(f"{args.path}/subnet_{args.k}_0.pt")

            for n, m in model.named_modules():
                if hasattr(m, 'weight'):
                    m.weight.data = state[f"{n}.weight"].data
        except FileNotFoundError as e:
            exit(f"The requested checkpoint does not exists. {e}")

    else:
        print(f"training {args.model_type} via weights updates through weights selection", args.k)
        model = model.weight_selections(args.k)
        if path.exists(f"{args.path}/init_weights_{args.k}.pt"):
            print("Loading a saved state")
            model.load_state_dict(torch.load(f"{args.path}/init_weights_{args.k}.pt"))
        else:
            print("Saving a new state to file")
            OSUtils.save_torch_object(model.state_dict(),
                                      args.path,
                                      f"init_weights_{args.k}.pt"
                                      )
            OSUtils.save_torch_object(save_slot(model, random=True),
                                      f"{args.path}",
                                      f"subnet_{args.k}_0.pt",
                                      )
    if not args.use_checkpoint:
        args.path = f"{args.path}/{args.method}"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(args.device)
    train_dl, val_dl, _ = get_dataset(args.batch_size, args.dataset)
    optimizer, scheduler = get_optimizer(model, args)
    train(model, train_dl, val_dl, args, optimizer, scheduler)


if __name__ == '__main__':
    configuration = ArgumentParser()
    configuration.add_argument("--model_type", default="conv2", type=str)
    configuration.add_argument("--method", default="prob", type=str)
    configuration.add_argument("--k", default=8, type=int)
    configuration.add_argument("--finetune_epoch", default=100, type=int)
    configuration.add_argument("--base_path", default="./outputs", type=str)
    configuration.add_argument("--seed", default=0, type=int)
    configuration.add_argument("--use_checkpoint", default=False, type=bool)

    train_schedule = schedule(configuration)
    main(train_schedule.parse_args())
