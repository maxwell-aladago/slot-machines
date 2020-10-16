from evaluate import evaluate
from os_utils import OSUtils
from save_slot_checkpoint import save_slot

from torch import nn
from tqdm import tqdm


def train(model, train_dl, val_dl, args, optimizer, scheduler=None):
    print(f"training on {len(train_dl.dataset)}, validating on {len(val_dl.dataset)} examples.")

    criterion = nn.CrossEntropyLoss()
    loss_tr = []
    acc_tr = []
    loss_val = []
    acc_val = []

    pbar = tqdm(range(1, args.num_epochs + 1), total=args.num_epochs, leave=False)
    for i in pbar:
        tr_loss = 0
        tr_acc = 0
        for j, (inputs, targets) in enumerate(train_dl):
            model.train()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_acc += targets.eq(predictions.argmax(dim=-1)).sum().item()

        tr_acc, tr_loss = tr_acc / len(train_dl.dataset), tr_loss / len(train_dl)
        val_loss, val_acc = evaluate(model, val_dl, criterion, args.device)

        if scheduler is not None:
            scheduler.step()

        acc_tr.append(tr_acc)
        loss_tr.append(tr_loss)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        pbar.set_description(f"[{i}/{args.num_epochs}]")
        pbar.set_postfix({"tr_acc": f"{tr_acc:.2%}", "val_acc": f"{val_acc:.2%}"}, refresh=False)

        metrics = {
            "tr_loss": loss_tr,
            "tr_acc": acc_tr,
            "val_acc": acc_val,
            "val_loss": loss_val
        }

        OSUtils.save_torch_object(metrics, f"{args.path}", f"metrics_{args.k}.pt")

        if i % args.check_point_every == 0:
            if args.method == "learned":
                OSUtils.save_torch_object(model.state_dict(),
                                          f"{args.path}/checks",
                                          f"subnet_{args.k}_{i}.pt")
            else:
                OSUtils.save_torch_object(save_slot(model, random=False),
                                          f"{args.path}/checks",
                                          f"subnet_{args.k}_{i}.pt"
                                          )

    print(f"Final results tr_acc {acc_tr[-1]:.2%} val_acc {acc_val[-1]:.2%}")
