from torch import optim


def schedule(configuration):
    args = configuration.parse_args()

    if args.model_type == "lenet":
        dataset = "mnist"
        num_epochs = 120
        batch_norm = False
    elif args.model_type in ["conv2", "conv4", "conv6"]:
        batch_norm = False
        num_epochs = 140
        dataset = "cifar10"
    elif args.model_type == "vgg19":
        num_epochs = 120
        batch_norm = True
        dataset = "cifar10"
    else:
        raise ValueError(f"unknown model type {args.model_type}")

    if args.method == "greedy":
        if args.k >= 8:
            lr = 0.01
        else:
            lr = 0.1
    elif args.method == "learned":
        lr = 0.01
        num_epochs = num_epochs + 100  # account for possible fine-tuning of slot checkpoints later
    elif args.method == "prob":
        num_epochs = num_epochs + 40  # probabilistic models train slower: train for 40 more epochs
        if args.model_type == "conv6":
            lr = 25.0
        else:
            lr = 50.0
    else:
        raise ValueError(f"Unknown method {args.model_type}.")

    batch_size = 128
    check_model_every = 10

    # arguments
    configuration.add_argument("--dataset", default=dataset, type=str)
    configuration.add_argument("--bn", default=batch_norm, type=bool)
    configuration.add_argument("--lr", default=lr, type=float)
    configuration.add_argument("--batch_size", default=batch_size, type=int)
    configuration.add_argument("--num_epochs", default=num_epochs, type=int)
    configuration.add_argument("--check_point_every", default=check_model_every, type=int)
    return configuration


def get_optimizer(model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None
    if args.method == "greedy":
        if args.model_type in ["conv6", "vgg19"]:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        elif args.model_type == 'vgg19':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    elif args.model_type == "learned":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        if args.model_type in ["conv2", "conv4"]:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
        elif args.model_type == "vgg19":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

        if args.model_type == "lenet":
            scheduler = None

    return optimizer, scheduler
