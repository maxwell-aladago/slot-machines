from torch import argmax, sum, eye, rand_like, nn


def save_slot(model, random=False):
    """

    :param model:
    :param random:
    :return:
    """
    subnetwork = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if isinstance(module, nn.BatchNorm2d):
                subnetwork[f"{name}.mean"] = module.running_mean.data
                subnetwork[f"{name}.var"] = module.running_var.data
            else:
                basis_gen = eye(module.score.size(-1), device=module.score.device)

                if random:
                    effective_weights = sum(module.weight * basis_gen[argmax(rand_like(module.score), dim=-1)], dim=-1)
                else:
                    effective_weights = sum(module.weight * basis_gen[argmax(module.score, dim=-1)], dim=-1)

                subnetwork[f"{name}.weight"] = effective_weights

    return subnetwork
