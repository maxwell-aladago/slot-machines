def evaluate(model, data_loader, criterion, device):
    model.to(device)
    model.eval()
    e_loss = e_acc = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs.float())
        loss = criterion(predictions, targets)
        e_loss += loss.item()
        e_acc += targets.eq(predictions.argmax(dim=1)).sum().item()

    e_loss /= len(data_loader)
    e_acc /= len(data_loader.dataset)

    return e_loss, e_acc
