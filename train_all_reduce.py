import time
import torch
import pickle
import multiprocessing as mp
import matplotlib.pyplot as pyplot

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model_definition import LeNet5
from loss_definition import Criterion

BATCH_SIZE = 200
DEVICES = ['cuda:0', 'cuda:1', 'cuda:2']
NUM_PROCESS = len(DEVICES)

def save_file(object, path):
    with open(path, 'wb') as file:
        pickle.dump(object, file)

def load_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def test(model, test_loader, loss_fn, device):
    correct = 0
    total = 0
    num_batches = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            distances = model(images)
            loss = loss_fn(distances, labels)

            predictions = torch.argmin(distances.data, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            test_loss += loss.item()
            num_batches += 1

    return test_loss / num_batches, 100 * correct / total

def train(process_id, queue_to_master, queue_from_master, train_loader, test_loader, state_init, num_epochs, metrics_queue, device):
    history = {'train_loss' : [], 'test_loss' : [], 'train_acc' : [], 'test_acc' : [], 'time' : []}
    print("Process", process_id, "is using", device)

    model = LeNet5(num_classes=10, conv_type='normal')
    model.load_state_dict(state_init)
    model = model.to(device)
    loss_fn = Criterion(j=0.3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        samples = 0
        num_batches = 0
        if process_id == 0:
            start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            distances = model(images)
            loss = loss_fn(distances, labels)

            predictions = torch.argmin(distances.data, dim=1)
            correct += (predictions == labels).sum().item()
            samples += labels.shape[0]
            train_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()

            grad_list = [param.grad.clone().cpu() for param in model.parameters()]
            queue_to_master.put(grad_list)

            if process_id == 0:
                grad_lists = []
                for i in range(NUM_PROCESS):
                    grad_lists.append(queue_to_master.get())

                sum_grad_list = []
                for grads in zip(*grad_lists):
                    sum_grad = sum(grads)
                    sum_grad_list.append(sum_grad)

                for i in range(NUM_PROCESS):
                    queue_from_master.put(sum_grad_list)

            sum_grad_list = queue_from_master.get()
            average_grad_list = [sum_grad / NUM_PROCESS for sum_grad in sum_grad_list]

            for param, avg_grad in zip(model.parameters(), average_grad_list):
                param.grad = avg_grad.to(device)
            optimizer.step()
        
        local_metrics = (train_loss, correct, num_batches, samples)
        metrics_queue.put(local_metrics)

        if process_id == 0:
            all_metrics = []
            for i in range(NUM_PROCESS):
                all_metrics.append(metrics_queue.get())
            end = time.time()

            total_loss = sum([metric[0] for metric in all_metrics])
            total_correct = sum([metric[1] for metric in all_metrics])
            total_batches = sum([metric[2] for metric in all_metrics])
            total_samples = sum([metric[3] for metric in all_metrics])

            global_avg_loss = total_loss / total_batches
            global_accuracy = 100 * total_correct / total_samples

            # Test Step
            test_loss, test_acc = test(model, test_loader, loss_fn, device)
 
            print(f"Epoch {epoch+1} \tTime : {end - start} \tTraining Loss : {global_avg_loss} \tTest Loss : {test_loss}")
            print(f"Train Accuracy : {global_accuracy} \tTest Accuracy : {test_acc}")
            print()

            history['train_loss'].append(global_avg_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(global_accuracy)
            history['test_acc'].append(test_acc)
            history['time'].append(end - start)

    save_file(history, './saved/all_reduce/history_'+str(process_id)+'.pkl')
    torch.save(model.state_dict(), './saved/all_reduce/weights_'+str(process_id)+'.pth')
    queue_to_master.put((process_id, './saved/all_reduce/history_'+str(process_id)+'.pkl', './saved/all_reduce/weights_'+str(process_id)+'.pth'))

def split_dataset(dataset, k=NUM_PROCESS):
    total_size = len(dataset)
    subset_size = total_size // k
    subsets = []
    for i in range(NUM_PROCESS):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < (k-1) else total_size
        subsets.append(Subset(dataset, range(start_idx, end_idx)))
    return subsets

def make_plot(train_values, test_values, x_label, y_label, legends, title, file_name):
    pyplot.plot(train_values, '-r')
    pyplot.plot(test_values, '-b')
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(legends)
    pyplot.title(title)
    pyplot.grid(True)
    pyplot.savefig('./saved/all_reduce/' + file_name)
    pyplot.close()

def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_subsets = split_dataset(train_dataset)
    train_loaders = [DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True) for subset in train_subsets]

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mp.set_start_method('spawn', force=True)

    queue_to_master = mp.Queue()
    queue_from_master = mp.Queue()
    metrics_queue = mp.Queue()

    model = LeNet5(num_classes=10, conv_type='normal')

    processes = []
    for i in range(NUM_PROCESS):
        p = mp.Process(
            target=train,
            args=(i, queue_to_master, queue_from_master, train_loaders[i], test_loader, model.state_dict(), 100, metrics_queue, DEVICES[i])
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = []
    for i in range(NUM_PROCESS):
        process_id, history_path, state_path = queue_to_master.get()
        results.append((process_id, load_file(history_path), torch.load(state_path)))

    for i in range(1, NUM_PROCESS):
        for (k1, v1), (k2, v2) in zip(results[0][2].items(), results[i][2].items()):
            assert torch.equal(v1.cpu(), v2.cpu()), f"Model parameters differ between process 0 and {i}"

    for i in range(NUM_PROCESS):
        if results[i][0] == 0:
            history = results[i][1]
            torch.save(results[i][2], './saved/all_reduce/model.pth')
            save_file(history, './saved/all_reduce/history.pkl')

    print("Average Time taken to train one epoch of the model using ALL Reduce :", sum(history['time']) / len(history['time']))

    legends = ['Training Loss Convolution', 'Test Loss Convolution']
    make_plot(history['train_loss'], history['test_loss'], 'epoch', 'loss', legends, 'Loss vs Number of Epochs', 'Normal_Loss_vs_Epoch.png')

    legends = ['Training Accuracy Convolution', 'Test Accuracy Convolution']
    make_plot(history['train_acc'], history['test_acc'], 'epoch', 'accuracy', legends, 'Accuracy vs Number of Epochs', 'Normal_Accuracy_vs_Epoch.png')

if __name__ == "__main__":
    main()