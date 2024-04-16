import torch
import torch.optim as optim
from torch_dataset import Main_dataset, RAW_dataset
from models.model import AudioClassifier
from tqdm import tqdm
# Define your training loop
def train(model, train_loader, test_loader, optimizer, num_epochs):
    best_loss = 100
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        loss_sum = 0
        label_list, output_list = [], []
        model.train()
        for i, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            optimizer.zero_grad()
            # model.classifier.visualize_feature(inputs, 'test.png')
            outputs = model(inputs)
            label_list.append(labels); output_list.append(outputs)
            loss = model.classifier.get_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {round(loss_sum/(i+1), 3)}')
        train_loss = loss_sum / len(train_loader)
        output_list = torch.cat(output_list, 0)
        label_list = torch.cat(label_list, 0)
        eval_dict = model.classifier.eval(output_list, label_list, single_source=True)
        print('train eval', eval_dict)
        model.eval()
        label_list, output_list = [], []
        with torch.no_grad():
            for inputs, labels in (test_loader):
                labels = labels.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(inputs)
                label_list.append(labels); output_list.append(outputs)
        output_list = torch.cat(output_list, 0)
        label_list = torch.cat(label_list, 0)
        eval_dict = model.classifier.eval(output_list, label_list, single_source=True)
        print('test eval', eval_dict)
        if train_loss < best_loss:
            best_loss = train_loss
            best_eval = eval_dict
            best_ckpt = model.state_dict()
    print('Best loss:', best_loss)
    print('Best eval:', best_eval)
    ckpt_name = save_folder + '/best.pth'
    torch.save(best_ckpt, ckpt_name)

    save_text = f'Best loss: {best_loss}, Best eval: {best_eval}'
    text_file = open(save_folder + '/info.txt', 'w')
    text_file.write(save_text)
    text_file.close()

# Example usage
if __name__ == '__main__':
    # Define your audio dataset and data loader
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true', default=False, help='use raw audio or not')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--backbone', type=str, default='DeepBSL')
    parser.add_argument('--classifier', type=str, default='DeepBSL')
    parser.add_argument('--save_folder', type=str, default='ckpts')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.raw:
        train_dataset = RAW_dataset('RAW_HRTF', 'TRAIN', args.classifier)
        test_dataset = RAW_dataset('RAW_HRTF', 'TEST', args.classifier)
    else:
        train_dataset = Main_dataset('TIMIT/HRTF_1', 'TRAIN', args.classifier)
        test_dataset = Main_dataset('TIMIT/HRTF_1', 'TEST', args.classifier)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    # Define your model, loss function, and optimizer
    model = AudioClassifier(backbone=args.backbone, classifier=args.classifier).to(device)
    if args.pretrained:
        model.pretrained() # also freeze
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    import time
    import os
    save_folder = 'ckpts/' + time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_folder, exist_ok=True)
    # Train your model
    train(model, train_loader, test_loader, optimizer, num_epochs=10)