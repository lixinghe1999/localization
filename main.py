import torch
import torch.optim as optim
from torch_dataset import Main_dataset
from models.model import Model
from tqdm import tqdm

# Define your training loop
def train(model, train_loader, test_loader, optimizer, num_epochs):
    save_text = ''
    best_loss = 100
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        loss_sum = 0
        label_list, output_list = [], []

        model.train()
        for i, (binaural, labels) in enumerate(pbar):
            binaural = {k: v.to(device) for k, v in binaural.items()}
            optimizer.zero_grad()
            outputs = model(binaural)
            label_list.append(labels); output_list.append(outputs)
            loss = model.classifier.get_loss(outputs, labels)
            if i % 200 == 0:
                model.classifier.vis(outputs, labels, epoch, i)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {round(loss_sum/(i+1), 3)}')
        train_loss = loss_sum / len(train_loader)
        eval_dict = model.classifier.eval(output_list, label_list)
        print('train eval', eval_dict)
        save_text += f'Epoch {epoch}: Train eval {eval_dict}, '
        model.eval()
        label_list, output_list = [], []
        with torch.no_grad():
            for binaural, labels in (test_loader):
                binaural = {k: v.to(device) for k, v in binaural.items()}
                outputs = model(binaural)
                label_list.append(labels); output_list.append(outputs)
        eval_dict = model.classifier.eval(output_list, label_list)
        print('test eval', eval_dict)
        save_text += f'Test eval {eval_dict}\n'
        if train_loss < best_loss:
            best_loss = train_loss
            best_eval = eval_dict
            best_ckpt = model.state_dict()
    print('Best loss:', best_loss)
    print('Best eval:', best_eval)
    ckpt_name = save_folder + '/best.pth'
    torch.save(best_ckpt, ckpt_name)

    save_text += f'Best loss: {best_loss}, Best eval: {best_eval}'
    text_file = open(save_folder + '/info.txt', 'w')
    text_file.write(save_text)
    text_file.close()
    json.dump(config, open(save_folder + '/config.json', 'w'), indent=4)

def inference(model, test_loader):
    model.eval()
    label_list, output_list = [], []
    with torch.no_grad():
        for binaural, labels in tqdm(test_loader):
            binaural = {k: v.to(device) for k, v in binaural.items()}
            outputs = model(binaural)
            label_list.append(labels); output_list.append(outputs)
    eval_dict = model.classifier.eval(output_list, label_list)
    print('test eval', eval_dict)
    return eval_dict
if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/binaural_default.json')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Main_dataset(config['train_datafolder'], config)
    test_dataset = Main_dataset(config['test_datafolder'], config)
    print('train dataset {}, test dataset {}'.format(len(train_dataset), len(test_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    # Define your model, loss function, and optimizer
    model = Model(backbone_config=config['backbone'], classifier_config=config['classifier']).to(device)
    model.pretrained(config['pretrained']) # also freeze

    # Train your model
    if config['test_only']:
        inference(model, test_loader)
    else:
        import time
        import os
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        save_folder = 'ckpts/' + time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(save_folder, exist_ok=True)
        train(model, train_loader, test_loader, optimizer, num_epochs=10)