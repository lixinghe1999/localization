import torch
import torch.optim as optim
from utils.public_dataset import STARSS23_dataset, Mobile_dataset
from utils.localization_dataset import Localization_dataset

from models.seldnet_model import SeldModel
from tqdm import tqdm
from utils.window_evaluation import Gaussian_evaluation, ACCDOA_evaluation, Multi_ACCDOA_evaluation
from utils.window_loss import ACCDOA_loss, Multi_ACCDOA_loss
import numpy as np
# Define your training loop
def train(model, train_loader, test_loader, optimizer, num_epochs):
    best_loss = 100
    text_file = open(save_folder + '/info.txt', 'w')
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        loss_sum = 0
        model.train()
        for i, (data, labels) in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Train loss: {round(loss_sum/(i+1), 5)}')
        train_loss = loss_sum / len(train_loader)
        print('train loss', train_loss)
        save_text = f'Epoch {epoch}: Train loss {train_loss}, '
        text_file.write(save_text)

        # evaluate the model
        model.eval()
        metrics = []
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for data, labels in pbar:
                outputs = model(data.to(device))
                eval_dict = evaluation(outputs.cpu().numpy(), labels.cpu().numpy())
                metrics.append(eval_dict)
                pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Test eval: {eval_dict["sed_F1"]}')

        # calculate the average distance for dict, metrics: {'precision': precision, 'recall': recall, 'F1': F1, 'distance': np.mean(distances)}    
        eval_dict = {}
        for key in metrics[0]:
            mean_metric = np.mean([m[key] for m in metrics])
            eval_dict[key] = mean_metric
        save_text = f'Test eval {eval_dict}\n'
        text_file.write(save_text)
        print('Test eval', eval_dict)
        if train_loss < best_loss:
            best_loss = train_loss
            best_eval = eval_dict
            best_ckpt = model.state_dict()
            ckpt_name = save_folder + '/best.pth'
            torch.save(best_ckpt, ckpt_name)
            print('Save best ckpt at epoch', epoch)
        last_ckpt = model.state_dict()
        ckpt_name = save_folder + '/last.pth'
        torch.save(last_ckpt, ckpt_name)

    print('Best loss:', best_loss, 'Best eval:', best_eval)
    save_text = f'Best loss: {best_loss}, Best eval: {best_eval}'
    text_file.write(save_text)
    text_file.close()
    json.dump(config, open(save_folder + '/config.json', 'w'), indent=4)

if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/smartglass.json')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(config)

    if config['dataset'] == 'mobile':
        train_dataset = Mobile_dataset(config['train_datafolder'], config)
        test_dataset = Mobile_dataset(config['test_datafolder'], config)
    elif config['dataset'] in ['earphone', 'smartglass']:
        train_dataset = Localization_dataset(config['train_datafolder'], config)
        test_dataset = Localization_dataset(config['test_datafolder'], config)

    # train_dataset._cache_('cache/train')
    # test_dataset._cache_('cache/test')
    train_dataset.cache_folder = 'cache/train'
    test_dataset.cache_folder = 'cache/test'


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    # Define your model, loss function, and optimizer
    if config['model']['name'] == 'seldnet':
        model = SeldModel(mic_channels=config['num_channel'], unique_classes=3).to(device)
    else:
        raise NotImplementedError

    if config['encoding'] == 'ACCDOA':
        criterion = ACCDOA_loss
        evaluation = ACCDOA_evaluation
    if config['encoding'] == 'Multi_ACCDOA':
        criterion = Multi_ACCDOA_loss
        evaluation = Multi_ACCDOA_evaluation

    if config['pretrained']:
        ckpt = torch.load(config['pretrained'])
        model.load_state_dict(ckpt)
        print('load pretrained model from', config['pretrained'])

    # Train your model
    if config['test_only']:
        raise NotImplementedError
    else:
        import time
        import os
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        save_folder = 'ckpts/' + time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(save_folder, exist_ok=True)
        train(model, train_loader, test_loader, optimizer, num_epochs=config['epochs'])