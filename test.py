import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from os.path import join
import os

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances

    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Use the model path from test configuration
    model_path = join(config['test']['model_path'], 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(model_path))
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    total_predicted_hours = config['test']['data_loader']['auto_regresive_steps'] # 0 Is the 'next' hour, 1 is the 'next next' hour, etc.
    weather_window_size = config['data_loader']['args']['prev_weather_hours'] + config['data_loader']['args']['next_weather_hours'] + 1
    
    output_imgs_dir = config['test']['ouput_imgs_dir']
    # Create outptu dir if it doesnt' exist
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)

    pollution_column_names, pollution_column_indices = data_loader.get_pollution_column_names_and_indices("pollutant_only")
    imputed_mask_columns, imputed_mask_columns_indices = data_loader.get_pollution_column_names_and_indices("imputed_mask")
    time_related_columns, time_related_columns_indices = data_loader.get_pollution_column_names_and_indices("time")

    plot_pollutant_name = "otres"
    # Find all indices that contain the pollutant name
    plot_pollutant_column_indices = [i for i, name in enumerate(pollution_column_names) if plot_pollutant_name in name]

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(data_loader)):
            x_pollution_data, x_weather_data = data
            x_pollution_data, x_weather_data = x_pollution_data.to(device), x_weather_data.to(device)

            predicted_outputs = []
            for predicted_hour in range(total_predicted_hours):
                cur_weather_input = x_weather_data[:, predicted_hour:predicted_hour+weather_window_size, :]

                # We need to plot all the inputs to be sure that the model is working correctly
                # Create two figures to plot pollution data features
                plt.figure(figsize=(15, 8))
                plt.plot(x_pollution_data[3, :, plot_pollutant_column_indices].cpu().numpy())
                plt.title(f'{plot_pollutant_name} - Hour {predicted_hour}')
                plt.xlabel('Hour (but dont know the starting hou)')
                plt.ylabel('Feature Value') 
                plt.tight_layout()
                plt.savefig(f'{output_imgs_dir}/{batch_id}_pollution_features_1_hour_{predicted_hour}.png')
                plt.close()

                output = model(cur_weather_input, x_pollution_data)
                predicted_outputs.append(output)

                # Shift all hours forward by 1 (dropping last hour)
                x_pollution_data = x_pollution_data[:, 1:, :]
                # Add model output as new first hour
                x_pollution_data = torch.cat([predicted_outputs[predicted_hour].unsqueeze(1), x_pollution_data], dim=1)


            y_pollution_data = target[0][:, predicted_hour, :].to(device)
            y_mask_data = target[1][:, predicted_hour, :].to(device) 
            new_target = (y_pollution_data, y_mask_data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    # Change the data_loader config  with the test data_loader config
    for key, value in config['test']['data_loader'].items():
        config['data_loader']['args'][key] = value
    
    main(config)
