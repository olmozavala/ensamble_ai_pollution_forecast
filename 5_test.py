import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from proj_preproc.viz import visualize_batch_data, visualize_pollution_input
from proj_io.inout import generateDateColumns
from proj_preproc.normalization import denormalize_data
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
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    total_predicted_hours = config['test']['data_loader']['auto_regresive_steps'] # 0 Is the 'next' hour, 1 is the 'next next' hour, etc.
    weather_window_size = config['data_loader']['args']['prev_weather_hours'] + config['data_loader']['args']['next_weather_hours'] + 1
    
    model_name = config['name']
    model_run = config['test']['model_path'].split('/')[-1]
    output_imgs_dir = config['test']['visualize']['output_folder']
    prediction_path = join(config['test']['prediction_path'], model_name, model_run)
    denormalization_file = config['test']['denormalization_file']
    
    # Create output dirs if they don't exist
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    pollution_column_names, pollution_column_indices = data_loader.get_pollution_column_names_and_indices("pollutant_only")
    imputed_mask_columns, imputed_mask_columns_indices = data_loader.get_pollution_column_names_and_indices("imputed_mask")
    time_related_columns, time_related_columns_indices = data_loader.get_pollution_column_names_and_indices("time")
    target_columns, target_columns_indices = data_loader.get_pollution_column_names_and_indices("target")

    contaminant_name = config['test']['visualize']['contaminant_name']
    weather_var_name = config['test']['visualize']['weather_var_name']
    weather_var_idx = config['test']['visualize']['weather_var_idx']
    prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
    next_weather_hours = config['data_loader']['args']['next_weather_hours']
    auto_regresive_steps = config['data_loader']['args']['auto_regresive_steps']

    # Find all indices that contain the pollutant name
    plot_pollutant_indices = [i for i, name in enumerate(pollution_column_names) if contaminant_name in name]

    # Initialize lists to store predictions and targets for each hour
    all_predictions = [[] for _ in range(total_predicted_hours)]
    all_targets = [[] for _ in range(total_predicted_hours)]
    all_timestamps = [[] for _ in range(total_predicted_hours)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # Print the current batch and the percentage of the data that has been processed
            print(f"Batch {batch_idx} of {len(data_loader)}")
            print(f"  Percentage of data processed: {batch_idx/len(data_loader)*100:.2f}%")

            x_pollution_data = batch[0][0].to(device)
            x_weather_data = batch[0][1].to(device)
            target = batch[1][0].to(device)
            y_mask_data = batch[1][1].to(device)
            batch_predictedtimes = pd.to_datetime(batch[2], unit='s')

            if config['test']['visualize_batch']:
                print(f"Batch {batch_idx}")
                print(f"  x pollution shape: {batch[0][0].shape} (batch, prev_pollutant_hours, stations*1(ozone) + (contaminants - 1)(means) + time related columns)")
                print(f"  x weather shape: {batch[0][1].shape} (batch, prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)")
                print(f"  y pollution shape: {batch[1][0].shape} (batch, auto_regresive_steps, stations*1(ozone) + (contaminants - 1)(means))")
                print(f"  y imputed pollution shape: {batch[1][1].shape} (batch, auto_regresive_steps, stations*1(ozone))")

                # Here we can plot the data to be sure that the data is loaded correctly
                viz_pollution_data = batch[0][0].numpy()[0,:,:]  # Final shape is (prev_pollutant_hours, stations*contaminants + time related columns)
                viz_weather_data = batch[0][1].numpy()[0,:,:,:,:]  # Final shape is (prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)
                viz_target_data = batch[1][0].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations*1(ozone) + (contaminants - 1)(means))
                viz_imputed_data = batch[1][1].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations*1(ozone))

                visualize_batch_data(viz_pollution_data, viz_target_data, viz_imputed_data, viz_weather_data, 
                             plot_pollutant_indices, pollution_column_names, target_columns, time_related_columns, time_related_columns_indices, weather_var_name, 
                             batch_predictedtimes[0], output_imgs_dir, batch_idx, prev_weather_hours, next_weather_hours, 
                             auto_regresive_steps, weather_var_idx, contaminant_name)


            predicted_outputs = []
            for predicted_hour in range(total_predicted_hours):
                # Set the current weather window input
                cur_weather_input = x_weather_data[:, predicted_hour:predicted_hour+weather_window_size, :]

                output = model(cur_weather_input, x_pollution_data)
                predicted_outputs.append(output)

                # Store predictions and targets
                all_predictions[predicted_hour].append(output.cpu().numpy())
                all_targets[predicted_hour].append(target[:, predicted_hour, :].cpu().numpy())
                all_timestamps[predicted_hour].extend(batch_predictedtimes + pd.Timedelta(hours=predicted_hour))

                if config['test']['visualize_batch']:
                    visualize_pollution_input(x_pollution_data.cpu().numpy(), 
                                              cur_weather_input.cpu().numpy(),
                                              target.cpu().numpy(),
                                              output_imgs_dir, 
                                              plot_pollutant_indices, pollution_column_names, target_columns, 
                                              time_related_columns, time_related_columns_indices,
                                              contaminant_name, 
                                              batch_predictedtimes[0].hour,
                                              predicted_hour,
                                              prev_weather_hours,
                                              next_weather_hours,
                                              auto_regresive_steps,
                                              weather_var_idx,
                                              weather_var_name,
                                              batch_idx)

                # Shift the pollution data and time related columns forward by 1 hour (dropping last hour)
                saved_data = x_pollution_data[:, 1:, :].clone()  # If I don't clone, the data is overwritten and it doesn't look right
                x_pollution_data[:, 0:-1, :] = saved_data
                x_pollution_data[:, -1, pollution_column_indices] = output

                # Generate Date columns
                new_date_columns = np.array([generateDateColumns([x], flip_order=True)[1] for x in batch_predictedtimes], dtype=np.float32).squeeze()
                # I need to validate the order of the columns so lets make all of the 0
                x_pollution_data[:, -1, time_related_columns_indices] = torch.from_numpy(new_date_columns).to(device)
                batch_predictedtimes = batch_predictedtimes + pd.Timedelta(hours=1)

            # Here we need to decide if we should compute the loss for all the predicted hours or just the last one
            y_pollution_data = target[:, predicted_hour, :].to(device)
            y_mask_data = target[:, predicted_hour, :].to(device) 
            new_target = (y_pollution_data, y_mask_data)

            # computing loss, metrics on test set
            loss = loss_fn(output, new_target)
            batch_size = data_loader.batch_size
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
                # total_metrics[i] += metric(output, new_target) * batch_size

            # Print the loss and metrics for the current batch
            print(f"Batch {batch_idx} loss: {loss.item()/batch_size:.6f}")
            # print(f"Batch {batch_idx} metrics: {total_metrics/batch_size}")

            if batch_idx > 3:
                break

    # Save predictions and targets to CSV files for each predicted hour
    for hour in range(0, total_predicted_hours):
        # Convert lists to numpy arrays
        predictions = np.vstack(all_predictions[hour])
        targets = np.vstack(all_targets[hour])
        timestamps = pd.Series(all_timestamps[hour])

        # Create DataFrames for predictions and targets
        pred_df = pd.DataFrame(predictions, columns=pollution_column_names)
        target_df = pd.DataFrame(targets, columns=pollution_column_names)

        # Denormalize the predictions and targets
        pred_df, _ = denormalize_data(denormalization_file, pred_df, None)
        target_df, _ = denormalize_data(denormalization_file, target_df, None)

        # Create final DataFrame with predictions and targets
        df = pd.DataFrame({
            'timestamp': timestamps,
            **{f'pred_{name}': pred_df[name] for name in pollution_column_names},
            **{f'target_{name}': target_df[name] for name in pollution_column_names}
        })

        # Sort by timestamp before saving
        df = df.sort_values(by='timestamp')

        # Save to CSV
        output_file = join(prediction_path, f'{model_name}_forecast_{hour+1}.csv')
        df.to_csv(output_file, index=False)
        logger.info(f'Saved predictions for hour {hour+1} to {output_file}')

        # Compute RMSE for the current hour
        rmse = np.sqrt(np.mean((pred_df.values - target_df.values) ** 2))
        logger.info(f'RMSE for hour {hour+1}: {rmse:.6f}')

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
