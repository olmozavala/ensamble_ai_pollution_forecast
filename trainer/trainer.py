import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import pandas as pd
from proj_io.inout import generateDateColumns
import data_loader.data_loaders as module_data

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def init_data_loaders(self, epoch, val_batch_size=None):
        """
        Initialize the data loader for the current epoch
        """
        self.data_loader = self.config.init_obj('data_loader', module_data)

        # TODO now this is hardcoded for the project, we HAVE to use epoch-based training
        print("Epoch-based training")
        # epoch-based training
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        self.valid_data_loader = self.data_loader.split_validation(batch_size=val_batch_size)
        self.do_validation = self.valid_data_loader is not None

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.original_batch_size = self.config['data_loader']['args']['batch_size']
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Create new data loader for this epoch
        # Modify the batch size depending on the auto_regresive_steps and the number of epochs before increasing the auto_regresive_steps
        epochs_before_increase_auto_regresive_steps = self.config['trainer']['epochs_before_increase_auto_regresive_steps']
        auto_regresive_steps = self.config['trainer']['auto_regresive_steps']

        divide_training_batch_size_by = int(np.ceil(float(epoch)/epochs_before_increase_auto_regresive_steps))
        training_batch_size = int(np.ceil(self.original_batch_size / divide_training_batch_size_by))
        divide_val_batch_size_by = int(auto_regresive_steps)
        val_batch_size = int(np.ceil(self.original_batch_size / divide_val_batch_size_by))

        self.config['data_loader']['args']['batch_size'] = training_batch_size

        # Print initializing new data loader and also the batch size used
        self.logger.info(f"Initializing new data loader with training batch size: {training_batch_size} and validation batch size: {val_batch_size}")
        self.init_data_loaders(epoch, val_batch_size=val_batch_size)

        self.model.train()
        self.train_metrics.reset()
        weather_window_size = self.config['data_loader']['args']['prev_weather_hours'] + self.config['data_loader']['args']['next_weather_hours'] + 1

        time_related_columns, time_related_columns_indices = self.data_loader.get_pollution_column_names_and_indices("time")
        pollution_column_names, pollution_column_indices = self.data_loader.get_pollution_column_names_and_indices("pollutant_only")

        for batch_idx, (data, target, current_datetime) in enumerate(self.data_loader):
            x_pollution_data, x_weather_data = data
            x_pollution_data, x_weather_data = x_pollution_data.to(self.device), x_weather_data.to(self.device)
            batch_predictedtimes = pd.to_datetime(current_datetime, unit='s')
            
            self.optimizer.zero_grad()
            total_loss = 0.0
            cur_x_pollution_data = x_pollution_data.clone()
            for predicted_hour in range(min(auto_regresive_steps, int(np.ceil(epoch/epochs_before_increase_auto_regresive_steps)))):
                # Set the current weather window input
                cur_weather_input = x_weather_data[:, predicted_hour:predicted_hour+weather_window_size, :]
                
                # Get target for current step
                y_pollution_data = target[0][:, predicted_hour, :].to(self.device)
                y_mask_data = target[1][:, predicted_hour, :].to(self.device)
                new_target = (y_pollution_data, y_mask_data)

                output = self.model(cur_weather_input, cur_x_pollution_data)
                
                # Check if output contains NaN values and stop training if it does
                if torch.isnan(output).any():
                    self.logger.error(f"NaN detected in model output at epoch {epoch}, batch {batch_idx}, predicted_hour {predicted_hour}")
                    self.logger.error(f"Output shape: {output.shape}, Output: {output}")
                    raise ValueError("Training stopped due to NaN values in model output")
                

                loss = self.criterion(output, new_target)
                total_loss += loss

                # For next iteration, shift pollution data and update with prediction
                next_x_pollution_data = cur_x_pollution_data.clone()
                next_x_pollution_data[:, 0:-1, :] = cur_x_pollution_data[:, 1:, :].clone()
                next_x_pollution_data[:, -1, pollution_column_indices] = output
                # Generate Date columns
                new_date_columns = np.array([generateDateColumns([x], flip_order=True)[1] for x in batch_predictedtimes], dtype=np.float32).squeeze()
                next_x_pollution_data[:, -1, time_related_columns_indices] = torch.from_numpy(new_date_columns).to(self.device)
                batch_predictedtimes = batch_predictedtimes + pd.Timedelta(hours=1)
                cur_x_pollution_data = next_x_pollution_data

            # If batch_idx is 0 then add the computational graph to the tensorboard
            if batch_idx == 0 and epoch == 0:
                self.writer.add_graph(self.model, (cur_weather_input, cur_x_pollution_data))

            total_loss = total_loss / (predicted_hour + 1)  # Divide by number of predicted hours
            total_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, new_target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} AutoRegresiveSteps: {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    predicted_hour+1,
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

            # Step the learning rate scheduler with validation loss
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_log['loss'])
                else:
                    self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Log that the validation is starting
        self.logger.info(f"Starting validation for epoch {epoch}")
        
        self.model.eval()
        self.valid_metrics.reset()
        auto_regresive_steps = self.config['trainer']['auto_regresive_steps']
        weather_window_size = self.config['data_loader']['args']['prev_weather_hours'] + self.config['data_loader']['args']['next_weather_hours'] + 1

        time_related_columns, time_related_columns_indices = self.data_loader.get_pollution_column_names_and_indices("time")
        pollution_column_names, pollution_column_indices = self.data_loader.get_pollution_column_names_and_indices("pollutant_only")

        # Initialize accumulators for epoch-level metrics
        epoch_loss = 0.0
        epoch_metrics = {met.__name__: 0.0 for met in self.metric_ftns}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (data, target, current_datetime) in enumerate(self.valid_data_loader):
                x_pollution_data, x_weather_data = data
                x_pollution_data, x_weather_data = x_pollution_data.to(self.device), x_weather_data.to(self.device)
                batch_predictedtimes = pd.to_datetime(current_datetime, unit='s')

                total_loss = 0.0
                cur_x_pollution_data = x_pollution_data.clone()
                # In this case we always predict the 'farthest' hour (be careful with comparing between models)
                for predicted_hour in range(min(8, auto_regresive_steps)):
                    # Set the current weather window input
                    cur_weather_input = x_weather_data[:, predicted_hour:predicted_hour+weather_window_size, :]
                    
                    # Get target for current step
                    y_pollution_data = target[0][:, predicted_hour, :].to(self.device)
                    y_mask_data = target[1][:, predicted_hour, :].to(self.device)
                    new_target = (y_pollution_data, y_mask_data)

                    output = self.model(cur_weather_input, cur_x_pollution_data)
                    
                    # Check if output contains NaN values and stop training if it does
                    if torch.isnan(output).any():
                        self.logger.error(f"NaN detected in model output at epoch {epoch}, batch {batch_idx}, predicted_hour {predicted_hour}")
                        self.logger.error(f"Output shape: {output.shape}, Output: {output}")
                        raise ValueError("Training stopped due to NaN values in model output")
                    
                    loss = self.criterion(output, new_target)
                    total_loss += loss

                    # For next iteration, shift pollution data and update with prediction
                    next_x_pollution_data = cur_x_pollution_data.clone()
                    next_x_pollution_data[:, 0:-1, :] = cur_x_pollution_data[:, 1:, :].clone()
                    next_x_pollution_data[:, -1, pollution_column_indices] = output
                    # Generate Date columns
                    new_date_columns = np.array([generateDateColumns([x], flip_order=True)[1] for x in batch_predictedtimes], dtype=np.float32).squeeze()
                    next_x_pollution_data[:, -1, time_related_columns_indices] = torch.from_numpy(new_date_columns).to(self.device)
                    batch_predictedtimes = batch_predictedtimes + pd.Timedelta(hours=1)
                    cur_x_pollution_data = next_x_pollution_data

                    # TODO we need to validate that the inputs are correct for the validation

                total_loss = total_loss / (predicted_hour + 1)  # Divide by number of predicted hours
                
                # Accumulate metrics
                epoch_loss += total_loss.item()
                for met in self.metric_ftns:
                    epoch_metrics[met.__name__] += met(output, new_target)
                num_batches += 1

        # Average metrics across all batches
        epoch_loss /= num_batches
        for met_name in epoch_metrics:
            epoch_metrics[met_name] /= num_batches

        # Update metrics for logging
        self.writer.set_step(epoch, 'valid')
        self.valid_metrics.update('loss', epoch_loss)
        for met_name, value in epoch_metrics.items():
            self.valid_metrics.update(met_name, value)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)