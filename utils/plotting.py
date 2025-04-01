import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import pandas as pd

def plot_variables(ds, variable_names, output_folder, date, n_timesteps):
    """Plot variables using cartopy with Mexico state boundaries and cmocean colormaps"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean


    date_str = date.strftime('%Y-%m-%d')
    print(F"Plotting variables for {date_str}")

    # Define colormaps for different variables
    var_colormaps = {
        'T2': cmocean.cm.thermal,
        'U10': cmocean.cm.balance,
        'V10': cmocean.cm.balance,
        'RAINC': cmocean.cm.deep,
        'RAINNC': cmocean.cm.deep,
        'SWDOWN': cmocean.cm.thermal,
        'GLW': cmocean.cm.thermal
    }

    # Use a lower resolution and specify the states feature correctly
    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor='black'
    )
    
    # Pre-compute data bounds to avoid repeated calculations
    lon_min = float(ds.lon.min()) - 0.1
    lon_max = float(ds.lon.max()) + 0.1
    lat_min = float(ds.lat.min()) - 0.1
    lat_max = float(ds.lat.max()) + 0.1
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Get number of timesteps
    n_cols = len(variable_names)
    n_rows = n_timesteps

    # Create figure with reduced DPI for faster rendering during development
    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(5*n_cols, 4*n_rows),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            dpi=100)  # Reduced from default 300 for faster plotting

    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Turn off interactive plotting
    plt.ioff()

    # for time_idx in range(n_timesteps):
    #     for var_idx, var_name in enumerate(variable_names):
    #         ax = axes[time_idx, var_idx]
            
    #         # Plot with simplified settings
    #         ds[var_name].isel(time=time_idx).plot(
    #             ax=ax,
    #             transform=ccrs.PlateCarree(),
    #             cmap=var_colormaps.get(var_name, 'viridis'),
    #             add_colorbar=True,
    #             cbar_kwargs={'fraction': 0.016, 'pad': 0.18}
    #         )
            
    #         # Add features with explicit zorder to ensure visibility
    #         ax.add_feature(states, linewidth=0.5, zorder=5)
    #         ax.coastlines(resolution='50m', zorder=5)
    #         ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            
    #         ax.set_extent(extent)
    #         cur_date = date + pd.Timedelta(hours=time_idx)
    #         time_str = cur_date.strftime('%Y-%m-%d %H:00')
    #         ax.set_title(F"{var_name} - {time_str}")

    # plt.tight_layout()
    # fig.savefig(f"{output_folder}/{date.strftime('%Y-%m-%d')}.png", 
    #             bbox_inches='tight', dpi=300)
    # plt.close(fig)

    
    print("Plotting spatial average T2 timeseries")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
    # Calculate mean over spatial dimensions for each timestep
    spatial_mean = ds['T2'].mean(dim=['lat', 'lon'])
    ax.plot(spatial_mean.time, spatial_mean.values)
    # Set the x axis ticks to be the hours
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Spatial Average Temperature - {date.strftime("%Y-%m-%d")}')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{output_folder}/{date.strftime('%Y-%m-%d')}_T2_spatial_avg_timeseries.png", 
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Done plotting {date.strftime('%Y-%m-%d')}")

