import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

def plot_variables(ds, variable_names, output_folder, date):
    """Plot variables using cartopy with Mexico state boundaries and cmocean colormaps"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean

    print(F"Plotting variables for {date.strftime('%Y-%m-%d')}")

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

    # Create figure with reduced DPI for faster rendering during development
    fig, axes = plt.subplots(1, len(variable_names),
                            figsize=(5*len(variable_names), 5),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            dpi=100)  # Reduced from default 300 for faster plotting

    if len(variable_names) == 1:
        axes = [axes]

    # Turn off interactive plotting
    plt.ioff()

    for var_name, ax in zip(variable_names, axes):
        # Plot with simplified settings
        ds[var_name].isel(time=0).plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=var_colormaps.get(var_name, 'viridis'),
            add_colorbar=True,
            cbar_kwargs={'fraction': 0.016, 'pad': 0.18}
        )
        
        # Add features with explicit zorder to ensure visibility
        ax.add_feature(states, linewidth=0.5, zorder=5)
        ax.coastlines(resolution='50m', zorder=5)
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        
        ax.set_extent(extent)
        ax.set_title(F"{var_name} - {date.strftime('%Y-%m-%d')}")

    plt.tight_layout()
    # Save with higher DPI for final output
    fig.savefig(f"{output_folder}/{date.strftime('%Y-%m-%d')}.png", 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Done plotting {date.strftime('%Y-%m-%d')}")

