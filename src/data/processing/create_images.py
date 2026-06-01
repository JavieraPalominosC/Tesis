import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

from PIL import Image

def create_overlay_images(obj_df, config, dataset_config, name_dataset):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for band_key, j in dataset_config['all_bands'].items():
        try:
            band_data = obj_df[obj_df[dict_columns['band']] == j]
        except:
            band_data = pd.DataFrame()

        if band_data.empty:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=ax.transAxes))
        else:
            ax.errorbar(band_data[dict_columns['mjd']], 
                        band_data[dict_columns['flux']], 
                        yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                        color=fig_params['colors'][j] if name_dataset == 'elasticc_1' else fig_params['colors'][j+2],
                        fmt=fig_params['fmt'], 
                        alpha=fig_params['alpha'], 
                        markersize=fig_params['markersize'], 
                        linewidth=fig_params['linewidth'])

            ax.set_xlim(fig_params['xlim'])

        ax.set_ylim(fig_params['ylim'])
        ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image


def create_2grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    # Calcular límites reales de los datos para normalizar
    mjd_all = obj_df[dict_columns['mjd']]
    flux_all = obj_df[dict_columns['flux']]

    mjd_min = mjd_all.min()
    mjd_max = mjd_all.max()
    mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0

    flux_min = flux_all.quantile(0.01)
    flux_max = flux_all.quantile(0.99)
    flux_range = flux_max - flux_min if flux_max > flux_min else 1.0
    margin = flux_range * 0.1

    fig, axs = plt.subplots(6, 1, figsize=(fig_params['figsize']))
    for band_key, j in dataset_config['all_bands'].items():
        row = j
        try:
            band_data = obj_df[obj_df[dict_columns['band']] == band_key].copy()
        except:
            band_data = pd.DataFrame()

        if band_data.empty:
            axs[row].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row].transAxes))
        else:
            # Normalizar MJD y flux a [0, 1]
            mjd_norm  = (band_data[dict_columns['mjd']] - mjd_min) / mjd_range
            flux_norm = (band_data[dict_columns['flux']] - flux_min) / flux_range

            if config['imgs_params']['use_err']:
                flux_err_norm = band_data[dict_columns['flux_err']] / flux_range
            else:
                flux_err_norm = None

            axs[row].errorbar(mjd_norm,
                              flux_norm,
                              yerr=flux_err_norm,
                              color=fig_params['colors'][j],
                              fmt=fig_params['fmt'],
                              alpha=fig_params['alpha'],
                              markersize=fig_params['markersize'],
                              linewidth=fig_params['linewidth'])

            axs[row].set_xlim([-0.05, 1.05])
            axs[row].set_ylim([-0.15, 1.15])

        axs[row].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    rect = patches.Rectangle((0, 0.5), 1, 0, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image


def create_6grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig, axs = plt.subplots(2, 3, figsize=(fig_params['figsize']))
    for band_key, j in dataset_config['all_bands'].items():
        row, col = divmod(j, 3)
        try:
            band_data = obj_df[obj_df[dict_columns['band']] == j]
        except:
            band_data = pd.DataFrame()

        if band_data.empty:
            axs[row, col].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row, col].transAxes))
        else:
            axs[row, col].errorbar(band_data[dict_columns['mjd']], 
                                   band_data[dict_columns['flux']], 
                                   yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                                   color=fig_params['colors'][j],
                                   fmt=fig_params['fmt'], 
                                   alpha=fig_params['alpha'], 
                                   markersize=fig_params['markersize'], 
                                   linewidth=fig_params['linewidth'])

            axs[row, col].set_xlim(fig_params['xlim'])

        axs[row, col].set_ylim(fig_params['ylim'])
        axs[row, col].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for col in range(3):
        rect = patches.Rectangle((col/3, 0), 1/3, 1, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    for row in range(2):
        rect = patches.Rectangle((0, row/2), 1, 0.5, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image
