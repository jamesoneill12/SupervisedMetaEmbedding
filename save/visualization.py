import numpy as np
from os import listdir, getcwd
from trainers.helpers import load_vocab
import matplotlib.pyplot as plt
from statsmodels import robust


def get_results(task='udpos', epoch_lim = 40):
    results_dict = {}
    path = getcwd() + '\\' + task
    for file in listdir(path):
        if 'pkl' in file and 'None' not in file:
            if str(epoch_lim) in file:
                results_dict[file] = load_vocab(path+'/'+file)
    return results_dict


def get_ss_name(fname):
    if 'static' in fname:
        return 'static'
    elif 'linear' in fname:
        return 'linear'
    elif 'sigmoid' in fname:
        return 'sigmoid'
    elif 'very slow' in fname:
        return 'exp'


def call_colors(plt):
    return (plt.gca().set_color_cycle(['red','red','red', # 0.2,0.2
                               'green','green','green', # 0.2, 0.3
                                'blue', 'blue', 'blue', # 0.2, 0.5
                               'cyan', 'cyan', 'cyan', # 0.2, 0.8
                               'purple', 'purple', 'purple',  # 0.3, 0.2
                               'orange', 'orange', 'orange',  # 0.3, 0.3
                               'brown', 'brown', 'brown',  # 0.3, 0.5
                               'brown', 'brown', 'brown'  # 0.3, 0.8
                               ])
            )


def old_ppl(axes, x, y, result, ss, ns):
    axes[x, y].plot(result['train_epoch'], result['train_ppl'], label = 'train_'+ss+'_'+ ns)
    axes[x, y].plot(result['val_epoch'], result['val_ppl'], label='val_'+ss+'_'+ ns)
    axes[x, y].plot(result['test_epoch'], len(result['test_epoch'])*result['test_ppl'], label='test_'+ss+'_'+ ns)
    return (axes)


# should make the linewidth larger for best performing model with bold in legend too
def get_visualization(results, task='udpos', multi_plot=False,
                      save=False, cmap_choice = 'viridis',
                      latex = True, error_bars = True, alpha = 0.1):

    if task == 'ner':
        lb, ub = 0, 100
    elif task == 'udpos':
        lb, ub = 84, 94

    if multi_plot:
        embeds =  {'all': [0, 0], 'fasttext': [0, 1],
                   'glove': [1, 0], 'skipgram': [1, 1],
                   'lexvec':[2, 0], 'numberbatch':[2, 1]
                   }
        fig, axes = plt.subplots(3, 2, sharex=True)

    eval_meas = ' Accuracy'
    cmap = plt.get_cmap(cmap_choice)

    if save or latex:
        plt.rcParams['text.latex.unicode'] = True
        # plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    for i, (fname, result) in enumerate(results.items()):
        # neighbor then scheduled probability, based on filename ordering
        emb_name = fname.split("_")[1].split("embedding")[1]
        model_name = fname.split("_")[2].split("model")[1]
        color = cmap(float(i) / len(results))
        val_m = list(np.array(result['val_acc']))
        test_m  = list(np.array(result['test_acc']))


        print("{}-{}: \t val acc: {} \t test acc: {}".format(model_name, emb_name, max(val_m), max(test_m)))

        data1 = (
            # result['train_epoch'], result['train_' + meas],\
               result['val_epoch'], val_m, '-', \
        )
        data2 = (
            result['val_epoch'], len(result['val_epoch']) * (test_m), '--'
                 )

        l =  model_name + '-' + emb_name

        if multi_plot:
            x, y = embeds[emb_name]
            axes[x, y].plot(*data1, label = l, c=color)
            axes[x, y].plot(*data2, label = '', c=color)
            axes[x, y].legend(loc = 'upper left', fontsize = 'xx-small')
            axes[x, y].grid(True)
        else:

            plt.plot(*data1, label=l.replace("LSTM-","").replace("all", "meta").title(), c=color)
            plt.plot(*data2, label= '', c=color)
            plt.legend(fontsize='xx-small', bbox_to_anchor=(-0.5, 1.05), ncol=3, fancybox=True, shadow=True)

            if error_bars:
                dev = np.exp(robust.mad(data1[1], axis=0))
                y_std_lower = abs(data1[1] - dev)
                y_std_higher = abs(data1[1] + dev)
                # print(data1[0])
                # print(data1[1])
                # plt.errorbar(data1[0], data1[1], yerr=float(dev), capthick=1, c=color)
                plt.fill_between(data1[0], y_std_lower, y_std_higher, alpha=alpha)

    # plt.legend([c], ["An ellipse, not a rectangle"],})
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig('meta_results_'+task.lower()+'.pdf')

    plt.show()


if __name__ == '__main__':
    #'GnBu','winter','nipy_spectral', 'ocean'

    task = ('ner', 2) #  (udpos, 40)
    results = get_results(task[0], epoch_lim=task[1])
    get_visualization(results, task=task[0], latex=True, alpha=0.05,
                      save=False, cmap_choice='Dark2')