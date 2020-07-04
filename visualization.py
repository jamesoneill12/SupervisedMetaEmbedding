import matplotlib.pyplot as plt
from six import iteritems
from web.similarity import fetch_MEN, fetch_WS353,\
    fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW


def normalize(y): return (y-min(y))/(max(y)-min(y))


def get_annotations():

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "SIMLEX999": fetch_SimLex999(),
        "MTURK": fetch_MTurk(),
        "RG65": fetch_RG65(),
        "RW": fetch_RW()
    }

    xs = [x+1 for x in list(range(len(tasks)))]
    ys = {name:normalize(data.y) for name, data in iteritems(tasks)}
    names = [name.replace("999", "") for name in list(ys.keys())]

    plt.boxplot(ys.values())
    plt.xticks(xs, names,fontsize=12)
    plt.grid(True)
    plt.title("Human Annotation Distribution", fontsize=20)
    plt.ylabel("Annotation Scores", fontsize=15)

    plt.savefig('annotation_distribution.pdf', transparent=True)

    plt.show()


if __name__ == "__main__":

    get_annotations()