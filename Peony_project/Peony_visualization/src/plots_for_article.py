import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from PeonyPackage.PeonyDb import MongoDb


def visualize_auc_evolutions(
    ax,
    markevery,
    auc_seq_passive_1,
    auc_seq_passive_2,
    auc_seq_active_1,
    auc_seq_active_2,
    model_1,
    model_2,
    title,
):

    auc_1_passive_mean = np.mean(auc_seq_passive_1, axis=0)
    auc_1_passive_std = np.std(auc_seq_passive_1, axis=0)

    auc_2_passive_mean = np.mean(auc_seq_passive_2, axis=0)
    auc_2_passive_std = np.std(auc_seq_passive_2, axis=0)

    auc_1_active_mean = np.mean(auc_seq_active_1, axis=0)
    auc_1_active_std = np.std(auc_seq_active_1, axis=0)

    auc_2_active_mean = np.mean(auc_seq_active_2, axis=0)
    auc_2_active_std = np.std(auc_seq_active_2, axis=0)

    ax.grid(alpha=0.2)
    ax.plot(
        [i for i in range(200)],
        auc_1_passive_mean,
        linestyle="--",
        marker="+",
        markevery=markevery,
        color="b",
        lw=1,
        label=f"Random Selection {model_1} mean",
        alpha=0.4,
    )
    ax.plot(auc_1_passive_mean + auc_1_passive_std, linestyle="-", color="b", alpha=0.1)
    ax.plot(auc_1_passive_mean - auc_1_passive_std, linestyle="-", color="b", alpha=0.1)

    ax.plot(
        [i for i in range(200)],
        auc_2_passive_mean,
        linestyle="--",
        marker="*",
        markevery=markevery,
        color="g",
        lw=1,
        label=f"Random Selection {model_2} mean",
        alpha=0.4,
    )
    ax.plot(auc_2_passive_mean + auc_2_passive_std, linestyle="-", color="g", alpha=0.1)
    ax.plot(auc_2_passive_mean - auc_2_passive_std, linestyle="-", color="g", alpha=0.1)

    ax.plot(
        [i for i in range(200)],
        auc_1_active_mean,
        linestyle="-",
        marker="+",
        markevery=markevery,
        color="b",
        lw=1,
        label=f"Active Learning {model_1} mean",
    )
    ax.plot(auc_1_active_mean + auc_1_active_std, linestyle="-", color="b", alpha=0.1)
    ax.plot(auc_1_active_mean - auc_1_active_std, linestyle="-", color="b", alpha=0.1)

    ax.plot(
        [i for i in range(200)],
        auc_2_active_mean,
        linestyle="-",
        marker="*",
        markevery=markevery,
        color="g",
        lw=1,
        label=f"Active Learning {model_2} mean",
    )
    ax.plot(auc_2_active_mean + auc_2_active_std, linestyle="-", color="g", alpha=0.1)
    ax.plot(auc_2_active_mean - auc_2_active_std, linestyle="-", color="g", alpha=0.1)

    ax.fill_between(
        [i for i in range(200)],
        (auc_1_passive_mean + auc_1_passive_std).reshape(
            len(auc_2_active_mean),
        ),
        (auc_1_passive_mean - auc_1_passive_std).reshape(
            len(auc_2_active_mean),
        ),
        alpha=0.05,
        color="b",
    )

    ax.fill_between(
        [i for i in range(200)],
        (auc_2_passive_mean + auc_2_passive_std).reshape(
            len(auc_2_active_mean),
        ),
        (auc_2_passive_mean - auc_2_passive_std).reshape(
            len(auc_2_active_mean),
        ),
        alpha=0.05,
        color="g",
    )

    ax.fill_between(
        [i for i in range(200)],
        (auc_1_active_mean + auc_1_active_std).reshape(
            len(auc_2_active_mean),
        ),
        (auc_1_active_mean - auc_1_active_std).reshape(
            len(auc_2_active_mean),
        ),
        alpha=0.05,
        color="b",
    )

    ax.fill_between(
        [i for i in range(200)],
        (auc_2_active_mean + auc_2_active_std).reshape(
            len(auc_2_active_mean),
        ),
        (auc_2_active_mean - auc_2_active_std).reshape(
            len(auc_2_active_mean),
        ),
        alpha=0.05,
        color="g",
    )

    ax.set_xlabel("Learning Iterations", fontsize=17)
    ax.set_ylabel("AUC", fontsize=17)
    ax.set_title(
        f"{title} categories",
        fontsize=17,
    )
    ax.legend(loc="lower right", fontsize=16)

    return ax


def main():
    api = MongoDb()

    alg_1 = "bayesian_dropout_nn_fast_text_embeddings"
    alg_legend_1 = "Dropout"
    alg_legend_2 = "DEnFi"

    # alg_1 = "bayesian_dropout_hot_start_w_noise_0.3_fast_text_embeddings"
    alg_2 = "bayesian_denfi_v_2_0.3_fast_text_embeddings"
    # alg_2 = "bayesian_denfi_nn_hot_start_fast_text_embeddings"

    categories = [
        "CRIME",
        "SPORTS",
        "POLITICS",
        "TECH",
        "COLLEGE",
        "POSITIVE EMOTIONS TWEETS",
    ]
    title_categories = [
        "Crime vs Good News",
        "Sports vs Comedy",
        "Politics vs Business",
        "Tech vs Science",
        "College vs Education",
        "Positive Tweets vs Negative Tweets",
    ]

    # font = {"size": 14}

    # matplotlib.rc("font", **font)

    for index, (category, title_category) in enumerate(
        zip(categories, title_categories)
    ):
        # Random acquisition function
        random_sampling_results_1 = api.get_model_results(
            {"model": alg_1, "acquisition_function": "random", "category_1": category}
        )[0]
        random_sampling_results_2 = api.get_model_results(
            {"model": alg_2, "acquisition_function": "random", "category_1": category}
        )[0]

        # Entropy acquisition function
        entropy_sampling_results_1 = api.get_model_results(
            {"model": alg_1, "acquisition_function": "entropy", "category_1": category}
        )[0]
        entropy_sampling_results_2 = api.get_model_results(
            {"model": alg_2, "acquisition_function": "entropy", "category_1": category}
        )[0]

        ax = plt.subplot(2, 3, index + 1)
        visualize_auc_evolutions(
            ax,
            14,
            random_sampling_results_1["results"],
            random_sampling_results_2["results"],
            entropy_sampling_results_1["results"],
            entropy_sampling_results_2["results"],
            alg_legend_1,
            alg_legend_2,
            title_category,
        )

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3
    )
    plt.show()


if __name__ == "__main__":
    main()
